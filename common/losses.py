"""
Training losses for OD forecasting (standardized target space).

When ``scale=1`` (log1p + z-score), a few near-zero-variance OD channels can produce
extreme z values (~1e7+). Plain MSE then explodes on val. Mitigations:

1. ``scale_std_floor`` in :mod:`scaler_utils` (data side)
2. ``loss_z_clip`` here: clamp pred/true before loss when scale=1
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_z_clip_from_args(args) -> float:
    """Return 0 to disable clamp (e.g. raw ``scale=0`` inputs)."""
    if not bool(getattr(args, "scale", 1)):
        return 0.0
    return max(0.0, float(getattr(args, "loss_z_clip", 50.0)))


def sanitize_od_loss_tensors(
    pred: torch.Tensor,
    true: torch.Tensor,
    z_clip: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clamp to ``±z_clip`` (if >0) and build a finite-element mask."""
    if z_clip > 0:
        bound = float(z_clip)
        pred = torch.clamp(pred, -bound, bound)
        true = torch.clamp(true, -bound, bound)
    finite = torch.isfinite(pred) & torch.isfinite(true)
    return pred, true, finite


class SafeMSELoss(nn.Module):
    def __init__(self, z_clip: float = 50.0):
        super().__init__()
        self.z_clip = float(z_clip)

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        p, t, finite = sanitize_od_loss_tensors(pred, true, self.z_clip)
        if not finite.any():
            return pred.new_zeros(())
        return F.mse_loss(p[finite], t[finite])


class SafeMAELoss(nn.Module):
    def __init__(self, z_clip: float = 50.0):
        super().__init__()
        self.z_clip = float(z_clip)

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        p, t, finite = sanitize_od_loss_tensors(pred, true, self.z_clip)
        if not finite.any():
            return pred.new_zeros(())
        return torch.abs(p[finite] - t[finite]).mean()


class WeightedSparseMAELoss(nn.Module):
    def __init__(
        self,
        nonzero_weight: float = 5.0,
        zero_weight: float = 1.0,
        z_clip: float = 50.0,
    ):
        super().__init__()
        self.nonzero_weight = nonzero_weight
        self.zero_weight = zero_weight
        self.z_clip = float(z_clip)

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        p, t, finite = sanitize_od_loss_tensors(pred, true, self.z_clip)
        if not finite.any():
            return pred.new_zeros(())
        err = torch.abs(p - t)
        weight = torch.where(
            t.abs() > 1e-8,
            torch.full_like(t, self.nonzero_weight),
            torch.full_like(t, self.zero_weight),
        )
        err = err * weight
        return err[finite].mean()


class NonzeroMAELoss(nn.Module):
    def __init__(self, z_clip: float = 50.0):
        super().__init__()
        self.z_clip = float(z_clip)

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        p, t, finite = sanitize_od_loss_tensors(pred, true, self.z_clip)
        mask = finite & (t.abs() > 1e-8)
        if not mask.any():
            return pred.new_zeros(())
        return torch.abs(p[mask] - t[mask]).mean()


class MaskedMSELoss(nn.Module):
    def __init__(self, z_clip: float = 50.0):
        super().__init__()
        self.z_clip = float(z_clip)

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        p, t, finite = sanitize_od_loss_tensors(pred, true, self.z_clip)
        mask = finite & (t.abs() > 1e-8)
        if not mask.any():
            return pred.new_zeros(())
        return F.mse_loss(p[mask], t[mask])


def build_loss(args) -> nn.Module:
    name = args.loss.lower()
    z_clip = loss_z_clip_from_args(args)

    if name == "mse":
        return SafeMSELoss(z_clip) if z_clip > 0 else nn.MSELoss()
    if name == "mae":
        return SafeMAELoss(z_clip) if z_clip > 0 else nn.L1Loss()
    if name == "masked_mse":
        return MaskedMSELoss(z_clip)
    if name == "sparse_mae":
        return WeightedSparseMAELoss(
            getattr(args, "nonzero_weight", 5.0),
            getattr(args, "zero_weight", 1.0),
            z_clip=z_clip,
        )
    if name == "nonzero_mae":
        return NonzeroMAELoss(z_clip)
    raise ValueError(f"Unknown loss: {args.loss}")
