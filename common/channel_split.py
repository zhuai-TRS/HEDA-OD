"""Static sparse/dense channel split for flattened OD matrices (C = N*N)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch


def mass_cover_k(activity: np.ndarray, alpha: float) -> int:
    """Smallest k such that top-k activity covers >= alpha of total mass."""
    a = np.asarray(activity, dtype=np.float64).ravel()
    c = a.shape[0]
    if c == 0:
        return 0
    alpha = float(alpha)
    if alpha <= 0:
        return 0
    if alpha >= 1.0:
        return c
    order = np.argsort(a)[::-1]
    sorted_a = a[order]
    total = float(sorted_a.sum())
    if total <= 0:
        return 0
    cum = np.cumsum(sorted_a) / total
    return int(np.searchsorted(cum, alpha) + 1)


def compute_channel_split(
    train_od: np.ndarray,
    max_dense_channels: int = 2000,
    dense_mass_cover: float = 0.80,
    activity_od: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Hot set = smallest top-k by raw activity covering ``alpha`` of train mass;
    C_d = min(C_max, |hot|).
    """
    flat = train_od.reshape(train_od.shape[0], -1).astype(np.float64)
    c = flat.shape[1]
    if activity_od is not None:
        act_flat = activity_od.reshape(activity_od.shape[0], -1).astype(np.float64)
        total_activity = np.abs(act_flat).sum(axis=0)
    else:
        total_activity = np.abs(flat).sum(axis=0)

    alpha = float(dense_mass_cover)
    if alpha <= 0:
        raise ValueError("dense_mass_cover (alpha) must be > 0")

    hot_k = mass_cover_k(total_activity, alpha)
    cmax = int(max_dense_channels) if max_dense_channels is not None else 0
    budget = hot_k if cmax <= 0 else min(cmax, hot_k)
    cmax_hit = cmax > 0 and hot_k > cmax
    order = np.argsort(total_activity)[::-1]
    keep = order[:budget]
    dense = np.zeros(c, dtype=bool)
    dense[keep] = True
    sparse = ~dense
    info = {
        "mode": "mass_cover",
        "alpha": alpha,
        "hot_set_size": hot_k,
        "budget": budget,
        "cmax_hit": cmax_hit,
        "dense_mass_cover": float(
            total_activity[dense].sum() / max(total_activity.sum(), 1e-12)
        ),
    }
    return sparse, dense, info


def channel_split_from_meta(
    meta: Dict,
    max_dense_channels: int = 2000,
    dense_mass_cover: float = 0.80,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Mass-cover uses **raw** train OD for activity (no val/test leakage)."""
    raw = meta["train_od"]
    return compute_channel_split(
        raw,
        max_dense_channels=max_dense_channels,
        dense_mass_cover=dense_mass_cover,
        activity_od=raw,
    )


def split_summary(sparse_mask: np.ndarray) -> Dict[str, int]:
    c = sparse_mask.shape[0]
    n_sparse = int(sparse_mask.sum())
    return {
        "total_channels": c,
        "sparse_channels": n_sparse,
        "dense_channels": c - n_sparse,
    }


def register_channel_split_buffers(
    module: torch.nn.Module,
    sparse_mask: np.ndarray,
) -> None:
    """Register ``sparse_mask``, ``sparse_idx``, ``dense_idx`` on *module*."""
    sparse = torch.from_numpy(sparse_mask.astype(np.bool_))
    dense = ~sparse
    module.register_buffer("sparse_mask", sparse)
    module.register_buffer("sparse_idx", torch.nonzero(sparse, as_tuple=False).view(-1))
    module.register_buffer("dense_idx", torch.nonzero(dense, as_tuple=False).view(-1))
