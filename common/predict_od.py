"""Collect model predictions on an OD DataLoader."""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def acquire_device(args) -> torch.device:
    if getattr(args, "use_gpu", 1) and torch.cuda.is_available():
        return torch.device(f"cuda:{args.gpu}")
    return torch.device("cpu")


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pred_len: int,
    forward_fn=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    preds, trues : (N_samples, pred_len, N, N) in standardized / raw space (same as loader)
    """
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            seq_y = batch["seq_y"].float().to(device)
            if forward_fn is not None:
                outputs = forward_fn(batch)
            else:
                seq_x = batch["seq_x"].float().to(device)
                cycle_index = batch["cycle_index"].to(device)
                outputs = model(seq_x, cycle_index)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            outputs = outputs[:, -pred_len:, :, :]
            target = seq_y[:, -pred_len:, :, :]
            preds.append(outputs.cpu().numpy())
            trues.append(target.cpu().numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)
