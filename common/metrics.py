"""Evaluation in original OD space (after inverse log1p)."""

from typing import Optional

import numpy as np

from .scaler_utils import Log1pODScaler, inverse_transform_od

# MAPE / mae_nonzero / rmse_nonzero share this threshold (meaningful OD flow).
MAPE_MIN_TRUE = 1.0


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mape(
    pred: np.ndarray,
    true: np.ndarray,
    min_true: float = MAPE_MIN_TRUE,
) -> float:
    """
    Mean absolute percentage error on entries with true >= min_true.

    Skips near-zero OD pairs where relative error is not meaningful.
    Returns 0.0 if no eligible entries.
    """
    mask = true >= min_true
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((pred[mask] - true[mask]) / true[mask])))


def compute_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    *,
    mape_min_true: float = MAPE_MIN_TRUE,
) -> dict:
    """
    All metrics in original count space.

    MAPE, mae_nonzero, and rmse_nonzero use the same mask: true >= mape_min_true.

    Keys: mae, rmse, mape, mae_nonzero, rmse_nonzero, active_n, mape_n, mape_min_true.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    mask_active = true >= mape_min_true
    active_n = int(mask_active.sum())
    metrics = {
        "mae": mae(pred, true),
        "rmse": rmse(pred, true),
        "mape": mape(pred, true, min_true=mape_min_true),
        "active_n": active_n,
        "mape_n": active_n,
        "mape_min_true": float(mape_min_true),
    }
    if mask_active.any():
        metrics["mae_nonzero"] = mae(pred[mask_active], true[mask_active])
        metrics["rmse_nonzero"] = rmse(pred[mask_active], true[mask_active])
    else:
        metrics["mae_nonzero"] = 0.0
        metrics["rmse_nonzero"] = 0.0
    return metrics


def evaluate_od_predictions(
    pred: np.ndarray,
    true: np.ndarray,
    scaler: Optional[Log1pODScaler] = None,
    mape_min_true: float = MAPE_MIN_TRUE,
) -> dict:
    """
    Inverse scaled preds/targets to raw counts, then MAE / RMSE / MAPE.

    ``pred`` and ``true`` are in the same space as training (standardized if scaler set).
    """
    pred_raw = inverse_transform_od(pred, scaler)
    true_raw = inverse_transform_od(true, scaler)
    return compute_metrics(pred_raw, true_raw, mape_min_true=mape_min_true)


def print_metrics(metrics: dict, prefix: str = "") -> None:
    parts = []
    for k, v in metrics.items():
        if k in ("active_n", "mape_n", "mape_min_true"):
            parts.append(f"{k}: {v}")
        else:
            parts.append(f"{k}: {v:.6f}")
    print(prefix + " | ".join(parts))
