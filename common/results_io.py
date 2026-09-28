"""Save evaluation outputs (metrics text, npz)."""

import os
from typing import Dict

import numpy as np


def save_metrics_txt(
    path: str,
    phase: str,
    metrics: Dict[str, float],
    *,
    metric_space: str = "original",
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"phase: {phase}\n")
        f.write(f"metric_space: {metric_space}\n")
        if "mape_min_true" in metrics:
            f.write(f"mape_min_true: {metrics['mape_min_true']}\n")
        f.write("-" * 40 + "\n")
        for key, value in metrics.items():
            if key in ("active_n", "mape_n", "mape_min_true"):
                f.write(f"{key}: {value}\n")
            else:
                f.write(f"{key}: {value:.6f}\n")


def save_results_npz(
    path: str,
    pred: np.ndarray,
    true: np.ndarray,
    metrics: Dict[str, float],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(
        path,
        pred=pred,
        true=true,
        metric_space="original",
        **{
            k: v
            for k, v in metrics.items()
            if k not in ("active_n", "mape_n")
        },
        active_n=metrics.get("active_n", metrics.get("mape_n", 0)),
        mape_n=metrics.get("mape_n", metrics.get("active_n", 0)),
    )
