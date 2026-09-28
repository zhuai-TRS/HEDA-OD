"""Evaluate predictions in original OD space; save metrics, npz, and OD-pair plots."""

import os
from typing import Dict, Optional, Tuple

import numpy as np

from common.metrics import MAPE_MIN_TRUE, compute_metrics, print_metrics
from common.od_pair_plots import save_od_pair_line_plots
from common.results_io import save_metrics_txt, save_results_npz
from common.scaler_utils import Log1pODScaler, inverse_transform_od


def build_setting(args) -> str:
    return f"{args.model_id}_{args.model}_sl{args.seq_len}_pl{args.pred_len}_seed{args.random_seed}"


def to_original_space(
    pred: np.ndarray,
    true: np.ndarray,
    scaler: Optional[Log1pODScaler],
) -> Tuple[np.ndarray, np.ndarray]:
    return (
        inverse_transform_od(pred, scaler),
        inverse_transform_od(true, scaler),
    )


def save_phase_results(
    args,
    phase: str,
    pred_raw: np.ndarray,
    true_raw: np.ndarray,
    metrics: Dict[str, float],
    setting: Optional[str] = None,
) -> str:
    out_dir = os.path.join(args.results, setting or build_setting(args))
    os.makedirs(out_dir, exist_ok=True)
    save_metrics_txt(os.path.join(out_dir, f"{phase}_metrics.txt"), phase, metrics)
    # save_results_npz(os.path.join(out_dir, f"{phase}_results.npz"), pred_raw, true_raw, metrics)

    if bool(getattr(args, "plot_od_pairs", 1)):
        plot_dir = os.path.join(out_dir, f"{phase}_od_pair_plots")
        paths = save_od_pair_line_plots(pred_raw, true_raw, plot_dir, phase=phase)
        print(f"Saved {len(paths)} OD-pair line plots to {plot_dir}")

    return out_dir


def run_phase_evaluation(
    args,
    pred: np.ndarray,
    true: np.ndarray,
    scaler: Optional[Log1pODScaler],
    phase: str,
    setting: Optional[str] = None,
) -> Dict[str, float]:
    pred_raw, true_raw = to_original_space(pred, true, scaler)
    mape_min = float(getattr(args, "mape_min_true", MAPE_MIN_TRUE))
    metrics = compute_metrics(pred_raw, true_raw, mape_min_true=mape_min)
    print_metrics(metrics, prefix=f"[{phase}] original space")
    out_dir = save_phase_results(args, phase, pred_raw, true_raw, metrics, setting=setting)
    print(f"Saved metrics and npz under {out_dir}")
    return metrics
