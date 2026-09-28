"""Line plots of pred vs true for selected OD pairs (original count space)."""

import os
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ten OD pairs as **1-based zone / region IDs** (user-facing labels).
DEFAULT_OD_PAIR_REGION_IDS: List[Tuple[int, int]] = [
    (110, 108),
    (75, 70),
    (108, 110),
    (108, 119),
    (93, 103),
    (119, 108),
    (106, 104),
    (90, 69),
    (57, 53),
    (93, 84),
]

RegionPair = Tuple[int, int]

LINE_WIDTH = 0.9


def region_ids_to_matrix_index(region_id: int) -> int:
    """Convert 1-based region ID to 0-based matrix row/column index."""
    return int(region_id) - 1


def _flatten_od_series(
    arr: np.ndarray, origin_idx: int, dest_idx: int
) -> np.ndarray:
    """
    ``arr``: (n_samples, pred_len, N, N) -> 1d series in sample-major order.
    """
    channel = arr[:, :, origin_idx, dest_idx]
    return channel.reshape(-1).astype(np.float64)


def save_od_pair_line_plots(
    pred: np.ndarray,
    true: np.ndarray,
    out_dir: str,
    *,
    phase: str = "test",
    od_pair_region_ids: Sequence[RegionPair] = DEFAULT_OD_PAIR_REGION_IDS,
    dpi: int = 150,
) -> List[str]:
    """
    One figure per OD pair: predicted vs true along the evaluation timeline.

    ``od_pair_region_ids``: (origin_region, dest_region), **1-based** zone numbers;
    matrix indices use ``region_id - 1``.

    Returns paths of saved PNG files.
    """
    os.makedirs(out_dir, exist_ok=True)
    saved: List[str] = []
    n_points = pred.shape[0] * pred.shape[1]
    x = np.arange(n_points)
    n = pred.shape[2]

    pairs = list(od_pair_region_ids)
    valid = [
        (o, d)
        for o, d in pairs
        if 0 <= region_ids_to_matrix_index(o) < n
        and 0 <= region_ids_to_matrix_index(d) < n
    ]
    if not valid:
        # Public / smaller grids: pick first 10 diagonal-ish in-range pairs.
        valid = [(i + 1, ((i * 3) % n) + 1) for i in range(min(10, n))]

    for origin_r, dest_r in valid:
        o_idx = region_ids_to_matrix_index(origin_r)
        d_idx = region_ids_to_matrix_index(dest_r)

        pred_line = _flatten_od_series(pred, o_idx, d_idx)
        true_line = _flatten_od_series(true, o_idx, d_idx)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(
            x, true_line, "-", label="True", color="#1f77b4", linewidth=LINE_WIDTH, alpha=1.0
        )
        ax.plot(
            x, pred_line, "-", label="Pred", color="#ff7f0e", linewidth=LINE_WIDTH, alpha=1.0
        )
        ax.set_xlabel("Sample index (sample-major, all forecast steps)")
        ax.set_ylabel("OD flow (original space)")
        ax.set_title(f"[{phase}] OD pair (region {origin_r} → {dest_r})")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        fname = f"{phase}_od_{origin_r}_{dest_r}.png"
        path = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    return saved
