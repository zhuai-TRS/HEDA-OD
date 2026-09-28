"""OD matrix tensor ops (top-K active pairs for cycle_query visualization)."""

import numpy as np


def compute_topk_flat_indices(train_od: np.ndarray, top_k: int) -> np.ndarray:
    """Select globally most active OD pairs from training matrices (T, N, N)."""
    activity = train_od.sum(axis=0).reshape(-1)
    k = min(top_k, activity.shape[0])
    return np.argsort(activity)[::-1][:k].astype(np.int64)
