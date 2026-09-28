"""log1p + per-OD-channel z-score (train-fit) and inverse to original OD space."""

from typing import Optional, Union

import numpy as np


class Log1pODScaler:
    """
    Per OD pair (flattened N×N channel):
      1. log1p on raw counts
      2. z-score on training timesteps with std + eps

    ``std_floor`` prevents divide-by-near-zero when a channel is constant in train
    but active later (otherwise z-scores can reach 1e7+).

    Inverse: z → log space → expm1, clipped to non-negative counts.
    """

    def __init__(self, eps: float = 1e-8, std_floor: float = 1e-2):
        self.eps = float(eps)
        self.std_floor = float(std_floor)
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.n_features_: int = 0

    def fit(self, x: np.ndarray) -> "Log1pODScaler":
        """x: (n_samples, n_features) raw OD counts."""
        log_x = np.log1p(x.astype(np.float64))
        self.mean_ = log_x.mean(axis=0)
        self.std_ = np.maximum(log_x.std(axis=0, ddof=0), self.std_floor)
        self.n_features_ = x.shape[1]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        log_x = np.log1p(x.astype(np.float64))
        z = (log_x - self.mean_) / (self.std_ + self.eps)
        return z.astype(np.float32)

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        log_x = z.astype(np.float64) * (self.std_ + self.eps) + self.mean_
        x = np.expm1(log_x)
        return np.maximum(x, 0.0).astype(np.float32)


def build_od_scaler(
    train_raw: np.ndarray,
    eps: float = 1e-8,
    std_floor: float = 1e-2,
) -> Log1pODScaler:
    """Fit on train segment ``(T, N, N)`` or ``(T, F)``."""
    flat = train_raw.reshape(train_raw.shape[0], -1)
    return Log1pODScaler(eps=eps, std_floor=std_floor).fit(flat)


def inverse_transform_od(
    arr: np.ndarray,
    scaler: Optional[Union[Log1pODScaler, object]] = None,
) -> np.ndarray:
    """Map (..., N, N) from standardized log-space back to raw OD counts."""
    if scaler is None:
        return arr.astype(np.float32)
    shape = arr.shape
    flat = arr.reshape(-1, shape[-2] * shape[-1])
    return scaler.inverse_transform(flat).reshape(shape).astype(np.float32)
