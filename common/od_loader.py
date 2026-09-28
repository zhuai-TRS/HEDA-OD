"""Load zone OD matrix time series from structured .npy files."""

import os
from typing import Tuple

import numpy as np

DEFAULT_DATA_FILE = "zone_od_matrix_20260526.npy"


def load_od_matrix(root_path: str, data_path: str = DEFAULT_DATA_FILE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load OD matrix sequence from disk.

    Expected .npy layout (structured array):
      - timestamp: datetime64[ns], one per time step
      - data: (N, N) int/float OD counts per step

    Returns
    -------
    timestamps : (T,) datetime64
    data : (T, N, N) float32
    """
    file_path = os.path.join(root_path, data_path)
    raw = np.load(file_path, allow_pickle=True)
    if raw.dtype.names is not None and "data" in raw.dtype.names:
        timestamps = raw["timestamp"]
        sample = raw[0]["data"]
        data = np.empty((len(raw), *sample.shape), dtype=np.float32)
        for i, row in enumerate(raw):
            data[i] = row["data"]
    else:
        timestamps = np.arange(len(raw))
        data = raw.astype(np.float32)
    return timestamps, data
