"""Training split in the same space as DataLoader (log1p+z or raw)."""

from typing import Dict

import numpy as np


def scaled_train_od(meta: Dict) -> np.ndarray:
    train = meta["train_od"]
    scaler = meta.get("scaler")
    if scaler is None:
        return train.astype(np.float32)
    flat = train.reshape(train.shape[0], -1)
    scaled = scaler.transform(flat).astype(np.float32)
    return scaled.reshape(train.shape)
