"""Sliding-window dataset: X-step OD matrices in, Y-step matrices out."""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .od_loader import load_od_matrix
from .scaler_utils import Log1pODScaler, build_od_scaler


class ODMatrixDataset(Dataset):
    """
    One sample = history ``seq_x`` + forecast ``seq_y``.

    Shapes (per sample, before batching):
      seq_x : (seq_len, N, N)   — input X matrices
      seq_y : (label_len + pred_len, N, N) — target; with label_len=0 this is (pred_len, N, N)

    When scaled: log1p + per-OD z-score (train-fit, σ+ε).
    """

    def __init__(
        self,
        data: np.ndarray,
        seq_len: int,
        pred_len: int,
        border: Tuple[int, int],
        label_len: int = 0,
        scaler: Optional[Log1pODScaler] = None,
        with_prev_day: bool = False,
        full_scaled: Optional[np.ndarray] = None,
        segment_offset: int = 0,
        cycle_len: int = 96,
        global_cycle: Optional[np.ndarray] = None,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.cycle_len = cycle_len
        self.with_prev_day = with_prev_day
        self.full_scaled = full_scaled
        self.segment_offset = segment_offset

        border1, border2 = border
        segment = data[border1:border2]
        num_nodes = data.shape[1]
        if scaler is None:
            self.data = segment.astype(np.float32)
        else:
            flat = segment.reshape(segment.shape[0], -1)
            scaled = scaler.transform(flat).astype(np.float32)
            self.data = scaled.reshape(-1, num_nodes, num_nodes)

        if global_cycle is None:
            global_cycle = np.arange(len(data)) % self.cycle_len
        self.cycle_index = global_cycle[border1:border2]

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - self.pred_len + 1

    def _prev_window(self, global_begin: int, global_end: int) -> np.ndarray:
        prev_begin = global_begin - self.cycle_len
        prev_end = global_end - self.cycle_len
        if prev_begin >= 0:
            return self.full_scaled[prev_begin:prev_end]
        pad_len = -prev_begin
        out = np.zeros((self.seq_len, *self.full_scaled.shape[1:]), dtype=np.float32)
        valid = self.full_scaled[0:prev_end] if prev_end > 0 else None
        if valid is not None and len(valid) > 0:
            out[pad_len : pad_len + len(valid)] = valid
        return out

    def __getitem__(self, index: int):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        item = {
            "seq_x": torch.from_numpy(self.data[s_begin:s_end]).float(),
            "seq_y": torch.from_numpy(self.data[r_begin:r_end]).float(),
            "cycle_index": torch.tensor(self.cycle_index[s_begin], dtype=torch.long),
        }
        if self.with_prev_day and self.full_scaled is not None:
            global_begin = self.segment_offset + s_begin
            global_end = self.segment_offset + s_end
            item["prev_seq_x"] = torch.from_numpy(
                self._prev_window(global_begin, global_end)
            ).float()
            prev_r_begin = (self.segment_offset + r_begin) - self.cycle_len
            prev_r_end = (self.segment_offset + r_end) - self.cycle_len
            if prev_r_begin >= 0:
                prev_y = self.full_scaled[prev_r_begin:prev_r_end]
            else:
                pad = -prev_r_begin
                prev_y = np.zeros(
                    (self.label_len + self.pred_len, *self.full_scaled.shape[1:]),
                    dtype=np.float32,
                )
                valid_end = max(prev_r_end, 0)
                if valid_end > 0:
                    prev_y[pad : pad + valid_end] = self.full_scaled[0:valid_end]
            item["prev_seq_y"] = torch.from_numpy(prev_y).float()
        return item


def split_borders(
    total_len: int,
    seq_len: int,
    pred_len: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    split_lens: Optional[Tuple[int, int, int]] = None,
) -> Tuple[list, list]:
    """Chronological train / val / test index ranges (with seq_len overlap for val/test).

    If ``split_lens=(n_train, n_val, n_test)`` is given and sums to ``total_len``,
    use those absolute lengths (for public ODMixer-stitched series).
    """
    del pred_len  # reserved for API symmetry with callers
    if split_lens is not None:
        num_train, num_val, num_test = (int(x) for x in split_lens)
        if num_train + num_val + num_test != total_len:
            raise ValueError(
                f"split_lens {split_lens} sum != total_len {total_len}"
            )
    else:
        num_train = int(total_len * train_ratio)
        num_test = int(total_len * (1.0 - train_ratio - val_ratio))
        num_val = total_len - num_train - num_test
    border1s = [0, num_train - seq_len, total_len - num_test - seq_len]
    border2s = [num_train, num_train + num_val, total_len]
    return border1s, border2s


def _transform_full_series(
    data: np.ndarray, scaler: Log1pODScaler, chunk: int = 128
) -> np.ndarray:
    """Transform (T, N, N) in time chunks to limit memory."""
    t, n, _ = data.shape
    flat = data.reshape(t, -1)
    out = np.empty((t, flat.shape[1]), dtype=np.float32)
    for i in range(0, t, chunk):
        out[i : i + chunk] = scaler.transform(flat[i : i + chunk])
    return out.reshape(t, n, n)


def od_data_provider(args, flag: str, with_prev_day: bool = False):
    """
    Build DataLoader for train / val / test.

    Scaling (``scale=1``): log1p → per-OD z-score on train split (σ + ``scale_eps``).
    ``scale=0``: raw counts, no transform.

    Batched tensors:
      seq_x : (B, seq_len, N, N)
      seq_y : (B, label_len + pred_len, N, N)

    Evaluate with ``inverse_transform_od`` then ``compute_metrics`` in original space.
    """
    timestamps, data = load_od_matrix(args.root_path, args.data_path)
    args.num_nodes = data.shape[1]
    split_lens = None
    raw_lens = str(getattr(args, "split_lens", "") or "").strip()
    if raw_lens:
        parts = [int(x) for x in raw_lens.split(",")]
        if len(parts) != 3:
            raise ValueError(f"--split_lens needs 3 ints, got {raw_lens!r}")
        split_lens = (parts[0], parts[1], parts[2])
    border1s, border2s = split_borders(
        len(data),
        args.seq_len,
        args.pred_len,
        args.train_ratio,
        args.val_ratio,
        split_lens=split_lens,
    )
    type_map = {"train": 0, "val": 1, "test": 2}
    border = (border1s[type_map[flag]], border2s[type_map[flag]])

    train_border = (border1s[0], border2s[0])
    use_scale = bool(getattr(args, "scale", 1))
    scale_eps = float(getattr(args, "scale_eps", 1e-8))
    scale_std_floor = float(getattr(args, "scale_std_floor", 1e-2))
    scaler: Optional[Log1pODScaler] = None
    if use_scale:
        train_raw = data[train_border[0] : train_border[1]]
        scaler = build_od_scaler(
            train_raw, eps=scale_eps, std_floor=scale_std_floor
        )

    full_scaled = None
    if with_prev_day and scaler is not None:
        full_scaled = _transform_full_series(data, scaler)
    elif with_prev_day:
        full_scaled = data.astype(np.float32)

    meta: Dict[str, object] = {
        "raw_data": data,
        "train_od": data[train_border[0] : train_border[1]],
        "scaler": scaler,
        "timestamps": timestamps,
        "scale_method": "log1p_zscore" if use_scale else "none",
    }

    global_cycle = np.arange(len(data)) % args.cycle_len
    dataset = ODMatrixDataset(
        data=data,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        label_len=args.label_len,
        border=border,
        scaler=scaler,
        with_prev_day=with_prev_day,
        full_scaled=full_scaled if with_prev_day else None,
        segment_offset=border1s[type_map[flag]],
        cycle_len=args.cycle_len,
        global_cycle=global_cycle,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(flag == "train"),
        num_workers=args.num_workers,
        drop_last=False,
    )
    return dataset, loader, meta
