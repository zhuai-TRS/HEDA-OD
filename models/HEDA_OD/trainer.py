"""HEDA-OD trainer: emp-daily cycleQuery init + cycle viz + OOM skip."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch

from common.cycle_viz import get_model_cycle_query, save_cycle_query_viz
from common.scaled_data import scaled_train_od
from common.trainer import ODTrainer
from models.HEDA_OD.args import build_setting
from models.HEDA_OD.memory_guard import (
    build_skip_report,
    clear_cuda_cache,
    estimate_paftr_mix_peak_bytes,
    get_memory_snapshot,
    peak_allocated_bytes,
    should_skip_for_memory,
)


class MemorySkipped(RuntimeError):
    """Raised when a run is skipped due to insufficient memory."""


def _empirical_daily(train_scaled: np.ndarray, cycle_len: int) -> np.ndarray:
    """Mean profile over full days: (P, N, N) in the same space as model inputs."""
    t = (train_scaled.shape[0] // cycle_len) * cycle_len
    if t < cycle_len:
        raise ValueError("train split shorter than one cycle_len")
    x = train_scaled[:t].reshape(-1, cycle_len, train_scaled.shape[1], train_scaled.shape[2])
    return x.mean(axis=0).astype(np.float32)


class HEDAODTrainer(ODTrainer):
    def __init__(self, args, model, data_provider, forward_fn=None):
        super().__init__(args, model, data_provider, forward_fn=forward_fn)
        self._init_cycle_query()

    def _init_cycle_query(self) -> None:
        """Always initialize cycleQuery from train-only empirical daily mean."""
        _, _, meta = self.data_provider(self.args, "train")
        train_scaled = scaled_train_od(meta)
        emp = _empirical_daily(train_scaled, int(self.args.cycle_len))
        with torch.no_grad():
            self.model.cycleQuery.copy_(torch.from_numpy(emp).to(self.device))
        print(
            f"cycleQuery initialized from empirical daily mean "
            f"(shape={tuple(emp.shape)}, mean={emp.mean():.4f})"
        )

    def _dense_sparse_counts(self) -> tuple[int, int]:
        n_dense = int(self.model.dense_idx.numel())
        n_sparse = int(self.model.sparse_idx.numel())
        return n_dense, n_sparse

    def _memory_context(self, setting: str) -> dict:
        n_dense, n_sparse = self._dense_sparse_counts()
        enc_in = int(getattr(self.model, "enc_in", 0))
        needed = estimate_paftr_mix_peak_bytes(
            batch_size=int(self.args.batch_size),
            seq_len=int(self.args.seq_len),
            d_model=int(self.args.d_model),
            n_heads=int(getattr(self.args, "n_heads", 4)),
            n_dense=n_dense,
            n_sparse=n_sparse,
            enc_in=enc_in,
            cycle_len=int(self.args.cycle_len),
        )
        snap = get_memory_snapshot(self.device)
        return {
            "needed": needed,
            "snap": snap,
            "n_dense": n_dense,
            "n_sparse": n_sparse,
            "setting": setting,
        }

    def preflight_memory(self, setting: str) -> None:
        ctx = self._memory_context(setting)
        if should_skip_for_memory(ctx["needed"], ctx["snap"]):
            report = build_skip_report(
                reason="preflight estimate exceeds available memory",
                needed_bytes=ctx["needed"],
                snap=ctx["snap"],
                n_dense=ctx["n_dense"],
                n_sparse=ctx["n_sparse"],
                batch_size=int(self.args.batch_size),
                setting=setting,
            )
            raise MemorySkipped(report)

    def _handle_oom(self, setting: str, err: BaseException) -> None:
        clear_cuda_cache(self.device)
        ctx = self._memory_context(setting)
        peak = peak_allocated_bytes(self.device)
        needed = max(ctx["needed"], peak)
        report = build_skip_report(
            reason="runtime out-of-memory",
            needed_bytes=needed,
            snap=get_memory_snapshot(self.device),
            n_dense=ctx["n_dense"],
            n_sparse=ctx["n_sparse"],
            batch_size=int(self.args.batch_size),
            setting=setting,
            extra=str(err),
        )
        raise MemorySkipped(report) from err

    def _is_oom(self, err: BaseException) -> bool:
        if isinstance(err, torch.cuda.OutOfMemoryError):
            return True
        msg = str(err).lower()
        return "out of memory" in msg or "cuda error: out of memory" in msg

    def train(self, setting: str):
        self.preflight_memory(setting)
        try:
            return super().train(setting)
        except Exception as err:
            if self._is_oom(err):
                self._handle_oom(setting, err)
            raise

    def evaluate(self, phase: str = "test", setting: Optional[str] = None):
        name = setting or build_setting(self.args)
        try:
            metrics = super().evaluate(phase=phase, setting=name)
        except Exception as err:
            if self._is_oom(err):
                self._handle_oom(name, err)
            raise

        if not bool(getattr(self.args, "plot_cycle", 1)):
            return metrics

        cycle_query = get_model_cycle_query(self.model)
        num_nodes = getattr(self.model, "num_nodes", None)
        if cycle_query is None or num_nodes is None:
            return metrics

        _, _, meta = self.data_provider(self.args, "train")
        train_od = meta.get("train_od")
        if train_od is None:
            return metrics

        try:
            save_cycle_query_viz(
                cycle_query,
                int(num_nodes),
                train_od,
                int(getattr(self.args, "top_k_od", 10)),
                self._results_dir(name),
                phase=phase,
                gif_fps=int(getattr(self.args, "cycle_gif_fps", 4)),
            )
        except Exception as err:
            if self._is_oom(err):
                self._handle_oom(name, err)
            raise
        return metrics

    def _results_dir(self, setting: Optional[str] = None) -> str:
        return os.path.join(self.args.results, setting or build_setting(self.args))
