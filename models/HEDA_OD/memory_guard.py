"""Memory estimation and OOM skip reporting for HEDA_OD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class MemorySnapshot:
    device: str
    total_bytes: int
    available_bytes: int
    allocated_bytes: int = 0
    reserved_bytes: int = 0

    @property
    def total_gib(self) -> float:
        return self.total_bytes / (1024**3)

    @property
    def available_gib(self) -> float:
        return self.available_bytes / (1024**3)

    @property
    def allocated_gib(self) -> float:
        return self.allocated_bytes / (1024**3)


def format_gib(num_bytes: float) -> str:
    return f"{num_bytes / (1024**3):.2f} GiB"


def get_memory_snapshot(device: torch.device) -> MemorySnapshot:
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        props = torch.cuda.get_device_properties(idx)
        total = int(props.total_memory)
        allocated = int(torch.cuda.memory_allocated(idx))
        reserved = int(torch.cuda.memory_reserved(idx))
        available = max(0, total - reserved)
        return MemorySnapshot(
            device=str(device),
            total_bytes=total,
            available_bytes=available,
            allocated_bytes=allocated,
            reserved_bytes=reserved,
        )

    total, available = _linux_mem_total_available()
    return MemorySnapshot(
        device="cpu",
        total_bytes=total,
        available_bytes=available,
    )


def _linux_mem_total_available() -> tuple[int, int]:
    info: dict[str, int] = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                key, rest = line.split(":", 1)
                info[key.strip()] = int(rest.split()[0]) * 1024
    except OSError:
        return 0, 0
    total = info.get("MemTotal", 0)
    available = info.get("MemAvailable", info.get("MemFree", 0))
    return total, available


def estimate_paftr_mix_peak_bytes(
    *,
    batch_size: int,
    seq_len: int,
    d_model: int,
    n_heads: int,
    n_dense: int,
    n_sparse: int,
    enc_in: int,
    cycle_len: int,
) -> int:
    """
    Rough peak memory for one HEDA_OD train step (forward + backward).

    Dominant term: dense-branch global attention ``(B, H, C_dense, C_dense)``.
    """
    b = max(1, batch_size)
    cd = max(0, n_dense)
    cs = max(0, n_sparse)

    dense_attn_scores = b * n_heads * cd * cd * 4
    dense_attn_maps = b * n_heads * cd * cd * 4
    dense_qkv = 3 * b * cd * seq_len * 4
    dense_ffn = b * cd * (seq_len + d_model) * 4 * 4

    sparse_path = b * cs * seq_len * 4 * 3
    cycle_buffers = b * enc_in * (seq_len + 1) * 4 * 2
    cycle_params = enc_in * cycle_len * 4

    forward_bytes = (
        dense_attn_scores
        + dense_attn_maps
        + dense_qkv
        + dense_ffn
        + sparse_path
        + cycle_buffers
    )
    # activations + gradients + optimizer state (Adam ~2x params for dense layers)
    return int(forward_bytes * 2.5 + cycle_params * 3)


def build_skip_report(
    *,
    reason: str,
    needed_bytes: int,
    snap: MemorySnapshot,
    n_dense: int,
    n_sparse: int,
    batch_size: int,
    setting: str,
    extra: Optional[str] = None,
) -> str:
    lines = [
        "=" * 60,
        f"[MEMORY SKIP] {reason}",
        f"  setting: {setting}",
        f"  device: {snap.device}",
        f"  estimated peak needed: {format_gib(needed_bytes)}",
        f"  available now: {format_gib(snap.available_bytes)} "
        f"(total {format_gib(snap.total_bytes)})",
        f"  dense channels: {n_dense} | sparse channels: {n_sparse} | batch_size: {batch_size}",
    ]
    if snap.allocated_bytes > 0:
        lines.append(
            f"  already allocated on device: {format_gib(snap.allocated_bytes)}"
        )
    if extra:
        lines.append(f"  detail: {extra}")
    lines.extend(
        [
            "  action: skip this run and continue.",
            "  hint: lower --batch_size, --mix_max_dense_channels, or avoid --mix_branch_mode all_dense",
            "=" * 60,
        ]
    )
    return "\n".join(lines)


def should_skip_for_memory(
    needed_bytes: int,
    snap: MemorySnapshot,
    safety_ratio: float = 0.90,
) -> bool:
    if snap.available_bytes <= 0:
        return needed_bytes > 0
    return needed_bytes > snap.available_bytes * safety_ratio


def clear_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        if device.index is not None:
            torch.cuda.reset_peak_memory_stats(device.index)
        else:
            torch.cuda.reset_peak_memory_stats()


def peak_allocated_bytes(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    idx = device.index if device.index is not None else 0
    return int(torch.cuda.max_memory_allocated(idx))
