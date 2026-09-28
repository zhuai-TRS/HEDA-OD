"""
HEDA-OD — activity-aware dual-operator OD forecasting.

- Shared ``cycleQuery (cycle_len, N, N)`` for every OD pair (emp-daily init).
- Cold (sparse) channels: CycleNet-style cycle residual + shared FFN.
- Hot (dense) channels: frequency-complement fusion → period-conditioned
  channel attention (Q=cycle, K=fused, V=seq) → MLP prediction head.
- Allocation: train-flow mass cover α under hard budget C_max.
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.channel_split import channel_split_from_meta, register_channel_split_buffers
from models.HEDA_OD.layers.attention import AttentionLayer, FullAttention


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.num_nodes = configs.num_nodes
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.num_nodes * configs.num_nodes
        self.cycle_len = configs.cycle_len
        self.d_model = configs.d_model
        self.dropout = configs.dropout

        self.valid_fre_points = self.seq_len // 2 + 1
        self.freq_threshold = nn.Parameter(
            torch.tensor(float(getattr(configs, "threshold", 3.5)))
        )
        self.temperature = 2.0
        n = self.num_nodes
        self.cycleQuery = nn.Parameter(torch.zeros(self.cycle_len, n, n))

        attn_drop = getattr(configs, "attn_dropout", 0.5)
        self.channel_attention = AttentionLayer(
            FullAttention(mask_flag=False, attention_dropout=attn_drop),
            d_model=self.seq_len,
            n_heads=getattr(configs, "n_heads", 4),
        )

        self.sparse_ffn = nn.Sequential(
            nn.Linear(self.seq_len, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.pred_len),
        )

        self.input_proj = nn.Linear(self.seq_len, self.d_model)
        self.dense_backbone = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )
        self.output_proj = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len),
        )

        # Ablation: keep freq-fuse + MLP, skip dense FullAttention.
        self.use_dense_attention = not bool(
            getattr(configs, "mix_disable_dense_attn", 0)
        )

        self._split_ready = False

    def init_channel_split(self, meta: Dict, configs) -> None:
        """Fit static sparse/dense masks from train split (call once before training)."""
        mode = str(getattr(configs, "mix_branch_mode", "auto")).lower()
        if mode == "all_sparse":
            sparse_mask = np.ones(self.enc_in, dtype=bool)
        elif mode == "all_dense":
            sparse_mask = np.zeros(self.enc_in, dtype=bool)
        elif mode == "auto":
            sparse_mask, _, split_info = channel_split_from_meta(
                meta,
                max_dense_channels=int(getattr(configs, "mix_max_dense_channels", 2000)),
                dense_mass_cover=float(getattr(configs, "mix_dense_mass_cover", 0.80)),
            )
            self._split_info = split_info
        else:
            raise ValueError(f"Unknown mix_branch_mode: {mode}")
        if mode != "auto":
            self._split_info = {"mode": mode}
        register_channel_split_buffers(self, sparse_mask)
        self._split_ready = True

    @staticmethod
    def _to_flat(matrix: torch.Tensor) -> torch.Tensor:
        b, t, n, _ = matrix.shape
        return matrix.reshape(b, t, n * n)

    def _to_matrix(self, flat: torch.Tensor) -> torch.Tensor:
        b, y, _ = flat.shape
        return flat.reshape(b, y, self.num_nodes, self.num_nodes)

    def _gather_cycle_flat(self, cycle_index: torch.Tensor, length: int) -> torch.Tensor:
        gather_index = (
            cycle_index.view(-1, 1)
            + torch.arange(length, device=cycle_index.device).view(1, -1)
        ) % self.cycle_len
        cycle = self.cycleQuery[gather_index]
        return cycle.reshape(cycle.shape[0], cycle.shape[1], self.enc_in)

    def _freq_fuse(self, seq: torch.Tensor, cycle: torch.Tensor) -> torch.Tensor:
        """Frequency-complement fusion (locked design)."""
        freqs = torch.arange(self.valid_fre_points, device=seq.device).float()
        threshold = F.softplus(self.freq_threshold)
        cycle_weight = torch.sigmoid((threshold - freqs) / self.temperature)
        cycle_weight = cycle_weight.view(1, 1, -1)
        x_fre = torch.fft.rfft(seq, dim=-1, norm="ortho")
        cycle_fre = torch.fft.rfft(cycle, dim=-1, norm="ortho")
        y_real = cycle_weight * cycle_fre.real + (1.0 - cycle_weight) * x_fre.real
        y_imag = cycle_weight * cycle_fre.imag + (1.0 - cycle_weight) * x_fre.imag
        y = torch.complex(y_real, y_imag)
        return torch.fft.irfft(y, n=self.seq_len, dim=-1, norm="ortho")

    def _forward_sparse(
        self,
        seq_x: torch.Tensor,
        cycle_in: torch.Tensor,
        cycle_out: torch.Tensor,
    ) -> torch.Tensor:
        residual = seq_x - cycle_in
        y = self.sparse_ffn(residual.permute(0, 2, 1)).permute(0, 2, 1)
        return y + cycle_out

    def _forward_dense(
        self,
        seq_x: torch.Tensor,
        cycle_in: torch.Tensor,
    ) -> torch.Tensor:
        seq_x_t = seq_x.permute(0, 2, 1)
        cycle_x_t = cycle_in.permute(0, 2, 1)
        fused_x_t = self._freq_fuse(seq_x_t, cycle_x_t)
        if self.use_dense_attention:
            # Locked QKV: Q=cycle, K=fused, V=seq
            channel_information = self.channel_attention(
                cycle_x_t, fused_x_t, seq_x_t, attn_mask=None
            )[0]
            dense_in = seq_x_t + channel_information
        else:
            dense_in = fused_x_t
        hidden_in = self.input_proj(dense_in)
        hidden = self.dense_backbone(hidden_in)
        return self.output_proj(hidden + hidden_in).permute(0, 2, 1)

    def forward(self, seq_x: torch.Tensor, cycle_index: torch.Tensor) -> torch.Tensor:
        if not self._split_ready:
            raise RuntimeError("Call init_channel_split() before forward.")

        seq_x = self._to_flat(seq_x)
        b = seq_x.shape[0]

        cycle_in = self._gather_cycle_flat(cycle_index, self.seq_len)
        cycle_out = self._gather_cycle_flat(
            (cycle_index + self.seq_len) % self.cycle_len, self.pred_len
        )

        out = torch.zeros(
            b, self.pred_len, self.enc_in, device=seq_x.device, dtype=seq_x.dtype
        )

        if self.sparse_idx.numel() > 0:
            s_idx = self.sparse_idx
            out[:, :, s_idx] = self._forward_sparse(
                seq_x[:, :, s_idx],
                cycle_in[:, :, s_idx],
                cycle_out[:, :, s_idx],
            )

        if self.dense_idx.numel() > 0:
            d_idx = self.dense_idx
            out[:, :, d_idx] = self._forward_dense(
                seq_x[:, :, d_idx],
                cycle_in[:, :, d_idx],
            )

        return self._to_matrix(out)
