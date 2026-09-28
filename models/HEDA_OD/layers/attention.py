"""Full multi-head attention for dense-channel (hot) branch."""

import math

import torch
import torch.nn as nn


class FullAttention(nn.Module):
    def __init__(self, mask_flag=False, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        del attn_mask
        b, length, h, e = queries.shape
        _, s, _, d = values.shape
        scale = 1.0 / math.sqrt(e)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        out = torch.einsum("bhls,bshd->blhd", attn, values)
        if self.output_attention:
            return out.contiguous(), attn
        return out.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model: int, n_heads: int):
        super().__init__()
        d_keys = d_model // n_heads
        d_values = d_model // n_heads
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        b, length, _ = queries.shape
        h = self.n_heads
        queries = self.query_projection(queries).view(b, length, h, -1)
        keys = self.key_projection(keys).view(b, length, h, -1)
        values = self.value_projection(values).view(b, length, h, -1)
        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        return self.out_projection(out.reshape(b, length, -1)), attn
