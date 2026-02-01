from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple


class QueryBuilder(nn.Module):

    def __init__(self, hidden_size: int, query_len: int, num_layers: int = 2, num_heads: int = 8,
                 dropout: float = 0.0, ff_mult: int = 4, max_len: int = 2048):
        super().__init__()
        self.hidden_size = hidden_size
        self.query_len = query_len
        self.q_init = nn.Parameter(torch.randn(query_len, hidden_size) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, hidden_size) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, H: torch.Tensor, H_key_padding_mask: torch.BoolTensor | None = None) -> torch.Tensor:

        B, L, D = H.shape
        q = self.q_init.unsqueeze(0).expand(B, -1, -1)  # (B,K,D)
        x = torch.cat([H, q], dim=1)  # (B, L+K, D)
        if x.size(1) > self.pos_emb.size(1):
            raise ValueError(f"Sequence too long for pos_emb: {x.size(1)} > {self.pos_emb.size(1)}. Increase max_len.")
        x = x + self.pos_emb[:, : x.size(1), :]


        total = L + self.query_len
        attn_mask = torch.zeros(total, total, device=H.device, dtype=torch.float32)
        attn_mask[:L, L:] = float("-inf")

        src_kpm = None
        if H_key_padding_mask is not None:
            q_kpm = torch.zeros((B, self.query_len), device=H.device, dtype=torch.bool)
            src_kpm = torch.cat([H_key_padding_mask, q_kpm], dim=1)
        out = self.encoder(x, mask=attn_mask, src_key_padding_mask=src_kpm)
        out = self.ln(out)
        Q = out[:, -self.query_len:, :]
        return Q
