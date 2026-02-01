from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional

class TinyMemoryFormer(nn.Module):

    def __init__(self, hidden_size: int, mem_len: int, num_layers: int = 2, num_heads: int = 8,
                 dropout: float = 0.0, ff_mult: int = 4):
        super().__init__()
        self.mem_len = mem_len
        self.m_init = nn.Parameter(torch.randn(mem_len, hidden_size) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, X: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        B, _, D = X.shape
        m = self.m_init.unsqueeze(0).expand(B, -1, -1)
        inp = torch.cat([X, Q, m], dim=1)
        out = self.ln(self.encoder(inp))
        return out[:, -self.mem_len:, :]
