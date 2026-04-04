"""Heritable epigenome: binary expression mask + methylation [0,1]."""

from __future__ import annotations

import torch
import torch.nn as nn

from eden.core.genome import N_CONN_GENES, N_TYPE_GENES


class HeritableEpigenome(nn.Module):
    def __init__(self, n_genes: int | None = None, drift_rate: float = 0.02) -> None:
        super().__init__()
        n = n_genes if n_genes is not None else (N_TYPE_GENES + N_CONN_GENES)
        self.n_genes = n
        self.drift_rate = drift_rate
        self.register_buffer("expression_mask", torch.ones(n, dtype=torch.bool))
        self.methylation = nn.Parameter(torch.zeros(n))

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        meth = torch.sigmoid(self.methylation)
        return self.expression_mask.float(), meth

    def drift_step(self, enabled: bool, scale: float = 1.0) -> None:
        if not enabled:
            return
        with torch.no_grad():
            noise = torch.randn_like(self.methylation) * self.drift_rate * scale
            self.methylation.add_(noise)
            flip = torch.rand_like(self.methylation) < (self.drift_rate * scale * 0.1)
            self.expression_mask ^= flip
