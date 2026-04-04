"""Astrocyte and oligodendrocyte-inspired modulation (from v0.5 lineage)."""

from __future__ import annotations

import torch
import torch.nn as nn


class AstrocyteModulator(nn.Module):
    """Slow EMA scaling of activations (metabolic support)."""

    def __init__(self, dim: int, momentum: float = 0.99) -> None:
        super().__init__()
        self.momentum = momentum
        self.register_buffer("scale", torch.ones(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            with torch.no_grad():
                s = x.detach().abs().mean(dim=(0, 1), keepdim=True).clamp(0.1, 10.0)
                self.scale = self.momentum * self.scale + (1 - self.momentum) * s
        return x * (1.0 / (self.scale.clamp(0.1, 10.0)))


class OligodendrocyteSheath(nn.Module):
    """Light mixing across nodes (1x1 conv on node dimension)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        y = x.transpose(1, 2)
        y = self.lin(y)
        return y.transpose(1, 2)
