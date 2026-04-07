"""Three-stage differentiation Φ with Hox-like waves."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiationPhi(nn.Module):
    """
    Stages θ, α, γ with adaptive thresholds from regulator dict.
    Dropout between stages for regularization on harder datasets.
    """

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.theta = nn.Linear(dim, dim)
        self.alpha = nn.Linear(dim, dim)
        self.gamma = nn.Linear(dim, dim)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, reg: dict[str, torch.Tensor], hox_waves: bool) -> torch.Tensor:
        tau = reg["tau"].view(1, 1, 1)
        lam = reg["lambda"].view(1, 1, 1)
        k = reg["k"].view(1, 1, 1)

        h = self.theta(x)
        h = self.drop(h * torch.sigmoid(k * (h - tau)))
        if hox_waves:
            h = self.alpha(h)
            h = self.drop(h * torch.sigmoid(lam * h))
            h = self.gamma(h)
        else:
            h = F.relu(self.alpha(h))
            h = self.drop(h)
            h = F.relu(self.gamma(h))
        return h
