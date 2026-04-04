"""Competitive (microglia) and programmed apoptosis."""

from __future__ import annotations

import torch
import torch.nn as nn


class MicrogliaCompetition(nn.Module):
    """Prune weak nodes by masking (simulated apoptosis on node dimension)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("node_alive", torch.tensor(True))

    def forward(
        self,
        node_acts: torch.Tensor,
        threshold: torch.Tensor,
        enabled: bool,
    ) -> torch.Tensor:
        if not enabled:
            return node_acts
        mag = node_acts.abs().mean(dim=-1, keepdim=True)
        mask = (mag > threshold).float()
        return node_acts * (0.15 + 0.85 * mask)


class ProgrammedApoptosis(nn.Module):
    """Stage-based decay of low-magnitude activations."""

    def forward(self, x: torch.Tensor, theta: torch.Tensor, enabled: bool) -> torch.Tensor:
        if not enabled:
            return x
        return x * torch.sigmoid(10.0 * (x.abs().mean(dim=-1, keepdim=True) - theta))
