"""Stem perceptrons and stem pool (top-k retention)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StemPerceptron(nn.Module):
    """Standard stem with configurable activation (default ReLU)."""

    def __init__(self, in_dim: int, hidden: int, activation: nn.Module | None = None) -> None:
        super().__init__()
        act = activation if activation is not None else nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            act,
            nn.Linear(hidden, hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StemPerceptronSkip(nn.Module):
    """Stem with residual skip connection from input to output."""

    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.proj = nn.Linear(in_dim, hidden) if in_dim != hidden else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        return self.fc2(h) + self.proj(x)


_STEM_ACTIVATIONS: list[nn.Module | None] = [
    nn.ReLU(inplace=True),
    nn.GELU(),
    nn.SiLU(),
    None,  # sentinel → StemPerceptronSkip
]


class StemPool(nn.Module):
    """
    Fixed initial stem count; optional mitosis up to ``max_stems``.
    Every `retention_interval` epochs keep top `keep` by score.
    During training, forward uses softmax weights over stem outputs.

    When ``heterogeneous=True`` the four initial stems use different activations
    (ReLU, GELU, SiLU, skip-connection), encouraging diverse representations.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        n_stems: int = 4,
        keep: int = 2,
        retention_interval: int = 10,
        max_stems: int = 12,
        heterogeneous: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.stem_hidden = hidden
        self.keep = keep
        self.retention_interval = retention_interval
        self.max_stems = max(1, max_stems)
        self.heterogeneous = heterogeneous
        n0 = min(n_stems, self.max_stems)
        self.stems = nn.ModuleList(self._make_stem(i) for i in range(n0))

        self.register_buffer("stem_scores", torch.zeros(len(self.stems)))
        self.register_buffer("active_mask", torch.ones(len(self.stems), dtype=torch.bool))

    def _make_stem(self, idx: int) -> nn.Module:
        if not self.heterogeneous:
            return StemPerceptron(self.in_dim, self.stem_hidden)
        act = _STEM_ACTIVATIONS[idx % len(_STEM_ACTIVATIONS)]
        if act is None:
            return StemPerceptronSkip(self.in_dim, self.stem_hidden)
        return StemPerceptron(self.in_dim, self.stem_hidden, activation=act)

    @property
    def n_stems(self) -> int:
        return len(self.stems)

    def update_scores(self, scores: torch.Tensor) -> None:
        self.stem_scores = scores.detach().clone()

    def apply_retention(self, epoch: int) -> None:
        if epoch <= 0 or epoch % self.retention_interval != 0:
            return
        ns = self.n_stems
        k = min(self.keep, ns)
        _, top = self.stem_scores.topk(k)
        mask = torch.zeros(ns, dtype=torch.bool, device=self.stem_scores.device)
        mask[top] = True
        self.active_mask = mask

    def try_mitosis_stem(self, noise: float = 0.02) -> list[nn.Parameter]:
        """
        Append one stem by copying the highest-scoring stem with Gaussian noise (cell-like division).
        Returns new parameters for registering with the optimizer.
        """
        if self.n_stems >= self.max_stems:
            return []
        device = self.stem_scores.device
        parent_idx = int(self.stem_scores.argmax().item()) if self.stem_scores.numel() else 0
        parent_idx = max(0, min(parent_idx, self.n_stems - 1))
        child = self._make_stem(self.n_stems)
        child.to(device=device, dtype=next(self.stems[parent_idx].parameters()).dtype)
        with torch.no_grad():
            for (n, cp), (_, pp) in zip(child.named_parameters(), self.stems[parent_idx].named_parameters()):
                cp.copy_(pp + torch.randn_like(cp) * noise)
        self.stems.append(child)
        ns = self.n_stems
        new_scores = torch.zeros(ns, device=device, dtype=self.stem_scores.dtype)
        if self.stem_scores.numel():
            new_scores[:-1].copy_(self.stem_scores)
        self.stem_scores = new_scores
        new_mask = torch.ones(ns, dtype=torch.bool, device=device)
        if self.active_mask.numel():
            new_mask[:-1].copy_(self.active_mask)
        self.active_mask = new_mask
        return list(child.parameters())

    def forward(self, x: torch.Tensor, stem_weights: torch.Tensor | None = None) -> torch.Tensor:
        outs = torch.stack([s(x) for s in self.stems], dim=1)
        if stem_weights is None:
            w = self.active_mask.float()
            w = w / (w.sum() + 1e-8)
            w = w.view(1, -1, 1)
        else:
            sw = stem_weights[: self.n_stems].clone()
            sw = sw * self.active_mask.to(sw.dtype)
            s = sw.sum()
            if s < 1e-6:
                sw = self.active_mask.to(sw.dtype)
                s = sw.sum() + 1e-8
            sw = sw / s
            w = sw.view(1, -1, 1)
        return (outs * w).sum(dim=1)
