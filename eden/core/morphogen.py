"""Local morphogen field (15-dim) and k-NN paracrine cascade (k=3)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MORPHOGEN_DIM: int = 15


def activation_entropy(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    p = F.softmax(x, dim=dim)
    logp = torch.log(p + eps)
    return -(p * logp).sum(dim=dim)


class LocalMorphogen(nn.Module):
    """
    15-dim: local mean/std of activations (4), neighbour mean/std (4),
    position encoding (2), global loss/accuracy proxy (2), padding to 15.
    """

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(MORPHOGEN_DIM, MORPHOGEN_DIM)

    def compute(
        self,
        node_acts: torch.Tensor,
        global_loss: float,
        global_acc: float,
    ) -> torch.Tensor:
        """
        node_acts: (B, N, D) batch, nodes, dim.
        Returns (B, N, 15).
        """
        B, N, D = node_acts.shape
        mean = node_acts.mean(dim=-1, keepdim=True)
        std = node_acts.std(dim=-1, keepdim=True)
        neigh = torch.roll(node_acts, 1, dims=1) + torch.roll(node_acts, -1, dims=1)
        nmean = neigh.mean(dim=-1, keepdim=True)
        nstd = neigh.std(dim=-1, keepdim=True)
        pos = torch.linspace(-1.0, 1.0, N, device=node_acts.device).view(1, N, 1).expand(B, -1, -1)
        gl = torch.full((B, N, 1), global_loss, device=node_acts.device)
        ga = torch.full((B, N, 1), global_acc, device=node_acts.device)
        pad = torch.zeros(B, N, 4, device=node_acts.device)
        raw = torch.cat([mean, std, nmean, nstd, pos, gl, ga, pad], dim=-1)
        if raw.shape[-1] > MORPHOGEN_DIM:
            raw = raw[..., :MORPHOGEN_DIM]
        elif raw.shape[-1] < MORPHOGEN_DIM:
            pad2 = torch.zeros(B, N, MORPHOGEN_DIM - raw.shape[-1], device=node_acts.device)
            raw = torch.cat([raw, pad2], dim=-1)
        return torch.tanh(self.proj(raw))


class ParacrineCascade(nn.Module):
    """
    Mix each node with k=3 nearest neighbours in feature space (per batch).
    Uses O(N^2 * D) pairwise distances only for small N (node count), not dataset size squared.
    Batched samples can be further vectorized with torch.vmap over the batch dimension if needed.
    """

    def __init__(self, k: int = 3) -> None:
        super().__init__()
        self.k = k

    def forward(self, features: torch.Tensor, strength: torch.Tensor) -> torch.Tensor:
        """
        features: (B, N, D)
        strength: scalar tensor in [0,1]
        """
        B, N, D = features.shape
        if N <= 1:
            return features
        k = min(self.k, N - 1)
        x = features.detach()
        dist = torch.cdist(x, x)
        eye = torch.eye(N, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1, -1)
        dist = dist + eye * 1e6
        _, idx = dist.topk(k, dim=-1, largest=False)
        gathered = torch.gather(
            x.unsqueeze(2).expand(B, N, N, D),
            2,
            idx.unsqueeze(-1).expand(B, N, k, D),
        )
        mix = gathered.mean(dim=2)
        s = strength.clamp(0.0, 1.0)
        return (1.0 - s) * features + s * mix


def cosine_similarity_matrix(x: torch.Tensor) -> torch.Tensor:
    """x: (N, D) -> (N, N) cosine similarity."""
    xn = F.normalize(x, dim=-1)
    return xn @ xn.T
