"""Hierarchical genome and gene regulator (L2-evolvable hyperparameters)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


N_TYPE_GENES: int = 150
N_CONN_GENES: int = 350


@dataclass
class HierarchicalGenome:
    """
    subgenome_type: 150 genes — cell-type / module biases.
    subgenome_conn: 350 genes — connectivity / mixing coefficients.
    """

    subgenome_type: torch.Tensor
    subgenome_conn: torch.Tensor

    @classmethod
    def random(cls, device: torch.device, generator: torch.Generator | None = None) -> HierarchicalGenome:
        g = generator
        t = (torch.randn(N_TYPE_GENES, generator=g) * 0.1).to(device)
        c = (torch.randn(N_CONN_GENES, generator=g) * 0.1).to(device)
        return cls(subgenome_type=t, subgenome_conn=c)

    def expressed_ratio(self, epigenome_mask: torch.Tensor) -> float:
        """Fraction of expressed (unmasked) loci."""
        n = epigenome_mask.numel()
        if n == 0:
            return 1.0
        return float(epigenome_mask.float().mean().item())


class GeneRegulator(nn.Module):
    """
    Adaptive thresholds and scales for differentiation Φ and contact inhibition.
    Eight L2-evolvable scalars (for sensitivity / Eggroll L2).
    """

    def __init__(self) -> None:
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(0.5))
        self.lambda_ = nn.Parameter(torch.tensor(0.5))
        self.k_soft = nn.Parameter(torch.tensor(1.0))
        self.rho = nn.Parameter(torch.tensor(0.85))
        self.paracrine_strength = nn.Parameter(torch.tensor(0.3))
        self.drift_rate_scale = nn.Parameter(torch.tensor(1.0))
        self.apoptosis_theta = nn.Parameter(torch.tensor(0.2))
        self.entropy_theta = nn.Parameter(torch.tensor(0.3))

    def forward(self) -> dict[str, torch.Tensor]:
        return {
            "tau": torch.sigmoid(self.tau),
            "lambda": torch.sigmoid(self.lambda_),
            "k": torch.nn.functional.softplus(self.k_soft) + 0.01,
            "rho": 0.5 + 0.45 * torch.sigmoid(self.rho),
            "paracrine_strength": torch.sigmoid(self.paracrine_strength),
            "drift_rate_scale": torch.nn.functional.softplus(self.drift_rate_scale) + 0.01,
            "apoptosis_theta": torch.sigmoid(self.apoptosis_theta),
            "entropy_theta": 0.1 + 0.4 * torch.sigmoid(self.entropy_theta),
        }

    def l2_vector(self) -> torch.Tensor:
        """Flat tensor of 8 parameters for Eggroll L2."""
        p = self.forward()
        return torch.stack(
            [
                p["tau"],
                p["lambda"],
                p["k"],
                p["rho"],
                p["paracrine_strength"],
                p["drift_rate_scale"],
                p["apoptosis_theta"],
                p["entropy_theta"],
            ]
        )


def genome_to_gates(
    genome: HierarchicalGenome,
    epigenome_mask: torch.Tensor,
    n_stems: int,
    hidden_per_stem: int,
) -> torch.Tensor:
    """Map genome + mask to stem gating weights (n_stems,) on device."""
    device = genome.subgenome_type.device
    vec = torch.cat([genome.subgenome_type, genome.subgenome_conn], dim=0)
    if epigenome_mask.shape[0] != vec.shape[0]:
        m = torch.ones_like(vec)
    else:
        m = epigenome_mask.to(vec.dtype)
    masked = vec * m
    flat = masked.flatten()
    chunk = max(1, flat.numel() // n_stems)
    buckets: list[torch.Tensor] = []
    for i in range(n_stems):
        sl = flat[i * chunk : (i + 1) * chunk]
        buckets.append(sl.mean() if sl.numel() > 0 else flat.mean())
    raw = torch.stack(buckets)
    return torch.softmax(raw, dim=0)
