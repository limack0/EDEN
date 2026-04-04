"""Forward FLOPs estimate and training-event summaries."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import torch
import torch.nn as nn

from eden.config import TrainingState
from eden.core.genome import GeneRegulator

log = logging.getLogger(__name__)


def node_activation_redundancy(acts: torch.Tensor) -> float:
    """
    Mean off-diagonal cosine similarity between nodes, averaged over batch.
    Maps to [0, 1] as (1 + cos_mean) / 2 — higher when node representations align.
    """
    if acts.dim() != 3 or acts.size(1) < 2:
        return 0.5
    B, N, _ = acts.shape
    hn = torch.nn.functional.normalize(acts.detach(), dim=-1)
    sim = torch.bmm(hn, hn.transpose(1, 2))
    eye = torch.eye(N, device=acts.device, dtype=torch.bool).unsqueeze(0)
    off = sim.masked_select(~eye)
    off = off.view(B, -1).mean(dim=1)
    cos_mean = float(off.mean().item())
    return float(max(0.0, min(1.0, (1.0 + cos_mean) / 2.0)))


def training_event_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(e.get("kind", "unknown") for e in events))


def kaplan_meier_style_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Lightweight survival-style summary from logged events (no full KM curve).
    Counts 'death-like' vs 'growth-like' events for reporting.
    """
    kinds = training_event_counts(events)
    apoptotic = kinds.get("programmed_apoptosis", 0) + kinds.get("microglia_prune", 0)
    growth = kinds.get("neurogenesis_bump", 0) + kinds.get("neurogenesis_stagnation", 0)
    return {
        "event_counts_by_kind": kinds,
        "apoptotic_event_total": apoptotic,
        "growth_related_event_total": growth,
        "note": "Full Kaplan–Meier not implemented; use event counts as empirical survival proxy.",
    }


def estimate_forward_flops_eden(
    model: nn.Module,
    sample_x: torch.Tensor,
    regulator: GeneRegulator,
    epigenome_mask: torch.Tensor,
    training_state: TrainingState | None,
) -> int | None:
    """
    Best-effort forward-pass FLOPs (mul-add pairs) using ``thop`` if installed.
    Returns None if estimation fails.
    """
    device = sample_x.device
    reg = regulator()
    st = training_state or TrainingState()
    mask = epigenome_mask.to(device)

    class _Wrap(nn.Module):
        def __init__(self, core: nn.Module) -> None:
            super().__init__()
            self.core = core

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.core(x, reg, mask, st)
            return out.logits

    try:
        from thop import profile
    except ImportError:
        log.debug("thop not installed; skip FLOPs (pip install thop)")
        return None

    w = _Wrap(model).to(device)
    w.train(False)
    try:
        macs, _ = profile(w, inputs=(sample_x,), verbose=False)
        return int(macs * 2)
    except Exception as e:
        log.debug("FLOPs profile failed: %s", e)
        return None
