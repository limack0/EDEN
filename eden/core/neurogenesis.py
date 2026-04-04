"""Local stagnation detection and structural neurogenesis (mitosis-style growth)."""

from __future__ import annotations

import torch
import torch.nn as nn

from eden.core.stem import StemPool


class LocalStagnationDetector:
    def __init__(self, grad_thresh: float = 1e-4, patience_batches: int = 20) -> None:
        self.grad_thresh = grad_thresh
        self.patience_batches = patience_batches
        self._bad = 0

    def observe(self, grad_norm: float) -> bool:
        if grad_norm < self.grad_thresh:
            self._bad += 1
        else:
            self._bad = 0
        return self._bad >= self.patience_batches


def _max_pathway_nodes(model: nn.Module) -> int:
    return int(getattr(model, "max_pathway_nodes", 24))


def apply_pathway_mitosis(model: nn.Module, noise: float = 0.03) -> list[nn.Parameter]:
    """
    Add one pathway ``cell`` (node): widen ``node_proj`` from ``n * hidden`` to ``(n+1) * hidden``,
    initializing the new block from a random parent node plus noise.
    """
    if not hasattr(model, "n_nodes") or not hasattr(model, "node_proj"):
        return []
    max_n = _max_pathway_nodes(model)
    if model.n_nodes >= max_n:
        return []
    old = model.node_proj
    if not isinstance(old, nn.Linear):
        return []
    H = int(model.hidden)
    n = int(model.n_nodes)
    device, dtype = old.weight.device, old.weight.dtype
    new_out = (n + 1) * H
    has_bias = old.bias is not None
    new_lin = nn.Linear(H, new_out, bias=has_bias, device=device, dtype=dtype)
    with torch.no_grad():
        new_lin.weight[: n * H].copy_(old.weight)
        if has_bias and old.bias is not None:
            new_lin.bias[: n * H].copy_(old.bias)
        parent = int(torch.randint(0, n, (1,), device=device).item())
        src_w = old.weight[parent * H : (parent + 1) * H]
        new_lin.weight[n * H : (n + 1) * H].copy_(src_w + torch.randn_like(src_w) * noise)
        if has_bias and old.bias is not None and new_lin.bias is not None:
            src_b = old.bias[parent * H : (parent + 1) * H]
            new_lin.bias[n * H : (n + 1) * H].copy_(src_b + torch.randn(H, device=device, dtype=dtype) * noise)
    model.n_nodes = n + 1
    model.node_proj = new_lin
    model.register_module("node_proj", new_lin)
    return list(new_lin.parameters())


def apply_neurogenesis_structural_growth(
    model: nn.Module,
    *,
    pathway_noise: float = 0.03,
    stem_noise: float = 0.02,
) -> tuple[list[nn.Parameter], str | None]:
    """
    On stagnation: grow the graph if below caps — first an extra pathway node, else an extra stem.
    Returns (new_parameters_for_optimizer, event_kind or None).
    """
    new_p = apply_pathway_mitosis(model, noise=pathway_noise)
    if new_p:
        return new_p, "neurogenesis_pathway_mitosis"
    pool = getattr(model, "stem_pool", None)
    if isinstance(pool, StemPool):
        new_p = pool.try_mitosis_stem(noise=stem_noise)
        if new_p:
            return new_p, "neurogenesis_stem_mitosis"
    return [], None


def apply_neurogenesis_exploration_bump(model: nn.Module, scale: float = 0.02) -> None:
    """Fallback when the network is already at structural caps: noise on ``node_proj``."""
    if not hasattr(model, "node_proj"):
        return
    lp = model.node_proj
    if not isinstance(lp, torch.nn.Linear):
        return
    with torch.no_grad():
        lp.weight.add_(torch.randn_like(lp.weight) * scale)
        if lp.bias is not None:
            lp.bias.add_(torch.randn_like(lp.bias) * scale)
