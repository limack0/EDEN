"""L1 Eggroll: antithetic ES on differentiation-related weights (every N batches)."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn


def eggroll_l1_step(
    model: nn.Module,
    get_loss: Callable[[], float],
    param_filter: Callable[[str, nn.Parameter], bool],
    lr: float = 0.001,
    sigma: float = 0.01,
    n_pairs: int = 20,
) -> None:
    """Antithetic finite-difference ES: accumulate n_pairs gradient estimates."""
    noise_targets: list[tuple[nn.Parameter, torch.Tensor]] = []
    for name, p in model.named_parameters():
        if p.requires_grad and param_filter(name, p):
            noise_targets.append((p, torch.zeros_like(p.data)))

    if not noise_targets:
        return

    for _ in range(n_pairs):
        epsilons: list[torch.Tensor] = []
        for p, _ in noise_targets:
            e = torch.randn_like(p.data)
            epsilons.append(e)
            p.data.add_(sigma * e)
        lp = get_loss()
        for p, e in zip([t[0] for t in noise_targets], epsilons, strict=True):
            p.data.sub_(2 * sigma * e)
        ln = get_loss()
        for p, e in zip([t[0] for t in noise_targets], epsilons, strict=True):
            p.data.add_(sigma * e)
            est = (lp - ln) / (2.0 * sigma + 1e-12)
            p.data.sub_(lr * est * e)


def differentiation_param_filter(name: str, p: torch.nn.Parameter) -> bool:
    return "diff." in name or "stem_pool." in name
