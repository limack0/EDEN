"""L2 Eggroll: ES on GeneRegulator hyperparameters (every N epochs)."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn


def eggroll_l2_step(
    regulator: nn.Module,
    get_metric: Callable[[], float],
    lr: float = 0.005,
    sigma: float = 0.05,
    n_pairs: int = 10,
    maximize: bool = False,
) -> None:
    """
    maximize=False minimizes metric (e.g. loss); maximize=True maximizes accuracy.
    """
    params = [p for p in regulator.parameters() if p.requires_grad]
    if not params:
        return
    sign = 1.0 if maximize else -1.0

    for _ in range(n_pairs):
        epsilons = [torch.randn_like(p.data) for p in params]
        for p, e in zip(params, epsilons, strict=True):
            p.data.add_(sigma * e)
        mp = get_metric()
        for p, e in zip(params, epsilons, strict=True):
            p.data.sub_(2 * sigma * e)
        mn = get_metric()
        for p, e in zip(params, epsilons, strict=True):
            p.data.add_(sigma * e)
            est = (mp - mn) / (2.0 * sigma + 1e-12)
            p.data.add_(sign * lr * est * e)
