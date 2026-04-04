"""Elementary-effects (Morris) screening for GeneRegulator parameters."""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn


def morris_elementary_effects(
    run_accuracy: Callable[[nn.Module], float],
    regulator_class: type[nn.Module],
    attr_names: list[str],
    *,
    delta: float = 0.2,
    n_trajectories: int = 10,
    seed: int = 0,
) -> dict[str, float]:
    """
    For each trajectory, draw a fresh random ``GeneRegulator``, then for each attribute
    compare accuracy after full training with baseline vs. (1+δ) multiplicative bump on that parameter.
    """
    mu_star: dict[str, list[float]] = {a: [] for a in attr_names}

    for t in range(n_trajectories):
        torch.manual_seed(seed + t)
        tpl = regulator_class()
        sd0 = {k: v.clone() for k, v in tpl.state_dict().items()}
        for attr in attr_names:
            b = regulator_class()
            b.load_state_dict(sd0)
            y0 = run_accuracy(b)
            r = regulator_class()
            r.load_state_dict(sd0)
            with torch.no_grad():
                getattr(r, attr).mul_(1.0 + delta)
            y1 = run_accuracy(r)
            ee = abs((y1 - y0) / (delta + 1e-12))
            mu_star[attr].append(ee)

    return {a: float(sum(v) / max(len(v), 1)) for a, v in mu_star.items()}


def morris_summary_to_jsonable(mu_star: dict[str, float]) -> dict[str, Any]:
    vals = list(mu_star.values())
    return {
        "method": "morris_elementary_effects",
        "mu_star": mu_star,
        "mean_mu_star": float(sum(vals) / max(len(vals), 1)),
        "relative_sensitivity": {k: v / (sum(vals) + 1e-12) for k, v in mu_star.items()},
    }
