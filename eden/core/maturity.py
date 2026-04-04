"""Maturity score (empirical stability proxy, no convergence theorem)."""

from __future__ import annotations

import math


def maturity_score(
    accuracy: float,
    expr_ratio: float,
    stability: float,
    redundancy: float,
) -> float:
    raw = 0.5 * accuracy - 0.3 * expr_ratio + 0.2 * stability + 0.1 * redundancy
    return float(1.0 / (1.0 + math.exp(-10.0 * (raw - 0.6))))


def empirical_stability(loss_history: list[float], window: int = 5) -> float:
    """Inverse coefficient of variation of recent losses (higher = more stable)."""
    if len(loss_history) < window:
        return 0.5
    recent = loss_history[-window:]
    m = sum(recent) / len(recent)
    if m < 1e-12:
        return 1.0
    v = sum((x - m) ** 2 for x in recent) / len(recent)
    std = math.sqrt(v)
    cv = std / m if m > 1e-12 else 0.0
    return float(1.0 / (1.0 + cv))
