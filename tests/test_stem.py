from __future__ import annotations

import torch

from eden.core.stem import StemPerceptron, StemPool


def test_stem_perceptron() -> None:
    s = StemPerceptron(32, 16)
    x = torch.randn(4, 32)
    y = s(x)
    assert y.shape == (4, 16)


def test_stem_pool() -> None:
    p = StemPool(32, 16, n_stems=4, keep=2, retention_interval=10, max_stems=8)
    x = torch.randn(4, 32)
    out = p(x)
    assert out.shape == (4, 16)
    p.update_scores(torch.tensor([0.1, 0.9, 0.8, 0.2]))
    p.apply_retention(10)
    assert p.active_mask.sum().item() == 2
