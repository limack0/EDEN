from __future__ import annotations

import torch

from eden.metrics import node_activation_redundancy, training_event_counts


def test_node_redundancy_range() -> None:
    acts = torch.randn(4, 8, 32)
    r = node_activation_redundancy(acts)
    assert 0.0 <= r <= 1.0


def test_event_counts() -> None:
    ev = [{"kind": "a"}, {"kind": "b"}, {"kind": "a"}]
    c = training_event_counts(ev)
    assert c["a"] == 2 and c["b"] == 1
