from __future__ import annotations

import torch

from eden.config import TrainingState
from eden.core.epigenome import HeritableEpigenome
from eden.core.genome import GeneRegulator
from eden.core.neurogenesis import apply_pathway_mitosis, apply_neurogenesis_structural_growth
from eden.core.network import SequenceEDENNetwork
from eden.core.stem import StemPool


def test_stem_mitosis_increases_pool() -> None:
    p = StemPool(32, 16, n_stems=4, max_stems=8)
    assert p.n_stems == 4
    new_ps = p.try_mitosis_stem(noise=0.0)
    assert p.n_stems == 5
    assert len(new_ps) > 0
    x = torch.randn(2, 32)
    assert p(x).shape == (2, 16)


def test_pathway_mitosis_expands_nodes_and_forward() -> None:
    m = SequenceEDENNetwork(3, 10, n_nodes=4, max_pathway_nodes=8)
    reg = GeneRegulator()
    epi = HeritableEpigenome()
    st = TrainingState()
    mask, _ = epi()
    x = torch.randn(2, 10)
    o = m(x, reg(), mask, st)
    assert o.logits.shape == (2, 3)
    assert o.node_activations.shape == (2, 4, m.hidden)
    new_ps = apply_pathway_mitosis(m, noise=0.0)
    assert m.n_nodes == 5
    assert len(new_ps) > 0
    o2 = m(x, reg(), mask, st)
    assert o2.node_activations.shape == (2, 5, m.hidden)


def test_structural_growth_prefers_pathway_then_stem() -> None:
    m = SequenceEDENNetwork(3, 10, n_nodes=4, max_pathway_nodes=5)
    ps, kind = apply_neurogenesis_structural_growth(m)
    assert kind == "neurogenesis_pathway_mitosis"
    assert m.n_nodes == 5
    assert ps
    ps2, kind2 = apply_neurogenesis_structural_growth(m)
    assert m.n_nodes == 5
    assert kind2 == "neurogenesis_stem_mitosis"
    assert ps2
