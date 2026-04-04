from __future__ import annotations

import torch

from eden.core.genome import GeneRegulator, HierarchicalGenome, N_CONN_GENES, N_TYPE_GENES, genome_to_gates


def test_hierarchical_genome_shape() -> None:
    g = HierarchicalGenome.random(torch.device("cpu"))
    assert g.subgenome_type.shape == (N_TYPE_GENES,)
    assert g.subgenome_conn.shape == (N_CONN_GENES,)


def test_gene_regulator_keys() -> None:
    r = GeneRegulator()
    d = r()
    assert "tau" in d and "lambda" in d and "k" in d
    assert d["rho"].numel() == 1


def test_genome_to_gates() -> None:
    g = HierarchicalGenome.random(torch.device("cpu"))
    m = torch.ones(N_TYPE_GENES + N_CONN_GENES)
    w = genome_to_gates(g, m, 4, 128)
    assert w.shape == (4,)
    assert torch.isclose(w.sum(), torch.tensor(1.0), atol=1e-5)
