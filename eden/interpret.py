"""UMAP of ATGC-like gene vectors vs functional similarity (cosine on activations)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from eden.core.genome import HierarchicalGenome


def gene_sequence_tensor(genome: HierarchicalGenome) -> np.ndarray:
    v = torch.cat([genome.subgenome_type, genome.subgenome_conn], dim=0).detach().cpu().numpy()
    return v


def functional_similarity_matrix(activations: torch.Tensor) -> np.ndarray:
    """activations (N, D) per sample or (B,N,D) — flatten nodes."""
    if activations.dim() == 3:
        a = activations.mean(dim=1).detach().cpu().numpy()
    else:
        a = activations.detach().cpu().numpy()
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    return a @ a.T


def genome_function_correlation_study(
    model: nn.Module,
    regulator: nn.Module,
    epigenome: nn.Module,
    x_batch: torch.Tensor,
    device: torch.device,
    n_genomes: int = 32,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Sample random genomes, fix a batch ``x_batch``, record gene vectors and mean node activations.
    UMAP/correlation: embedding distance vs functional dissimilarity (1 - cosine sim of activations).
    """
    from eden.config import TrainingState, set_seed
    from eden.core.genome import N_CONN_GENES, N_TYPE_GENES

    set_seed(seed)
    model.eval()
    st = TrainingState()
    xb = x_batch.to(device)[: min(32, x_batch.size(0))]
    rows_g: list[np.ndarray] = []
    rows_a: list[np.ndarray] = []

    gtor = torch.Generator()
    gtor.manual_seed(seed)
    for _ in range(n_genomes):
        genome = HierarchicalGenome.random(device, generator=gtor)
        model.set_genome_buffers(genome)
        mask, _ = epigenome()
        mvec = (
            mask[: N_TYPE_GENES + N_CONN_GENES]
            if mask.shape[0] >= N_TYPE_GENES + N_CONN_GENES
            else torch.ones(N_TYPE_GENES + N_CONN_GENES, device=device)
        )
        with torch.no_grad():
            reg_out = regulator()
            out = model(xb, reg_out, mvec, st)
        rows_g.append(gene_sequence_tensor(genome))
        act = out.node_activations.mean(dim=(0, 1)).detach().cpu().numpy()
        rows_a.append(act)

    G = np.stack(rows_g, axis=0)
    A = np.stack(rows_a, axis=0)
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    func_sim = An @ An.T
    base = umap_correlation_stub(G, func_sim)
    base.update({"n_genomes": n_genomes, "gene_dim": int(G.shape[1])})
    return base


def umap_correlation_stub(
    gene_matrix: np.ndarray,
    func_sim: np.ndarray,
) -> dict[str, Any]:
    """
    If umap-learn is installed, embed genes and report correlation between
    embedding distance and functional dissimilarity; else return diagnostic.
    """
    try:
        import umap
    except ImportError:
        return {"umap_available": False, "note": "install umap-learn for embedding"}

    try:
        from scipy.spatial.distance import pdist, squareform
    except ImportError:
        return {"umap_available": True, "note": "install scipy for correlation metric"}

    if gene_matrix.shape[0] < 5:
        return {"umap_available": True, "note": "need at least 5 genomes for UMAP", "correlation_emb_vs_func_dissimilarity": None}

    reducer = umap.UMAP(n_components=2, random_state=42)
    emb = reducer.fit_transform(gene_matrix)
    ge = squareform(pdist(emb))
    fs = 1.0 - func_sim
    mask = np.triu(np.ones_like(ge, dtype=bool), k=1)
    c = np.corrcoef(ge[mask], fs[mask])[0, 1]
    if np.isnan(c):
        c = 0.0
    return {"umap_available": True, "correlation_emb_vs_func_dissimilarity": float(c)}
