"""EDEN network: DAG-like stack with stems, morphogens, differentiation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from eden.config import AblationFlags, TrainingState
from eden.core.apoptosis import MicrogliaCompetition, ProgrammedApoptosis
from eden.core.differentiation import DifferentiationPhi
from eden.core.genome import HierarchicalGenome, genome_to_gates
from eden.core.glia import AstrocyteModulator, OligodendrocyteSheath
from eden.core.morphogen import (
    MORPHOGEN_DIM,
    ParacrineCascade,
    LocalMorphogen,
    activation_entropy,
)
from eden.core.stem import StemPool


@dataclass
class EDENOutput:
    logits: torch.Tensor
    node_activations: torch.Tensor
    morphogen: torch.Tensor


class EDENNetwork(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        image_hw: tuple[int, int] = (28, 28),
        hidden: int = 128,
        n_nodes: int = 8,
        flags: AblationFlags | None = None,
        max_pathway_nodes: int = 24,
        max_stems: int = 12,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.flags = flags or AblationFlags()
        self.n_nodes = n_nodes
        self.max_pathway_nodes = max(max(1, max_pathway_nodes), n_nodes)
        self.hidden = hidden
        h, w = image_hw
        self.embed = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        with torch.no_grad():
            d = self.embed(torch.zeros(1, in_channels, h, w)).numel()
        self.flat_dim = d
        self.stem_pool = StemPool(
            self.flat_dim, hidden, n_stems=4, keep=2, retention_interval=10, max_stems=max_stems
        )
        self.node_proj = nn.Linear(hidden, n_nodes * hidden)
        self.diff = DifferentiationPhi(hidden)
        self.morphogen = LocalMorphogen()
        self.morph_to_hidden = nn.Linear(MORPHOGEN_DIM, hidden)
        self.paracrine = ParacrineCascade(k=3)
        self.astro = AstrocyteModulator(hidden)
        self.oligo = OligodendrocyteSheath(hidden)
        self.micro = MicrogliaCompetition()
        self.prog = ProgrammedApoptosis()
        self.head = nn.Linear(hidden, num_classes)
        self.register_buffer("_genome_type", torch.zeros(150))
        self.register_buffer("_genome_conn", torch.zeros(350))

    def set_genome_buffers(self, genome: HierarchicalGenome) -> None:
        self._genome_type = genome.subgenome_type.detach().clone()
        self._genome_conn = genome.subgenome_conn.detach().clone()

    def forward(
        self,
        x: torch.Tensor,
        regulator: dict[str, torch.Tensor],
        epigenome_mask: torch.Tensor,
        training_state: TrainingState | None = None,
    ) -> EDENOutput:
        B = x.shape[0]
        if x.dim() == 2:
            z = x
            if z.shape[1] != self.flat_dim:
                z = F.pad(z, (0, self.flat_dim - z.shape[1]))
        else:
            z = torch.flatten(self.embed(x), 1)

        genome = HierarchicalGenome(subgenome_type=self._genome_type, subgenome_conn=self._genome_conn)
        gates = genome_to_gates(genome, epigenome_mask, self.stem_pool.n_stems, self.hidden)
        h = self.stem_pool(z, stem_weights=gates)
        h = h.view(B, -1)
        h = self.node_proj(h).view(B, self.n_nodes, self.hidden)

        gl = training_state.last_loss if training_state else 0.0
        ga = training_state.last_accuracy if training_state else 0.0
        morph = self.morphogen.compute(h, gl, ga)
        h = h + 0.1 * self.morph_to_hidden(morph)

        if self.flags.paracrine:
            h = self.paracrine(h, regulator["paracrine_strength"])

        h = self.diff(h, regulator, self.flags.hox_waves)
        if self.flags.glia:
            h = self.astro(h)
            h = self.oligo(h)

        if self.flags.microglia:
            h = self.micro(h, regulator["apoptosis_theta"], True)
        if self.flags.programmed_apoptosis:
            h = self.prog(h, regulator["apoptosis_theta"], True)

        if self.flags.contact_inhibition:
            rho = regulator["rho"]
            eth = regulator["entropy_theta"]
            hn = F.normalize(h, dim=-1)
            sim = torch.bmm(hn, hn.transpose(1, 2))
            ent = activation_entropy(h, dim=-1)
            ent_sim = (ent.unsqueeze(2) - ent.unsqueeze(1)).abs()
            inhibit = ((sim > rho) & (ent_sim < eth)).to(h.dtype)
            n = h.size(1)
            eye = torch.eye(n, device=h.device, dtype=h.dtype).view(1, n, n)
            inhibit = inhibit * (1.0 - eye)
            penalty = inhibit.sum(dim=-1, keepdim=True).clamp(max=1.0)
            h = h * (1.0 - 0.5 * penalty)

        pooled = h.mean(dim=1)
        logits = self.head(pooled)
        return EDENOutput(logits=logits, node_activations=h, morphogen=morph)

    def prune_synapses(self, epsilon: float = 0.01) -> int:
        if not self.flags.synaptic_pruning:
            return 0
        n = 0
        with torch.no_grad():
            for p in self.head.parameters():
                if p.dim() >= 1:
                    mask = p.data.abs() < epsilon
                    n += int(mask.sum().item())
                    p.data[mask] = 0
        return n


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class SequenceEDENNetwork(nn.Module):
    """EDEN stack for flat sequences (ECG, protein embeddings)."""

    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        hidden: int = 128,
        n_nodes: int = 8,
        flags: AblationFlags | None = None,
        max_pathway_nodes: int = 24,
        max_stems: int = 12,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.flags = flags or AblationFlags()
        self.n_nodes = n_nodes
        self.max_pathway_nodes = max(max(1, max_pathway_nodes), n_nodes)
        self.hidden = hidden
        self.in_proj = nn.Linear(seq_len, 512)
        self.flat_dim = 512
        self.stem_pool = StemPool(
            self.flat_dim, hidden, n_stems=4, keep=2, retention_interval=10, max_stems=max_stems
        )
        self.node_proj = nn.Linear(hidden, n_nodes * hidden)
        self.diff = DifferentiationPhi(hidden)
        self.morphogen = LocalMorphogen()
        self.morph_to_hidden = nn.Linear(MORPHOGEN_DIM, hidden)
        self.paracrine = ParacrineCascade(k=3)
        self.astro = AstrocyteModulator(hidden)
        self.oligo = OligodendrocyteSheath(hidden)
        self.micro = MicrogliaCompetition()
        self.prog = ProgrammedApoptosis()
        self.head = nn.Linear(hidden, num_classes)
        self.register_buffer("_genome_type", torch.zeros(150))
        self.register_buffer("_genome_conn", torch.zeros(350))

    def set_genome_buffers(self, genome: HierarchicalGenome) -> None:
        self._genome_type = genome.subgenome_type.detach().clone()
        self._genome_conn = genome.subgenome_conn.detach().clone()

    def forward(
        self,
        x: torch.Tensor,
        regulator: dict[str, torch.Tensor],
        epigenome_mask: torch.Tensor,
        training_state: TrainingState | None = None,
    ) -> EDENOutput:
        B = x.shape[0]
        z = F.relu(self.in_proj(x))
        genome = HierarchicalGenome(subgenome_type=self._genome_type, subgenome_conn=self._genome_conn)
        gates = genome_to_gates(genome, epigenome_mask, self.stem_pool.n_stems, self.hidden)
        h = self.stem_pool(z, stem_weights=gates)
        h = h.view(B, -1)
        h = self.node_proj(h).view(B, self.n_nodes, self.hidden)
        gl = training_state.last_loss if training_state else 0.0
        ga = training_state.last_accuracy if training_state else 0.0
        morph = self.morphogen.compute(h, gl, ga)
        h = h + 0.1 * self.morph_to_hidden(morph)
        if self.flags.paracrine:
            h = self.paracrine(h, regulator["paracrine_strength"])
        h = self.diff(h, regulator, self.flags.hox_waves)
        if self.flags.glia:
            h = self.astro(h)
            h = self.oligo(h)
        if self.flags.microglia:
            h = self.micro(h, regulator["apoptosis_theta"], True)
        if self.flags.programmed_apoptosis:
            h = self.prog(h, regulator["apoptosis_theta"], True)
        if self.flags.contact_inhibition:
            rho = regulator["rho"]
            eth = regulator["entropy_theta"]
            hn = F.normalize(h, dim=-1)
            sim = torch.bmm(hn, hn.transpose(1, 2))
            ent = activation_entropy(h, dim=-1)
            ent_sim = (ent.unsqueeze(2) - ent.unsqueeze(1)).abs()
            inhibit = ((sim > rho) & (ent_sim < eth)).to(h.dtype)
            n = h.size(1)
            eye = torch.eye(n, device=h.device, dtype=h.dtype).view(1, n, n)
            inhibit = inhibit * (1.0 - eye)
            penalty = inhibit.sum(dim=-1, keepdim=True).clamp(max=1.0)
            h = h * (1.0 - 0.5 * penalty)
        pooled = h.mean(dim=1)
        logits = self.head(pooled)
        return EDENOutput(logits=logits, node_activations=h, morphogen=morph)

    def prune_synapses(self, epsilon: float = 0.01) -> int:
        if not self.flags.synaptic_pruning:
            return 0
        n = 0
        with torch.no_grad():
            for p in self.head.parameters():
                if p.dim() >= 1:
                    mask = p.data.abs() < epsilon
                    n += int(mask.sum().item())
                    p.data[mask] = 0
        return n
