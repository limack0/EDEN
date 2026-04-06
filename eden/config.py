"""Global configuration: reproducibility and device selection."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

SEED: int = int(os.environ.get("EDEN_SEED", "42"))


def set_seed(seed: int | None = None) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    s = SEED if seed is None else seed
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class AblationFlags:
    """Toggle biological mechanisms for ablation studies."""

    microglia: bool = True
    contact_inhibition: bool = True
    hox_waves: bool = True
    programmed_apoptosis: bool = True
    epigenome_drift: bool = True
    synaptic_pruning: bool = True
    glia: bool = True
    node_attention: bool = True

    def copy(self) -> AblationFlags:
        return AblationFlags(
            microglia=self.microglia,
            contact_inhibition=self.contact_inhibition,
            hox_waves=self.hox_waves,
            programmed_apoptosis=self.programmed_apoptosis,
            epigenome_drift=self.epigenome_drift,
            synaptic_pruning=self.synaptic_pruning,
            glia=self.glia,
            node_attention=self.node_attention,
        )

    def to_dict(self) -> dict[str, bool]:
        return {
            "microglia": self.microglia,
            "contact_inhibition": self.contact_inhibition,
            "hox_waves": self.hox_waves,
            "programmed_apoptosis": self.programmed_apoptosis,
            "epigenome_drift": self.epigenome_drift,
            "synaptic_pruning": self.synaptic_pruning,
            "glia": self.glia,
            "node_attention": self.node_attention,
        }


@dataclass
class TrainingState:
    """Mutable training-time metrics for maturity and logging."""

    epoch: int = 0
    batch: int = 0
    last_loss: float = 0.0
    last_accuracy: float = 0.0
    stability_ema: float = 0.5
    events: list[dict[str, Any]] = field(default_factory=list)

    def log_event(self, kind: str, **kwargs: Any) -> None:
        self.events.append({"kind": kind, "epoch": self.epoch, **kwargs})
