"""EDEN inference wrapper.

Problème résolu : EDENNetwork.forward() exige regulator et epigenome_mask
comme arguments positionnels à chaque appel. Créer ces objets à chaque
inférence coûte ~3.5 ms fixes (overhead mesuré sur T4 Kaggle).

Solution : EDENPredictor pré-instancie GeneRegulator et HeritableEpigenome
une seule fois, met en cache leurs sorties sur le device cible, et expose
une interface predict/predict_proba standard.

Usage minimal
-------------
    from eden.inference import EDENPredictor
    from eden.core.network import EDENNetwork
    from eden.core.genome import GeneRegulator
    from eden.core.epigenome import HeritableEpigenome

    model     = EDENNetwork(num_classes=10, in_channels=1, image_hw=(28, 28))
    regulator = GeneRegulator()
    epigenome = HeritableEpigenome()
    # ... charger les poids ...

    predictor = EDENPredictor(model, regulator, epigenome, device="cuda")
    probs  = predictor.predict_proba(x)   # (B, C) float32
    labels = predictor.predict(x)         # (B,)   int64

Chargement depuis checkpoint
-----------------------------
    predictor = EDENPredictor.from_checkpoint(
        "run.pt",
        model, regulator, epigenome,
        device="cuda",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from eden.core.epigenome import HeritableEpigenome
from eden.core.genome import GeneRegulator
from eden.core.network import EDENNetwork, EDENOutput, SequenceEDENNetwork

_AnyEDEN = Union[EDENNetwork, SequenceEDENNetwork]


class EDENPredictor:
    """Stateful inference wrapper — regulator et epigenome_mask pré-calculés.

    Les deux modules auxiliaires (GeneRegulator, HeritableEpigenome) sont
    évalués une seule fois à la construction et leurs sorties sont mises en
    cache sur le device cible. Aucun overhead de création à l'inférence.
    """

    def __init__(
        self,
        model: _AnyEDEN,
        regulator: GeneRegulator,
        epigenome: HeritableEpigenome,
        device: str | torch.device = "cpu",
    ) -> None:
        self._device = torch.device(device)
        self._model = model.to(self._device).eval()
        self._regulator = regulator.to(self._device).eval()
        self._epigenome = epigenome.to(self._device).eval()
        self._reg_cache: dict[str, torch.Tensor] | None = None
        self._epi_cache: torch.Tensor | None = None
        self._warm_up()

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model: _AnyEDEN,
        regulator: GeneRegulator,
        epigenome: HeritableEpigenome,
        device: str | torch.device = "cpu",
    ) -> EDENPredictor:
        """Charge un checkpoint et retourne un EDENPredictor prêt à l'emploi."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        regulator.load_state_dict(ckpt["regulator"])
        if "epigenome" in ckpt:
            epigenome.load_state_dict(ckpt["epigenome"])
        return cls(model, regulator, epigenome, device=device)

    # ── Cache interne ─────────────────────────────────────────────────────────

    def _warm_up(self) -> None:
        """Calcule et met en cache regulator + epigenome_mask (sans gradient)."""
        with torch.no_grad():
            self._reg_cache = {
                k: v.to(self._device) for k, v in self._regulator().items()
            }
            _, epi_soft = self._epigenome()
            self._epi_cache = epi_soft.to(self._device)

    def to(self, device: str | torch.device) -> EDENPredictor:
        """Déplace le predictor sur un nouveau device et rafraîchit le cache."""
        self._device = torch.device(device)
        self._model = self._model.to(self._device)
        self._regulator = self._regulator.to(self._device)
        self._epigenome = self._epigenome.to(self._device)
        self._warm_up()
        return self

    # ── Inférence ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _forward(self, x: torch.Tensor) -> EDENOutput:
        x = x.to(self._device)
        return self._model(x, self._reg_cache, self._epi_cache)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne les probabilités softmax de shape (B, num_classes)."""
        out = self._forward(x)
        return F.softmax(out.logits, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne la classe prédite de shape (B,)."""
        out = self._forward(x)
        return out.logits.argmax(dim=-1)

    def predict_with_nodes(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Retourne (classes, node_activations) — utile pour l'interprétabilité."""
        out = self._forward(x)
        return out.logits.argmax(dim=-1), out.node_activations

    # ── Méta ──────────────────────────────────────────────────────────────────

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def num_classes(self) -> int:
        return self._model.num_classes

    def __repr__(self) -> str:
        params = sum(p.numel() for p in self._model.parameters())
        return (
            f"EDENPredictor("
            f"model={type(self._model).__name__}, "
            f"params={params:,}, "
            f"device={self._device}, "
            f"classes={self.num_classes})"
        )
