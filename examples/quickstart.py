"""
EDEN — Quickstart
=================
Ce script illustre le cycle complet : entraînement → sauvegarde → inférence.
Fonctionne sur CPU ou GPU (détection automatique).

Usage
-----
    python examples/quickstart.py

Prérequis
---------
    pip install -e ".[vision]"
"""

from pathlib import Path

import torch

# ── 1. Imports EDEN ───────────────────────────────────────────────────────────
from eden import EDENPredictor
from eden.benchmarks.datasets import get_torchvision_loaders
from eden.config import AblationFlags, set_seed
from eden.core.epigenome import HeritableEpigenome
from eden.core.genome import GeneRegulator
from eden.core.network import EDENNetwork
from eden.training import save_checkpoint, train_eden

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = Path("eden_quickstart.pt")
DATA_ROOT  = Path(".data")

print(f"Device : {DEVICE}")

# ── 2. Données ────────────────────────────────────────────────────────────────
set_seed(42)
tr_l, va_l, meta = get_torchvision_loaders("mnist", DATA_ROOT, batch_size=128)
print(f"MNIST  : {meta['num_classes']} classes, images {meta['image_hw']}")

# ── 3. Modèle ─────────────────────────────────────────────────────────────────
model     = EDENNetwork(
    num_classes=int(meta["num_classes"]),
    in_channels=int(meta["in_channels"]),
    image_hw=tuple(meta["image_hw"]),
    flags=AblationFlags(),          # configuration par défaut (tous mécanismes actifs)
)
regulator = GeneRegulator()
epigenome = HeritableEpigenome()

params = sum(p.numel() for p in model.parameters())
print(f"Params : {params:,}")

# ── 4. Entraînement ───────────────────────────────────────────────────────────
print("\nEntraînement (5 epochs — rapide pour la démo)...")
result = train_eden(
    model, regulator, epigenome,
    tr_l, va_l,
    epochs=5,
    seed=42,
)
print(f"Val acc : {result['final_val_accuracy']:.4f}")

# ── 5. Sauvegarde checkpoint ──────────────────────────────────────────────────
save_checkpoint(CHECKPOINT, model, regulator, {"val_acc": result["final_val_accuracy"]})
print(f"Checkpoint sauvegardé : {CHECKPOINT}")

# ── 6. Inférence avec EDENPredictor ──────────────────────────────────────────
print("\nChargement du predictor...")
model2     = EDENNetwork(int(meta["num_classes"]), int(meta["in_channels"]), tuple(meta["image_hw"]))
regulator2 = GeneRegulator()
epigenome2 = HeritableEpigenome()

predictor = EDENPredictor.from_checkpoint(
    CHECKPOINT, model2, regulator2, epigenome2, device=DEVICE
)
print(predictor)

# Prédiction sur un batch de validation
batch_x, batch_y = next(iter(va_l))
labels = predictor.predict(batch_x[:8])
probs  = predictor.predict_proba(batch_x[:8])

print("\nPrédictions sur 8 exemples :")
for i, (pred, truth) in enumerate(zip(labels.tolist(), batch_y[:8].tolist())):
    conf = probs[i, pred].item()
    mark = "✓" if pred == truth else "✗"
    print(f"  [{mark}] prédit={pred}  vrai={truth}  confiance={conf:.3f}")
