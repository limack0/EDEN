"""
EDEN — Exemple d'inférence production
======================================
Montre comment charger un checkpoint entraîné et l'utiliser efficacement
pour de l'inférence batch ou single-sample.

Usage
-----
    python examples/inference_example.py --checkpoint run.pt --dataset mnist
    python examples/inference_example.py --checkpoint run.pt --dataset cifar10
    python examples/inference_example.py --checkpoint run.pt --dataset ecg

Prérequis
---------
    pip install -e ".[vision]"
    # + avoir un checkpoint entraîné (eden train --checkpoint-out run.pt)
"""

import argparse
import time
from pathlib import Path

import torch

from eden import EDENPredictor
from eden.core.epigenome import HeritableEpigenome
from eden.core.genome import GeneRegulator
from eden.core.network import EDENNetwork, SequenceEDENNetwork


def build_predictor(checkpoint: Path, dataset: str, device: str) -> EDENPredictor:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    if dataset == "ecg":
        # ECG : SequenceEDENNetwork (seq_len=188 par défaut)
        model = SequenceEDENNetwork(num_classes=2, seq_len=188)
    elif dataset in ("mnist", "fashion_mnist"):
        model = EDENNetwork(num_classes=10, in_channels=1, image_hw=(28, 28))
    elif dataset == "cifar10":
        model = EDENNetwork(num_classes=10, in_channels=3, image_hw=(32, 32))
    else:
        raise ValueError(f"Dataset inconnu : {dataset!r}")

    return EDENPredictor.from_checkpoint(
        checkpoint,
        model,
        GeneRegulator(),
        HeritableEpigenome(),
        device=device,
    )


def benchmark_latency(predictor: EDENPredictor, sample: torch.Tensor, label: str) -> None:
    """Mesure la latence réelle avec le cache pré-chargé."""
    predictor.predict(sample)  # warmup

    if torch.cuda.is_available() and predictor.device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    n = 200
    for _ in range(n):
        predictor.predict(sample)

    if torch.cuda.is_available() and predictor.device.type == "cuda":
        torch.cuda.synchronize()

    ms = (time.perf_counter() - t0) * 1000 / n
    bs = sample.shape[0]
    print(f"  {label:<30} {ms:6.2f} ms/batch  {ms/bs*1000:7.1f} µs/sample  {bs/ms*1000:8.0f} samples/s")


def main() -> None:
    parser = argparse.ArgumentParser(description="EDEN inference example")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset",    choices=["mnist", "fashion_mnist", "cifar10", "ecg"], default="mnist")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"\nChargement : {args.checkpoint}  |  dataset={args.dataset}  |  device={args.device}")
    predictor = build_predictor(args.checkpoint, args.dataset, args.device)
    print(predictor)

    # ── Benchmark latence ─────────────────────────────────────────────────────
    if args.dataset == "ecg":
        dummy_1  = torch.randn(1,  188)
        dummy_32 = torch.randn(32, 188)
    elif args.dataset == "cifar10":
        dummy_1  = torch.randn(1,  3, 32, 32)
        dummy_32 = torch.randn(32, 3, 32, 32)
    else:  # mnist / fashion_mnist
        dummy_1  = torch.randn(1,  1, 28, 28)
        dummy_32 = torch.randn(32, 1, 28, 28)

    print(f"\nLatence inférence ({args.device}) :")
    print(f"  {'label':<30} {'ms/batch':>10}  {'µs/sample':>12}  {'samples/s':>12}")
    print("  " + "-"*60)
    benchmark_latency(predictor, dummy_1,  f"{args.dataset} batch=1")
    benchmark_latency(predictor, dummy_32, f"{args.dataset} batch=32")

    # ── Prédiction sur données synthétiques ───────────────────────────────────
    print(f"\nPrédictions sur 4 exemples synthétiques :")
    probs  = predictor.predict_proba(dummy_1.repeat(4, *([1] * (dummy_1.dim() - 1))))
    labels = predictor.predict(dummy_1.repeat(4, *([1] * (dummy_1.dim() - 1))))
    for i, (lbl, p) in enumerate(zip(labels.tolist(), probs)):
        conf = p[lbl].item()
        print(f"  sample {i}  →  classe {lbl}  (confiance {conf:.3f})")


if __name__ == "__main__":
    main()
