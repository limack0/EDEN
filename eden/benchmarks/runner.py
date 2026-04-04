"""Benchmark orchestration: EDEN vs baselines, JSON/CSV output."""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from eden.benchmarks import baselines, datasets
from eden.config import AblationFlags, get_device, set_seed
from eden.core.epigenome import HeritableEpigenome
from eden.core.genome import GeneRegulator
from eden.core.network import EDENNetwork, SequenceEDENNetwork
from eden.training import train_eden

log = logging.getLogger(__name__)


@torch.no_grad()
def eval_model(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    forward_fn: Any | None = None,
) -> float:
    model.eval()
    correct, total = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if forward_fn:
            logits = forward_fn(model, xb)
        else:
            logits = model(xb)
        correct += (logits.argmax(-1) == yb).sum().item()
        total += yb.numel()
    return correct / max(total, 1)


def train_baseline_simple(
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    epochs: int,
    lr: float = 1e-3,
) -> dict[str, Any]:
    device = get_device()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    t0 = time.perf_counter()
    name = model.__class__.__name__
    log.info("baseline | %s | epochs=%d | batches/epoch=%d", name, epochs, len(train_loader))
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = ce(model(xb), yb)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        log.info(
            "baseline | %s | epoch %d/%d | train loss %.4f",
            name,
            ep + 1,
            epochs,
            ep_loss / max(n, 1),
        )
    acc = eval_model(model, val_loader, device)
    sec = time.perf_counter() - t0
    log.info("baseline | %s | done | val_acc=%.4f | %.1fs", name, acc, sec)
    return {"accuracy": acc, "params": sum(p.numel() for p in model.parameters()), "seconds": sec}


def build_eden(name: str, meta: dict[str, Any], flags: AblationFlags) -> EDENNetwork | SequenceEDENNetwork:
    nc = int(meta["num_classes"])
    if meta.get("kind") in ("ecg", "spiral", "protein"):
        return SequenceEDENNetwork(nc, int(meta["seq_len"]), flags=flags)
    return EDENNetwork(nc, int(meta["in_channels"]), tuple(meta["image_hw"]), flags=flags)


def run_benchmark_suite(
    datasets_list: list[str],
    seeds: int,
    epochs: int,
    batch_size: int,
    data_root: Path,
    output_dir: Path,
    quick: bool = False,
    log_batches_every: int = 0,
    eden_only: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"datasets": {}}
    if eden_only:
        log.info("benchmark | eden-only mode (skipping MLP/CNN/LSTM baselines)")

    for ds in datasets_list:
        per_ds: list[dict[str, Any]] = []
        for s in range(seeds):
            set_seed(s)
            if ds in ("mnist", "fashion_mnist", "cifar10", "cifar100"):
                tr, va, meta = datasets.get_torchvision_loaders(ds, data_root, batch_size)
            elif ds in ("ecg", "protein"):
                tr, va, meta = datasets.get_sequence_loaders(ds, batch_size, seed=s)
            elif ds == "synth":
                full = datasets.spiral_dataset(600, seed=s)
                n = len(full)
                train_d, val_d = torch.utils.data.random_split(
                    full, [int(0.85 * n), n - int(0.85 * n)], generator=torch.Generator().manual_seed(s)
                )
                tr = DataLoader(train_d, batch_size=batch_size, shuffle=True)
                va = DataLoader(val_d, batch_size=batch_size, shuffle=False)
                meta = {"num_classes": 2, "seq_len": 2, "kind": "spiral"}
            else:
                raise ValueError(ds)

            e_epochs = min(3, epochs) if quick else epochs
            flags = AblationFlags()
            model = build_eden(ds, meta, flags)
            reg = GeneRegulator()
            epi = HeritableEpigenome()
            log.info("benchmark | EDEN | dataset=%s seed=%d epochs=%d quick=%s", ds, s, e_epochs, quick)
            r = train_eden(
                model, reg, epi, tr, va, e_epochs, results_dir=None, log_batches_every=log_batches_every
            )
            r.update({"model": "EDEN", "dataset": ds, "seed": s})
            per_ds.append(r)
            rows.append(r)

            if not eden_only:
                if ds in ("mnist", "fashion_mnist"):
                    ch, hw = int(meta["in_channels"]), int(meta["image_hw"][0])
                    flat = ch * hw * hw
                    mlp = baselines.MLPBaseline(flat, int(meta["num_classes"]))
                    b = train_baseline_simple(mlp, tr, va, e_epochs)
                    b.update({"model": "MLP", "dataset": ds, "seed": s})
                    rows.append(b)
                    lenet = baselines.LeNet5(ch, int(meta["num_classes"]))
                    b2 = train_baseline_simple(lenet, tr, va, e_epochs)
                    b2.update({"model": "LeNet5", "dataset": ds, "seed": s})
                    rows.append(b2)
                if ds in ("cifar10", "cifar100"):
                    ch = int(meta["in_channels"])
                    rn = baselines.ResNet8(ch, int(meta["num_classes"]))
                    b3 = train_baseline_simple(rn, tr, va, e_epochs)
                    b3.update({"model": "ResNet8", "dataset": ds, "seed": s})
                    rows.append(b3)
                if ds == "ecg":
                    cnn = baselines.CNN1DBaseline(int(meta["seq_len"]), int(meta["num_classes"]))
                    b4 = train_baseline_simple(cnn, tr, va, e_epochs)
                    b4.update({"model": "CNN1D", "dataset": ds, "seed": s})
                    rows.append(b4)
                if ds == "protein":
                    lstm = baselines.LSTMOnFlattenedSeq(128, 20, int(meta["num_classes"]))
                    b5 = train_baseline_simple(lstm, tr, va, e_epochs)
                    b5.update({"model": "LSTM", "dataset": ds, "seed": s})
                    rows.append(b5)

        summary["datasets"][ds] = {
            "eden_mean_acc": float(sum(x["final_val_accuracy"] for x in per_ds) / max(len(per_ds), 1)),
            "runs": len(per_ds),
        }

    with open(output_dir / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows, "meta": {"eden_only": eden_only}}, f, indent=2)

    if rows:
        keys = sorted(rows[0].keys())
        with open(output_dir / "benchmark_rows.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k, "") for k in keys})

    return summary
