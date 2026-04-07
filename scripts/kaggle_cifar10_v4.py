"""CIFAR-10 validation — v4 (current architecture).

Architecture expected:
  - 3 conv blocks + AdaptiveAvgPool2d(4) → flat_dim=2048
  - hidden=256, n_nodes=8
  - Gradient clipping max_norm=1.0
  - AdamW + warmup(5ep) + cosine LR
  - Label smoothing=0.1

v1: flat_dim=4096 (2 conv), hidden=128 → mean=0.6741
v2: flat_dim=4096 (2 conv), hidden=128, +augment+BN → mean=0.6759
v3: flat_dim=8192 (3 conv, no pool), hidden=256 → mean=0.2844 (unstable)
v4: flat_dim=2048 (3 conv + AdaptiveAvgPool4), hidden=256, +grad_clip → TBD
"""
import subprocess, sys
from pathlib import Path

subprocess.run(["rm", "-rf", "/kaggle/working/EDEN"], check=True)
subprocess.run(["git", "clone", "https://github.com/limack0/EDEN", "/kaggle/working/EDEN"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-e", "/kaggle/working/EDEN", "-q"], check=True)

import statistics
sys.path.insert(0, "/kaggle/working/EDEN")

import torch
print("CUDA:", torch.cuda.is_available())

from eden.benchmarks import datasets
from eden.config import AblationFlags, set_seed
from eden.core.epigenome import HeritableEpigenome
from eden.core.genome import GeneRegulator
from eden.core.network import EDENNetwork
from eden.training import train_eden

SEEDS     = list(range(5))
EPOCHS    = 100
BATCH     = 128
DATA_ROOT = Path("/kaggle/working/.data")

V1_MEAN = 0.6741
V2_MEAN = 0.6759
V3_MEAN = 0.2844  # unstable (buggy run, flat_dim=8192)

tr_l, va_l, meta = datasets.get_torchvision_loaders("cifar10", DATA_ROOT, BATCH)

model_test = EDENNetwork(int(meta["num_classes"]), int(meta["in_channels"]), tuple(meta["image_hw"]))
flat = model_test.flat_dim
hidden = model_test.hidden
params = sum(p.numel() for p in model_test.parameters())
print(f"flat_dim={flat}  hidden={hidden}  params={params:,}")
assert flat == 2048, f"Expected flat_dim=2048, got {flat} — wrong code version on GitHub"
del model_test

accs = []
for seed in SEEDS:
    set_seed(seed)
    model = EDENNetwork(
        int(meta["num_classes"]),
        int(meta["in_channels"]),
        tuple(meta["image_hw"]),
        flags=AblationFlags(),
    )
    r = train_eden(model, GeneRegulator(), HeritableEpigenome(), tr_l, va_l, EPOCHS, seed=seed)
    acc = r["final_val_accuracy"]
    accs.append(acc)
    print(f"seed={seed}  acc={acc:.4f}")

m = statistics.mean(accs)
s = statistics.stdev(accs)
print(f"\n===== CIFAR-10 v4 — 3conv+AdaptPool+grad_clip ({len(SEEDS)} seeds, {EPOCHS} epochs) =====")
print(f"mean={m:.4f}  std={s:.4f}  min={min(accs):.4f}  max={max(accs):.4f}")
print(f"\nvs v1  Δ={m-V1_MEAN:+.4f}   vs v2  Δ={m-V2_MEAN:+.4f}   vs v3(buggy)  Δ={m-V3_MEAN:+.4f}")
