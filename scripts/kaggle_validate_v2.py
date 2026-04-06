import subprocess, sys

# 1. clone + install
subprocess.run(["rm", "-rf", "/kaggle/working/EDEN"], check=True)
subprocess.run(["git", "clone", "https://github.com/limack0/EDEN", "/kaggle/working/EDEN"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-e", "/kaggle/working/EDEN", "-q"], check=True)

# 2. imports
import statistics
sys.path.insert(0, "/kaggle/working/EDEN")

import torch
from eden.benchmarks import datasets
from eden.config import AblationFlags, set_seed
from eden.core.epigenome import HeritableEpigenome
from eden.core.genome import GeneRegulator
from eden.core.network import EDENNetwork
from eden.training import train_eden

SEEDS = list(range(10))
EPOCHS = 50
BATCH = 64
from pathlib import Path
DATA_ROOT = Path("/kaggle/working/.data")

V1_MEAN = 0.9117
V1_STD  = 0.0110

tr_l, va_l, meta = datasets.get_torchvision_loaders("mnist", DATA_ROOT, BATCH)

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
print(f"\n===== RÉSULTAT FINAL (10 seeds) =====")
print(f"v2_full   mean={m:.4f}  std={s:.4f}  min={min(accs):.4f}  max={max(accs):.4f}")
print(f"\nvs v1     Δ mean={m - V1_MEAN:+.4f}  (v1={V1_MEAN:.4f} → v2={m:.4f})")
