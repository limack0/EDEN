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

V1_MEAN = 0.6741  # 50 epochs, no augment, no BN
V2_MEAN = 0.6759  # 100 epochs, augment+BN, shallow embed

tr_l, va_l, meta = datasets.get_torchvision_loaders("cifar10", DATA_ROOT, BATCH)

model_test = EDENNetwork(int(meta["num_classes"]), int(meta["in_channels"]), tuple(meta["image_hw"]))
print(f"flat_dim={model_test.flat_dim}  hidden={model_test.hidden}  params={sum(p.numel() for p in model_test.parameters()):,}")
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
print(f"\n===== CIFAR-10 v3 — deeper embed ({len(SEEDS)} seeds, {EPOCHS} epochs) =====")
print(f"mean={m:.4f}  std={s:.4f}  min={min(accs):.4f}  max={max(accs):.4f}")
print(f"\nvs v1  Δ={m-V1_MEAN:+.4f}   vs v2  Δ={m-V2_MEAN:+.4f}")
