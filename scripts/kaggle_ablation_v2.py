"""
Ablation: node_attention vs heterogeneous stems — lesquels contribuent au +7.93% MNIST ?

4 configurations × 5 seeds × 50 epochs
"""
import subprocess, sys
from pathlib import Path

subprocess.run(["rm", "-rf", "/kaggle/working/EDEN"], check=True)
subprocess.run(["git", "clone", "https://github.com/limack0/EDEN", "/kaggle/working/EDEN"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-e", "/kaggle/working/EDEN", "-q"], check=True)

import statistics
sys.path.insert(0, "/kaggle/working/EDEN")

import torch
from eden.benchmarks import datasets
from eden.config import AblationFlags, set_seed
from eden.core.epigenome import HeritableEpigenome
from eden.core.genome import GeneRegulator
from eden.core.network import EDENNetwork
from eden.core.stem import StemPool
from eden.training import train_eden

SEEDS     = list(range(5))
EPOCHS    = 50
BATCH     = 64
DATA_ROOT = Path("/kaggle/working/.data")

tr_l, va_l, meta = datasets.get_torchvision_loaders("mnist", DATA_ROOT, BATCH)

def make_model(node_attention: bool, heterogeneous: bool) -> EDENNetwork:
    flags = AblationFlags(node_attention=node_attention)
    m = EDENNetwork(
        int(meta["num_classes"]),
        int(meta["in_channels"]),
        tuple(meta["image_hw"]),
        flags=flags,
    )
    # override stem heterogeneity
    m.stem_pool.heterogeneous = heterogeneous
    if not heterogeneous:
        import torch.nn as nn
        from eden.core.stem import StemPerceptron
        m.stem_pool.stems = nn.ModuleList(
            StemPerceptron(m.stem_pool.in_dim, m.stem_pool.stem_hidden) for _ in range(4)
        )
    return m

configs = {
    "no_attn_no_hetero":  dict(node_attention=False, heterogeneous=False),
    "attn_only":          dict(node_attention=True,  heterogeneous=False),
    "hetero_only":        dict(node_attention=False, heterogeneous=True),
    "attn_and_hetero":    dict(node_attention=True,  heterogeneous=True),
}

results: dict[str, list[float]] = {k: [] for k in configs}

for name, cfg in configs.items():
    for seed in SEEDS:
        set_seed(seed)
        model = make_model(**cfg)
        r = train_eden(model, GeneRegulator(), HeritableEpigenome(), tr_l, va_l, EPOCHS, seed=seed)
        acc = r["final_val_accuracy"]
        results[name].append(acc)
        print(f"{name:<22} seed={seed}  acc={acc:.4f}")
    print()

print("===== ABLATION node_attention vs heterogeneous stems (MNIST) =====")
for name, vals in results.items():
    m = statistics.mean(vals)
    s = statistics.stdev(vals)
    print(f"{name:<22} mean={m:.4f}  std={s:.4f}")
