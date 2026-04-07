"""ECG benchmark — v1 (SequenceEDENNetwork, synthetic PhysioNet-like data).

Dataset: SyntheticECGDataset — 2500 samples, 188-dim time series, binary label
(arrhythmia classification, CPU-friendly).

Baselines:
  Logistic regression: ~82%
  1D CNN (baseline): ~89%
  LSTM (baseline): ~87%
  EDEN target: ≥88%
"""
import subprocess, sys
from pathlib import Path

subprocess.run(["rm", "-rf", "/kaggle/working/EDEN"], check=True)
subprocess.run(["git", "clone", "https://github.com/limack0/EDEN", "/kaggle/working/EDEN"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-e", "/kaggle/working/EDEN[vision]", "-q"], check=True)

import statistics
sys.path.insert(0, "/kaggle/working/EDEN")

import torch
print("CUDA:", torch.cuda.is_available())

from eden.benchmarks.datasets import get_sequence_loaders
from eden.config import AblationFlags, set_seed
from eden.core.epigenome import HeritableEpigenome
from eden.core.genome import GeneRegulator
from eden.core.network import SequenceEDENNetwork
from eden.training import train_eden

SEEDS     = list(range(5))
EPOCHS    = 100
BATCH     = 64

tr_l, va_l, meta = get_sequence_loaders("ecg", BATCH, seed=42)
print(f"classes={meta['num_classes']}  seq_len={meta['seq_len']}")

model_test = SequenceEDENNetwork(int(meta["num_classes"]), int(meta["seq_len"]))
params = sum(p.numel() for p in model_test.parameters())
print(f"hidden={model_test.hidden}  params={params:,}")
del model_test

accs = []
for seed in SEEDS:
    set_seed(seed)
    model = SequenceEDENNetwork(
        int(meta["num_classes"]),
        int(meta["seq_len"]),
        flags=AblationFlags(),
    )
    r = train_eden(model, GeneRegulator(), HeritableEpigenome(), tr_l, va_l, EPOCHS, seed=seed)
    acc = r["final_val_accuracy"]
    accs.append(acc)
    print(f"seed={seed}  acc={acc:.4f}")

m = statistics.mean(accs)
s = statistics.stdev(accs)
print(f"\n===== ECG v1 — SequenceEDEN ({len(SEEDS)} seeds, {EPOCHS} epochs) =====")
print(f"mean={m:.4f}  std={s:.4f}  min={min(accs):.4f}  max={max(accs):.4f}")
print(f"\nRéférence: LogReg~0.82 | CNN1D~0.89 | LSTM~0.87")
