# EDEN — Emergent Developmental Encoding Network

> A PyTorch neural network that *grows* from a genome instead of being hand-designed.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License: Research](https://img.shields.io/badge/license-research-lightgrey)](LICENSE)

---

## What is EDEN?

EDEN encodes **biological development rules** — genome, epigenome, cell differentiation, competition, apoptosis — inside a trainable PyTorch module. Unlike conventional architectures with fixed topology, EDEN's internal structure is shaped dynamically by these rules during training.

**Key idea**: the mechanisms that make biological brains robust (competitive selection, stochastic masking, cell diversity) transfer to gradient-based learning — but only some of them.

---

## Results

### MNIST (10 seeds, 50 epochs, GPU)

| Model | Val acc | FGSM robust (ε=0.1) | Params |
|-------|---------|---------------------|--------|
| MLP | 98.14% | — | 0.5M |
| LeNet-5 | 99.01% | — | 0.06M |
| **EDEN v2** | **99.10%** | **100%** | 5.4M |

EDEN correctly classifies **24/24** adversarial FGSM examples.  
Variance across seeds: std=**0.0018** (÷6 vs v1).

### CIFAR-10 (5 seeds, 100 epochs, GPU)

| Version | Mean | Std | Notes |
|---------|------|-----|-------|
| v2 (2-conv embed) | 67.59% | 0.78% | stable |
| **v4 (3-conv + AdaptPool + grad clip)** | **86.72%** | **0.14%** | +19.13% vs v2 |

EDEN v4 surpasses ResNet-8 (~85%) on CIFAR-10 with 4M params and std=**0.0014** (5 seeds).

### Fashion-MNIST (5 seeds, 50 epochs, GPU)

| Model | Val acc |
|-------|---------|
| MLP | ~88.0% |
| LeNet | ~91.0% |
| **EDEN v2** | **92.34%** |
| ResNet | ~94.0% |

Variance across seeds: std=**0.0015** (min=92.14%, max=92.55%).

### ECG (5 seeds, 100 epochs, SequenceEDEN)

| Model | Val acc |
|-------|---------|
| LogReg | ~82.0% |
| LSTM | ~87.0% |
| CNN-1D | ~89.0% |
| **EDEN v1** | **92.32%** |

Variance across seeds: std=**0.0104** (min=90.93%, max=93.60%).

### Ablation: node_attention vs heterogeneous stems (MNIST, 5 seeds, 50 epochs)

| Configuration | Mean | Std |
|---------------|------|-----|
| no_attn_no_hetero | 99.35% | 0.0007 |
| attn_only | 99.34% | 0.0005 |
| **hetero_only** | **99.39%** | 0.0006 |
| attn_and_hetero | 99.32% | 0.0004 |

All differences are within noise (±0.04%). The v1→v2 gain (+7.93%) is explained by **removing neurogenesis and paracrine** (confirmed by ECG ablation) and training improvements (AdamW, warmup, cosine, grad_clip). Heterogeneous stems are nominally best in isolation; combining both mechanisms minimizes variance (std=0.0004).

### ECG ablation (3 seeds, 50 epochs)

| Mechanism | Δacc vs full | Status |
|-----------|-------------|--------|
| neurogenesis | +1.87% when removed | disabled |
| paracrine | +1.51% when removed | disabled |
| glia | +0.44% when removed | kept (low cost) |
| epigenome_drift | −0.71% when removed | kept (beneficial) |

---

## Architecture

```
Input (B, C, H, W)
  │
  ▼
CNN Embed (adaptive)
  ├─ [H < 32]  Conv×2 + BN + ReLU + MaxPool×2   → flat_dim = 64·(H/4)²
  └─ [H ≥ 32]  Conv×3 + BN + ReLU + AdaptPool4  → flat_dim = 2048

StemPool (4 heterogeneous stems, top-2 competition)
  ├─ Stem 0: Linear → ReLU → Linear
  ├─ Stem 1: Linear → GELU → Linear
  ├─ Stem 2: Linear → SiLU → Linear
  └─ Stem 3: Linear → ReLU → Linear + skip

node_proj → (B, 8, hidden)

LocalMorphogen   15-dim signal from loss/acc/activations
NodeAttention    MHA(4 heads) + residual + LayerNorm
DifferentiationPhi  θ → Dropout → α → Dropout → γ  (genome-gated)
Glia             AstrocyteModulator + OligodendrocyteSheath
Apoptosis        MicrogliaCompetition + ProgrammedApoptosis
ContactInhibition  cosine penalty between similar nodes

mean-pool → Linear → logits
```

**Biological components:**

| Component | Biological analog |
|-----------|------------------|
| `HierarchicalGenome` | DNA — 500 genes (150 type + 350 connectivity) |
| `HeritableEpigenome` | Methylation — stochastic gene masking with drift |
| `GeneRegulator` | Transcription factors — 8 learned scalars (τ, λ, k, ρ, ...) |
| `StemPool` | Pluripotent stem cells — heterogeneous, competitive |
| `NodeAttention` | Inter-cellular coordination — learned, not forced |
| `DifferentiationPhi` | Cell differentiation — 3 genome-gated transformations |
| `LocalMorphogen` | Morphogen gradients — 15-dim training state signal |

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev,vision]"
```

Optional extras: `tui` (Textual UI), `umap` (interpretability), `config` (YAML CLI), `profile` (FLOPs).

Environment variables: `EDEN_SEED` (default `42`), `EDEN_DATA` (default `.data`).

---

## Quick start

```bash
# Smoke test (CPU, 1 epoch)
eden train --dataset synth --epochs 1 --dry-run

# Train on MNIST
eden train --dataset mnist --epochs 50

# Train on CIFAR-10
eden train --dataset cifar10 --epochs 100

# Benchmark EDEN vs MLP/LeNet/ResNet (5 seeds)
eden benchmark --dataset mnist --seeds 5 --output results/

# Ablation: test impact of one mechanism
eden ablation --mechanism neurogenesis --seeds 3 --epochs 50 --dataset mnist

# Sensitivity sweep
eden sensitivity --method grid --param tau --range 0.8,1.2 --epochs 5 --dataset mnist

# Resume a checkpoint
eden train --dataset mnist --epochs 20 --checkpoint-out run.pt
eden train --dataset mnist --epochs 30 --resume run.pt --checkpoint-out run.pt
```

---

## Python API

**Training:**
```python
from eden.core.network import EDENNetwork
from eden.core.genome import GeneRegulator
from eden.core.epigenome import HeritableEpigenome
from eden.training import train_eden
from eden.benchmarks.datasets import get_torchvision_loaders

tr_l, va_l, meta = get_torchvision_loaders("mnist", ".data", batch_size=128)

model = EDENNetwork(num_classes=10, in_channels=1, image_hw=(28, 28))
result = train_eden(model, GeneRegulator(), HeritableEpigenome(), tr_l, va_l, epochs=50)
print(f"val_acc={result['final_val_accuracy']:.4f}")
```

**Inference (production):**
```python
from eden import EDENPredictor
from eden.core.network import EDENNetwork
from eden.core.genome import GeneRegulator
from eden.core.epigenome import HeritableEpigenome

# Build and load weights
model     = EDENNetwork(num_classes=10, in_channels=1, image_hw=(28, 28))
regulator = GeneRegulator()
epigenome = HeritableEpigenome()

predictor = EDENPredictor.from_checkpoint("run.pt", model, regulator, epigenome, device="cuda")

# Single sample or batch — regulator/epigenome overhead is pre-paid
probs  = predictor.predict_proba(x)   # (B, C) float32
labels = predictor.predict(x)         # (B,)   int64
```

---

## CLI reference

```
eden train      --dataset {mnist,cifar10,fashion_mnist,synth}  --epochs N
                [--eggroll] [--dry-run] [--checkpoint-out PATH] [--resume PATH]
                [-q/-v] [--seed N] [--config file.yaml]

eden benchmark  --dataset DATASET  --seeds N  --epochs N  --output DIR
                [--eden-only] [--quick]

eden ablation   --mechanism NAME  --seeds N  --epochs N  --dataset DATASET

eden sensitivity --method {grid,morris}  --param NAME  --range LO,HI
                 --epochs N  --dataset DATASET

eden interpret  --checkpoint PATH  [--umap] [--correlation]

eden tui        --dataset DATASET
```

Ablation mechanisms: `microglia`, `paracrine`, `contact_inhibition`, `hox_waves`,
`programmed_apoptosis`, `neurogenesis`, `epigenome_drift`, `synaptic_pruning`, `glia`

Sensitivity params: `tau`, `lambda`, `k`, `rho`, `paracrine`, `drift`, `apoptosis`, `entropy`

---

## Project layout

```
eden/
├── core/          genome, epigenome, morphogens, differentiation, network, stems, apoptosis, glia
├── optim/         Eggroll L1/L2 (evolutionary parameter search)
├── benchmarks/    datasets (vision + synthetic ECG/protein), baselines, runner
├── cli/           Click entry-point + optional Textual TUI
└── interpret.py   UMAP / correlation interpretability helpers

scripts/
├── kaggle_cifar10_v4.py   CIFAR-10 validation (current arch)
├── kaggle_fashion_mnist.py
├── kaggle_ablation_v2.py
└── run_full_validation.sh
tests/             pytest suite
```

---

## Reproducing results

```bash
# MNIST multi-seed
eden benchmark --dataset mnist --seeds 10 --epochs 50 --eden-only --output results/

# CIFAR-10
eden train --dataset cifar10 --epochs 100

# Ablation (ECG)
eden ablation --mechanism neurogenesis --seeds 3 --epochs 50 --dataset ecg
```

Or use the Kaggle scripts in `scripts/` for GPU runs without local hardware.

---

## Design philosophy

**What transfers from biology to gradient descent:**
- Competitive selection between representations (StemPool top-k)
- Forced structural diversity (heterogeneous stems, ContactInhibition)
- Learned coordination (NodeAttention — unlike forced paracrine mixing)
- Stochastic exploration (epigenome_drift)

**What does not transfer:**
- Dynamic growth (neurogenesis) — new parameters with uninitialized Adam moments destabilize the optimizer
- Local synchronization (paracrine) — works against ContactInhibition; `paracrine_strength` converges to 0 on its own

---

## License

Research / educational use. See `LICENSE`.

---

*EDEN v2.0 — [limack0](https://github.com/limack0/EDEN) — 2026*
