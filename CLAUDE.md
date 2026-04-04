# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**EDEN** (Emergent Developmental Encoding Network) — a PyTorch research codebase that models biologically-inspired neural development: genome-driven topology, epigenetic masking, morphogen gradients, apoptosis, neurogenesis, and glia.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,vision,tui,neat,umap,config,profile]"
```

Environment variables: `EDEN_SEED` (default `42`), `EDEN_DATA` (default `.data`), `EDEN_RESULTS`.

## Commands

```bash
# Tests
pytest tests/ -v
pytest tests/test_genome.py -v          # single test file
pytest tests/ --cov=eden --cov-report=term-missing

# Smoke run
eden train --dataset synth --epochs 1 --dry-run

# Train
eden train --dataset mnist --epochs 50
eden train --dataset mnist --epochs 50 --eggroll   # Eggroll L1/L2 optimizers

# Checkpoint / resume
eden train --dataset mnist --epochs 20 --checkpoint-out eden_results/run.pt
eden train --dataset mnist --epochs 30 --resume eden_results/run.pt --checkpoint-out eden_results/run.pt

# Benchmark (fast)
eden benchmark --dataset mnist --epochs 2 --seeds 1 --quick --eden-only --output eden_results

# Ablation
eden ablation --mechanism contact_inhibition --seeds 2 --epochs 3 --dataset mnist

# Sensitivity
eden sensitivity --method grid --param tau --range 0.8,1.2 --epochs 3 --dataset mnist
eden sensitivity --method morris --epochs 1 --morris-epochs 1 --dataset mnist

# Full validation
bash scripts/run_full_validation.sh
```

Fallback when `eden` CLI is not on PATH: `python -m eden.cli.main <subcommand>`.

## Architecture

### Core data flow

1. **`HierarchicalGenome`** (`eden/core/genome.py`) — two subgenomes (150 type genes, 350 connectivity genes) that are masked by `HeritableEpigenome` (`eden/core/epigenome.py`). `genome_to_gates()` converts the genome into per-node gating scalars.

2. **`GeneRegulator`** (`eden/core/genome.py`) — eight `nn.Parameter` scalars (`tau`, `lambda_`, `k_soft`, `rho`, `paracrine_strength`, `drift_rate_scale`, `apoptosis_theta`, `entropy_theta`). These are the targets of Eggroll L2 and sensitivity sweeps.

3. **`EDENNetwork`** (`eden/core/network.py`) — the main `nn.Module`. A CNN embed stage feeds into a dynamic node stack governed by:
   - `StemPool` (`eden/core/stem.py`) — progenitor-style node pool with neurogenesis
   - `DifferentiationPhi` (`eden/core/differentiation.py`) — assigns cell-type identities
   - `ParacrineCascade` / `LocalMorphogen` (`eden/core/morphogen.py`) — k=3 paracrine signalling
   - `MicrogliaCompetition` / `ProgrammedApoptosis` (`eden/core/apoptosis.py`) — node pruning
   - `AstrocyteModulator` / `OligodendrocyteSheath` (`eden/core/glia.py`) — gain/myelination

4. **`AblationFlags`** (`eden/config.py`) — dataclass that enables/disables each biological mechanism; passed through to `EDENNetwork`. `TrainingState` tracks per-epoch metrics and events.

5. **Training loop** (`eden/training.py`) — `train_eden()` runs epochs, calls `eggroll_l1_step` / `eggroll_l2_step` (`eden/optim/`), computes FGSM robustness, saves full checkpoints (format tag `eden_train_v1`).

6. **Benchmarks** (`eden/benchmarks/`) — `datasets.py` wraps torchvision + synthetic loaders; `baselines.py` defines MLP, LeNet-5, ResNet-8, CNN1D, LSTM; `runner.py` orchestrates multi-seed runs and writes `benchmark_summary.json` + `benchmark_rows.csv`.

7. **CLI** (`eden/cli/main.py`) — Click entry-point with subcommands: `train`, `benchmark`, `ablation`, `sensitivity`, `interpret`, `tui`. Optional YAML overrides via `--config` (requires `[config]` extra).

8. **Interpretability** (`eden/interpret.py`) — UMAP/correlation helpers over saved checkpoints; requires `[umap]` extra.

### `SequenceEDENNetwork`

Variant of `EDENNetwork` for 1D sequence data (ECG, protein); uses a 1D CNN embed instead of 2D.

## Key conventions

- `--epochs` after `--resume` means *additional* epochs, not total.
- `--eden-only` on `benchmark` skips all baseline models (much faster for iteration).
- `--quick` flag reduces dataset size for fast CI-style runs.
- Benchmark outputs always land in `--output` dir as `benchmark_summary.json` and `benchmark_rows.csv`.
- Checkpoint format must match (`eden_train_v1`); weights-only files saved by `save_checkpoint` are not resumable with `--resume`.
