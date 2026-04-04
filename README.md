# EDEN v1.0

**Emergent Developmental Encoding Network** — a PyTorch research codebase with benchmarks, ablations, sensitivity tooling, and an optional Textual TUI.

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev,vision,tui,neat,umap,config,profile]"
```

Optional groups: **`config`** (YAML `--config` for `eden train`), **`profile`** (`thop` forward FLOPs in reports when installed).

Environment variables:

- `EDEN_SEED` — global seed (default `42`)
- `EDEN_DATA` — torchvision root (default `.data`)
- `EDEN_RESULTS` — result directory hint for the TUI

## Logging

Training prints **INFO** lines by default (epoch summary: train/val loss and accuracy, maturity, active stems, etc.). Baseline runs in `eden benchmark` log each epoch.

- `-q` / `--quiet` — warnings only  
- `-v` / `--verbose` — DEBUG (e.g. per-batch with `--log-batch-every N`)  
- `--save-log` — append to `<output>/train.log` (train) or `benchmark.log` (benchmark)  
- `--log-file PATH` — append to a chosen file  

Using `train_eden()` from Python without the CLI: configure logging yourself, e.g. `logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")`.

### Checkpoints and faster benchmarks

- **`--eden-only`** on `eden benchmark` trains only EDEN (no MLP/LeNet/ResNet/CNN/LSTM). Much faster for iteration.
- **`--checkpoint-out path.pt`** saves full training state after `eden train` (model, regulator, epigenome, optimizer).
- **`--checkpoint-every N`** with `--checkpoint-out` saves every N epochs (crash recovery).
- **`--resume path.pt`** continues training; **`--epochs`** then means *additional* epochs, not total from zero.

## CLI

```bash
eden train --dataset mnist --epochs 100
eden train --dataset mnist --epochs 100 --eggroll
eden train --dataset synth --epochs 5 --dry-run
eden benchmark --all --seeds 5 --output eden_results/
eden benchmark --dataset mnist --seeds 1 --epochs 5
eden benchmark --dataset mnist --quick --epochs 2 --eden-only --output eden_results
eden ablation --mechanism microglia --seeds 3
eden sensitivity --method grid --param lambda --range 0.8,1.2
eden sensitivity --method morris --dataset mnist --epochs 1 --morris-epochs 1
eden interpret --help
eden tui --dataset fashion_mnist
```

`eden_benchmark.py` forwards to `eden benchmark` with the same arguments.

### Ablation mechanisms

`microglia`, `paracrine`, `contact_inhibition`, `hox_waves`, `programmed_apoptosis`, `neurogenesis`, `epigenome_drift`, `synaptic_pruning`, `glia`

### Sensitivity parameters (`--param`)

Maps to `GeneRegulator`: `tau`, `lambda`, `k`, `rho`, `paracrine`, `drift`, `apoptosis`, `entropy`

## Layout

- `eden/core/` — genome, epigenome, morphogens (k=3 paracrine), differentiation Φ, network, stems, apoptosis, glia, maturity
- `eden/optim/` — Eggroll L1/L2
- `eden/benchmarks/` — datasets, baselines (MLP, LeNet-5, ResNet-8, 1D CNN, LSTM), runner
- `eden/cli/` — Click entry + optional Textual app
- `tests/` — pytest suite

## Validation script

```bash
bash scripts/run_full_validation.sh
```

On Windows without bash, run the commands inside that file manually in PowerShell.

## Notes

- Training targets in the design doc are **aspirational**; default epochs in CLI are modest for CI and laptops. Increase `--epochs` for serious runs.
- NEAT/HyperNEAT baselines: `neat-python` is optional; heavy NEAT runs are stubbed with a warning.
- Empirical **stability** is reported via loss CV; there is **no** closed-form convergence proof by design.
- **UMAP** interpretability helpers live in `eden/interpret.py` when `umap-learn` and `scipy` are installed.

## License

Research / educational use — add your license as needed.
