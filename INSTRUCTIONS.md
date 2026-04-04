# EDEN — Instructions

Step-by-step guide for installing and running **EDEN v1.0** on your machine. For design background, see `prompt.md`; for package overview, see `README.md`.

---

## 1. First-time setup

### 1.1 Create environment and install

**Windows (PowerShell), from the project root (`EDEN/`):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e ".[dev,vision]"
```

Optional extras: `tui`, `neat`, `umap`, **`config`** (YAML training overrides via `--config`), **`profile`** (forward FLOPs via `thop`). Example: `pip install -e ".[dev,vision,config,profile]"`.

### 1.2 Confirm the CLI

```powershell
eden --help
python -m eden.cli.main train --dataset synth --epochs 1 --dry-run
```

You should see JSON with `"dry_run_ok": true` and a short INFO log line.

---

## 2. Environment variables (optional)

| Variable      | Default | Meaning                          |
|---------------|---------|----------------------------------|
| `EDEN_SEED`   | `42`    | Global RNG seed                  |
| `EDEN_DATA`   | `.data` | Root for torchvision downloads   |
| `EDEN_RESULTS`| —       | Hint path for the TUI            |

**PowerShell example:**

```powershell
$env:EDEN_DATA = "D:\datasets\eden"
eden train --dataset mnist --epochs 5
```

---

## 3. Training (`eden train`)

### 3.1 Datasets

| `--dataset`     | Description                    |
|-----------------|--------------------------------|
| `mnist`         | MNIST 28×28                    |
| `fashion_mnist` | Fashion-MNIST                  |
| `cifar10`       | CIFAR-10                       |
| `cifar100`      | CIFAR-100                      |
| `synth`         | 2D synthetic spiral (2 classes)|
| `ecg`           | Synthetic ECG-like series      |
| `protein`       | Synthetic protein-style seq.   |

### 3.2 Common commands

```powershell
# Short smoke run with logs saved under eden_results\
eden train --dataset mnist --epochs 5 --output eden_results --save-log

# Enable spec-style Eggroll optimizers (slower)
eden train --dataset mnist --epochs 50 --eggroll

# Verbose + per-batch DEBUG every 100 steps
eden train --dataset mnist --epochs 2 -v --log-batch-every 100
```

### 3.3 Checkpoints and resume

- **`--checkpoint-out FILE.pt`** — After training, writes full state (model, regulator, epigenome, optimizer). Format: `eden_train_v1`.
- **`--checkpoint-every N`** — Also save every **N** epochs (must be used with `--checkpoint-out`).
- **`--resume FILE.pt`** — Continue from a full checkpoint. **`--epochs` means extra epochs**, not “total from zero”.

Example:

```powershell
eden train --dataset mnist --epochs 20 --checkpoint-out eden_results\run.pt --output eden_results
eden train --dataset mnist --epochs 30 --resume eden_results\run.pt --checkpoint-out eden_results\run.pt --output eden_results
```

Use the **same** `--dataset` (and layout) when resuming so the model architecture matches.

### 3.4 YAML overrides (`--config`)

Requires **`pip install -e ".[config]"`** (pulls in PyYAML). CLI flags still apply; values from the YAML file are merged **first**, then overridden by explicit options.

```powershell
eden train --dataset mnist --config examples\train.example.yaml --epochs 10 --output eden_results
```

### 3.5 Logging flags

- Default: **INFO** on the console (epoch summaries).
- **`-q` / `--quiet`** — Warnings and errors only.
- **`-v` / `--verbose`** — DEBUG (use with `--log-batch-every`).
- **`--save-log`** — Append to `<output>\train.log`.
- **`--log-file PATH`** — Append to a chosen file.

---

## 4. Benchmarks (`eden benchmark`)

```powershell
# One dataset, one seed, fast EDEN-only (no baselines)
eden benchmark --dataset mnist --epochs 2 --seeds 1 --quick --eden-only --output eden_results

# MNIST + baselines (MLP, LeNet), longer run
eden benchmark --dataset mnist --epochs 5 --seeds 1 --output eden_results
```

Outputs:

- `eden_results\benchmark_summary.json` — Summary, all rows, and `"meta": { "eden_only": ... }`.
- `eden_results\benchmark_rows.csv` — Flat table for spreadsheets.

**`--eden-only`** skips MLP / LeNet / ResNet / CNN1D / LSTM and only trains EDEN.

Benchmark logging: same idea as train (`--save-log`, `-v`, `-q`, `--log-file`, `--log-batch-every`).

---

## 5. Ablation, sensitivity, interpret

```powershell
eden ablation --mechanism contact_inhibition --seeds 2 --epochs 3 --dataset mnist
eden sensitivity --method grid --param tau --range 0.8,1.2 --epochs 3 --dataset mnist
# Morris elementary effects (more EDEN runs; keep --morris-epochs small while iterating)
eden sensitivity --method morris --epochs 1 --morris-epochs 1 --dataset mnist --morris-trajectories 4
```

**Mechanisms:** `microglia`, `paracrine`, `contact_inhibition`, `hox_waves`, `programmed_apoptosis`, `neurogenesis`, `epigenome_drift`, `synaptic_pruning`, `glia`

**Sensitivity:** `--method grid` uses **`--param`** and **`--range`** (`tau`, `lambda`, `k`, `rho`, `paracrine`, `drift`, `apoptosis`, `entropy`). **`--method morris`** varies regulator hyperparameters in trajectories (see `eden sensitivity --help`).

**Interpretability** (needs `[umap]`): `eden interpret --help` — correlation / projection helpers over saved checkpoints or in-process models.

---

## 6. TUI (optional)

Requires `pip install textual` (or install with `[tui]`).

```powershell
eden tui --dataset fashion_mnist
```

If Textual is missing, the command falls back to a short message.

---

## 7. Tests

```powershell
python -m pytest tests/ -v
```

---

## 8. Troubleshooting

| Issue | What to try |
|-------|-------------|
| `eden` not found | Use `python -m eden.cli.main ...` or ensure `.venv\Scripts` is on `PATH`. |
| torchvision / download errors | Set `EDEN_DATA` to a writable folder; check network. |
| CUDA | Install a CUDA build of PyTorch if you want GPU; CPU works without it. |
| Resume error about checkpoint format | Use a file saved with **`--checkpoint-out`** from this project, not the minimal `save_checkpoint` weights-only file. |
| Benchmark very slow | Add **`--eden-only`** and/or **`--quick`**. |

---

## 9. Project map

| Path | Role |
|------|------|
| `eden/core/` | Genome, network, stems, morphogens, training hooks |
| `eden/training.py` | Main training loop, FGSM metric, full checkpoints |
| `eden/benchmarks/` | Datasets, baselines, benchmark runner |
| `eden/cli/main.py` | All `eden` subcommands |
| `tests/` | Pytest suite |
| `scripts/run_full_validation.sh` | Full validation (Unix shell) |

---

*End of instructions.*
