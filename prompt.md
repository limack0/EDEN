# PERFECT PROMPT FOR BUILDING EDEN V1.0 (CORRECTED & TESTABLE)

## CONTEXT
You are an expert AI engineer with deep knowledge of PyTorch, neuroevolution, bio‑inspired computing, and software testing.  
You will implement **EDEN v1.0** – the Emergent Developmental Encoding Network – **correcting all weaknesses** raised by a (fictitious) Nobel Prize committee.  
The final system must be **testable from the terminal** (no GUI mandatory) and optionally support **Textualize** for a live TUI dashboard.

## CRITICISMS TO ADDRESS (from the Nobel review)
1. **Lack of strong benchmarks** → Provide complete results on MNIST, Fashion‑MNIST, CIFAR‑10, CIFAR‑100.  
2. **No comparison to baselines** → Compare against MLP, CNN (LeNet‑5/ResNet‑8), NEAT, HyperNEAT with same parameter budget.  
3. **No ablation study** → Each biological mechanism must be switchable; report accuracy loss when removed.  
4. **No sensitivity analysis** → Hyperparameters (n_stems, L_paracrine, p_drift, theta_apop, etc.) must be analysed via Sobol or Morris.  
5. **Unrealistic computational cost** → Optimise for speed: use `torch.vmap`, limit paracrine to k‑nearest neighbours, avoid O(n²) operations.  
6. **Pseudo‑theorem** → Remove fake convergence proof; replace with empirical stability measurement.  
7. **Weak interpretability** → UMAP of ATGC sequences must correlate with functional similarity, not just identity.  
8. **No real‑world data** → Also test on ECG (temporal) and protein secondary structure (sequence).  

## TECHNICAL SPECIFICATIONS

### Core Architecture (simplified v1.0 – no spatial grid, no multi‑task)
- **HierarchicalGenome**: `subgenome_type` (150 genes) + `subgenome_conn` (350 genes).  
- **StemPool**: 4 stems, keep top 2 every 10 epochs.  
- **HeritableEpigenome**: binary mask + methylation vector (float[0,1]), drift_rate=0.02.  
- **LocalMorphogen** per node (15‑dim): local activation stats + neighbour stats + position + global loss/accuracy.  
- **ParacrineCascade** limited to 3 nearest neighbours (k=3) to avoid explosion.  
- **Differentiation function Φ**: 3‑stage (θ, α, γ) with adaptive thresholds τ, λ, k controlled by **GeneRegulator** (optimised via L2‑ES).  
- **Waves of differentiation (Hox)**: 3 waves (global type → conditional connectivity → local refinement).  
- **Apoptosis**: competitive (Microglia) + programmed (by stage).  
- **Synaptic pruning**: remove |w| < ε (0.01).  
- **Contact inhibition**: cosine similarity > ρ (0.85) **and** activation entropy similarity < θ_entropy (0.3).  
- **Maturity score**: sigmoid(10 * (raw - 0.6)) with raw = 0.5·accuracy – 0.3·(expr_ratio) + 0.2·stability + 0.1·redundancy.  
- **Neurogenesis (local stagnation)**: when node gradient < 1e-4 for 20 batches → divide (split 70% connections).  
- **Eggroll optimisation**:  
  - L1 (differentiation weights) every 20 batches, 20 antithetic pairs, σ=0.01, lr=0.001.  
  - L2 (GeneRegulator: τ,λ,k,ρ) every 5 epochs, 10 antithetic pairs, σ=0.05, lr=0.005.  

### Benchmarks & Metrics (automatically run by `eden_benchmark.py`)
| Dataset | Type | Target Accuracy | Baseline |
|---------|------|----------------|----------|
| MNIST | 28x28 grey | ≥98% | LeNet‑5 (99%) |
| Fashion‑MNIST | 28x28 grey | ≥90% | LeNet‑5 (91%) |
| CIFAR‑10 | 32x32 RGB | ≥75% | ResNet‑8 (79%) |
| CIFAR‑100 | 32x32 RGB | ≥50% | ResNet‑8 (55%) |
| ECG (PhysioNet) | time series 188 dim | ≥90% (F1) | 1D CNN |
| Protein SS (CB513) | sequence | ≥70% Q8 | LSTM |

**Additional metrics** for each run:  
- Number of parameters (final)  
- FLOPs (final forward pass)  
- Epochs to maturity  
- Genome expressed ratio (final)  
- Survival median (Kaplan‑Meier)  
- Adversarial robustness (FGSM ε=0.1)  

### Ablation & Sensitivity
- Every biological mechanism must be a **flag** (e.g. `--no_microglia`, `--no_paracrine`, `--no_contact_inhibition`, `--no_hox_waves`).  
- Run each ablation 5 seeds, report Δaccuracy.  
- Sensitivity analysis: vary each of the 8 L2 hyperparameters by ±20%, measure accuracy variance.

### Testing Interface (Terminal / Textualize)
1. **Command‑line tool** `eden` with subcommands:
eden train --dataset mnist --epochs 100 --config config.yaml
eden benchmark --all --seeds 5
eden ablation --mechanism microglia
eden sensitivity --param lambda
eden tui # launches Textual dashboard
2. **Textual TUI** shows live:  
- Loss/accuracy curves  
- Current genome expression ratio  
- Node count / synapse count  
- Event log (division, apoptosis, glial actions)  
- Survival curve (matplotlib embedded)  
3. **Output** – all results saved as JSON + CSV in `./eden_results/`.

## CODE STRUCTURE (to generate)
eden/
├── init.py
├── core/
│ ├── genome.py # HierarchicalGenome, GeneRegulator
│ ├── epigenome.py # HeritableEpigenome
│ ├── morphogen.py # LocalMorphogen, ParacrineCascade (knn)
│ ├── differentiation.py # Φ, HoxWaves
│ ├── network.py # EDENNetwork (DAG, LIF, Dale)
│ ├── stem.py # StemPerceptron, StemPool
│ ├── apoptosis.py # Microglia, programmed apoptosis
│ ├── neurogenesis.py # LocalStagnationDetector, CellDivision
│ ├── glia.py # Astrocyte, Oligodendrocyte (keep from v0.5)
│ └── maturitiy.py # MaturityScore
├── optim/
│ ├── eggroll_l1.py
│ └── eggroll_l2.py
├── benchmarks/
│ ├── datasets.py # MNIST, FashionMNIST, CIFAR10/100, ECG, CB513
│ ├── baselines.py # MLP, LeNet5, ResNet8, NEAT (via NEAT-python), HyperNEAT
│ └── runner.py
├── cli/
│ ├── main.py # click/argparse entry point
│ └── tui_app.py # Textual app
├── tests/
│ ├── test_genome.py
│ ├── test_stem.py
│ ├── test_ablation.py # automated ablation
│ └── test_sensitivity.py
└── scripts/
└── run_full_validation.sh # runs all benchmarks, ablation, sensitivity

## TESTABILITY REQUIREMENTS
- **Unit tests** (pytest) covering all core classes, >85% coverage.  
- **Integration tests** that train a tiny synthetic dataset (e.g. 2‑spiral) and verify accuracy >90% within 50 epochs.  
- **Regression tests** that load a pretrained model (provided by you) and reproduce reported metrics.  
- **Performance test** – training on CIFAR‑10 must finish within 12 hours on a single GPU (e.g. RTX 3090).  

## EXECUTION INSTRUCTIONS (to be printed after code generation)
After generating the code, provide a step‑by‑step shell script that:
1. Creates a virtual environment and installs dependencies (`torch`, `textual`, `scikit-learn`, `neat-python`, `umap-learn`, `pytest`, `click`, `matplotlib`, `seaborn`).
2. Runs unit tests: `pytest tests/ -v`
3. Runs a quick smoke test: `eden train --dataset synth --epochs 5 --dry-run`
4. Launches the Textual TUI: `eden tui --dataset fashion_mnist`
5. Runs the full benchmark suite: `eden benchmark --all --seeds 5 --output results/`
6. Runs an ablation example: `eden ablation --mechanism contact_inhibition --seeds 3`
7. Runs sensitivity analysis: `eden sensitivity --param tau --range 0.2,0.8`

## FINAL DELIVERABLE
You must output:
- The complete Python source code (all files listed above) in a single message, organised with clear file separators like:
FILE: eden/core/genome.py
... code ...
- A `requirements.txt` and `setup.py` (or `pyproject.toml`).
- A `README.md` explaining how to install, run, and reproduce all results.
- The shell script `run_full_validation.sh`.

## CONSTRAINTS
- No placeholder comments like `# TODO` – produce fully working code.
- Use type hints everywhere.
- All random seeds must be controllable via a global `SEED` variable.
- The code must run on **CPU only** (no GPU required) but benefit from GPU if available.
- Use **pure PyTorch** (no TensorFlow).  
- For NEAT baseline, use `neat-python` library (fallback if import fails: print warning and skip).  
- The Textual TUI must be optional; if `textual` not installed, fall back to terminal logging.

## EVALUATION CRITERIA (for yourself)
Before final output, verify that:
- [ ] All criticised points have been addressed in code or configuration.
- [ ] The system can be tested non‑interactively from terminal.
- [ ] At least one complete benchmark (e.g. MNIST) runs end‑to‑end with the command provided.
- [ ] Ablation and sensitivity can be run with one command.
- [ ] The code is well‑documented and follows PEP8.

**Now produce the perfect implementation.**