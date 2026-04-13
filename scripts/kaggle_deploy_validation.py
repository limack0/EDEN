"""
EDEN — Validation complète pré-déploiement
==========================================
Gaps couverts en un seul run :

  1. Ablation CIFAR-10 : node_attention × hetero stems (3 seeds, 50 ep)
     → est-ce eux qui font les 86.72% ou bien le training / CNN ?

  2. Ablation ECG      : node_attention × hetero stems (5 seeds, 100 ep)
     → même question sur données séquentielles

  3. Latence inférence : GPU + CPU, batch=1 et batch=32
     → MNIST / CIFAR-10 / ECG

Durée estimée : ~5h GPU (T4 Kaggle)

Résultats connus pour comparaison :
  MNIST ablation (seeds 0-4, 50 ep) :
    no_attn_no_hetero  0.9935  attn_and_hetero  0.9932   → différences dans le bruit
  CIFAR-10 full (5 seeds, 100 ep)   : mean=0.8672  std=0.0014
  ECG full      (5 seeds, 100 ep)   : mean=0.9232  std=0.0104
"""
import subprocess, sys, time, statistics
from pathlib import Path

# ── Setup ─────────────────────────────────────────────────────────────────────
subprocess.run(["rm", "-rf", "/kaggle/working/EDEN"], check=True)
subprocess.run(["git", "clone", "https://github.com/limack0/EDEN", "/kaggle/working/EDEN"], check=True)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", "/kaggle/working/EDEN[vision]", "-q"],
    check=True,
)
sys.path.insert(0, "/kaggle/working/EDEN")

import torch
import torch.nn as nn

print("CUDA:", torch.cuda.is_available())
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from eden.benchmarks.datasets import get_torchvision_loaders, get_sequence_loaders
from eden.config import AblationFlags, set_seed
from eden.core.epigenome import HeritableEpigenome
from eden.core.genome import GeneRegulator
from eden.core.network import EDENNetwork, SequenceEDENNetwork
from eden.core.stem import StemPerceptron
from eden.training import train_eden

DATA_ROOT = Path("/kaggle/working/.data")

ABLATION_CONFIGS: dict[str, dict] = {
    "no_attn_no_hetero": dict(node_attention=False, heterogeneous=False),
    "attn_only":         dict(node_attention=True,  heterogeneous=False),
    "hetero_only":       dict(node_attention=False, heterogeneous=True),
    "attn_and_hetero":   dict(node_attention=True,  heterogeneous=True),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _apply_hetero(model: nn.Module, heterogeneous: bool) -> None:
    """Override stem heterogeneity on an already-built model."""
    pool = model.stem_pool
    pool.heterogeneous = heterogeneous
    if not heterogeneous:
        pool.stems = nn.ModuleList(
            StemPerceptron(pool.in_dim, pool.stem_hidden) for _ in range(4)
        )


def _make_vision(meta: dict, node_attention: bool, heterogeneous: bool) -> EDENNetwork:
    m = EDENNetwork(
        int(meta["num_classes"]),
        int(meta["in_channels"]),
        tuple(meta["image_hw"]),
        flags=AblationFlags(node_attention=node_attention),
    )
    _apply_hetero(m, heterogeneous)
    return m


def _make_sequence(meta: dict, node_attention: bool, heterogeneous: bool) -> SequenceEDENNetwork:
    m = SequenceEDENNetwork(
        int(meta["num_classes"]),
        int(meta["seq_len"]),
        flags=AblationFlags(node_attention=node_attention),
    )
    _apply_hetero(m, heterogeneous)
    return m


def run_ablation(
    label: str,
    make_fn,          # callable(meta, node_attention, heterogeneous) -> model
    meta: dict,
    tr_l,
    va_l,
    seeds: list,
    epochs: int,
) -> dict[str, list[float]]:
    results: dict[str, list[float]] = {k: [] for k in ABLATION_CONFIGS}
    for name, cfg in ABLATION_CONFIGS.items():
        for seed in seeds:
            set_seed(seed)
            model = make_fn(meta, cfg["node_attention"], cfg["heterogeneous"])
            r = train_eden(
                model, GeneRegulator(), HeritableEpigenome(), tr_l, va_l, epochs, seed=seed
            )
            acc = r["final_val_accuracy"]
            results[name].append(acc)
            print(f"[{label}] {name:<22} seed={seed}  acc={acc:.4f}")
        print()
    return results


def print_ablation_summary(label: str, results: dict[str, list[float]]) -> None:
    print(f"{'='*60}")
    print(f"ABLATION {label}")
    print(f"{'='*60}")
    baseline = statistics.mean(results["no_attn_no_hetero"])
    for name, vals in results.items():
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        delta = m - baseline
        mark = "  ← best" if m == max(statistics.mean(v) for v in results.values()) else ""
        print(f"  {name:<22} mean={m:.4f}  std={s:.4f}  Δbaseline={delta:+.4f}{mark}")
    print()


def measure_latency(model: nn.Module, sample: torch.Tensor, label: str) -> None:
    model.eval()
    bs = sample.shape[0]

    # GPU
    if torch.cuda.is_available():
        m = model.to("cuda")
        x = sample.to("cuda")
        with torch.no_grad():
            for _ in range(50):   # warmup
                m(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(200):
                m(x)
        torch.cuda.synchronize()
        gpu_ms = (time.perf_counter() - t0) * 1000 / 200
        print(
            f"  {label:<28} GPU  {gpu_ms:6.2f} ms/batch"
            f"  {gpu_ms/bs*1000:7.1f} µs/sample"
            f"  {bs/gpu_ms*1000:8.0f} samples/s"
        )

    # CPU
    m_cpu = model.to("cpu")
    x_cpu = sample.to("cpu")
    with torch.no_grad():
        for _ in range(5):
            m_cpu(x_cpu)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(30):
            m_cpu(x_cpu)
    cpu_ms = (time.perf_counter() - t0) * 1000 / 30
    print(
        f"  {label:<28} CPU  {cpu_ms:6.2f} ms/batch"
        f"  {cpu_ms/bs*1000:7.1f} µs/sample"
        f"  {bs/cpu_ms*1000:8.0f} samples/s"
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Ablation CIFAR-10  (~4h)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 1 — Ablation CIFAR-10 (3 seeds, 50 epochs)")
print("="*60 + "\n")

tr_cifar, va_cifar, meta_cifar = get_torchvision_loaders("cifar10", DATA_ROOT, batch_size=128)
cifar_results = run_ablation(
    "CIFAR-10",
    _make_vision,
    meta_cifar,
    tr_cifar, va_cifar,
    seeds=list(range(3)),
    epochs=50,
)
print_ablation_summary("CIFAR-10  (3 seeds, 50 ep)  — full=86.72%", cifar_results)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Ablation ECG  (~30min)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 2 — Ablation ECG (5 seeds, 100 epochs)")
print("="*60 + "\n")

tr_ecg, va_ecg, meta_ecg = get_sequence_loaders("ecg", batch_size=64, seed=42)
ecg_results = run_ablation(
    "ECG",
    _make_sequence,
    meta_ecg,
    tr_ecg, va_ecg,
    seeds=list(range(5)),
    epochs=100,
)
print_ablation_summary("ECG  (5 seeds, 100 ep)  — full=92.32%", ecg_results)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Latence inférence  (~5min)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 3 — Latence inférence (GPU + CPU)")
print("="*60 + "\n")
print(f"  {'label':<28} {'device':<5}  {'ms/batch':>10}  {'µs/sample':>12}  {'samples/s':>12}")
print("  " + "-"*68)

set_seed(0)

# MNIST
_, va_mnist, meta_mnist = get_torchvision_loaders("mnist", DATA_ROOT, batch_size=64)
model_mnist = EDENNetwork(
    int(meta_mnist["num_classes"]), int(meta_mnist["in_channels"]), tuple(meta_mnist["image_hw"])
)
batch_all_mnist = next(iter(va_mnist))[0]

for bs in (1, 32):
    measure_latency(model_mnist, batch_all_mnist[:bs], f"MNIST       batch={bs}")

# CIFAR-10
model_cifar_lat = EDENNetwork(
    int(meta_cifar["num_classes"]), int(meta_cifar["in_channels"]), tuple(meta_cifar["image_hw"])
)
batch_all_cifar = next(iter(va_cifar))[0]

for bs in (1, 32):
    measure_latency(model_cifar_lat, batch_all_cifar[:bs], f"CIFAR-10    batch={bs}")

# ECG
model_ecg_lat = SequenceEDENNetwork(int(meta_ecg["num_classes"]), int(meta_ecg["seq_len"]))
batch_all_ecg = next(iter(va_ecg))[0]

for bs in (1, 32):
    measure_latency(model_ecg_lat, batch_all_ecg[:bs], f"ECG         batch={bs}")


# ══════════════════════════════════════════════════════════════════════════════
# RÉSUMÉ FINAL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("RÉSUMÉ — Questions pré-déploiement")
print("="*60)

def _winner(res: dict[str, list[float]]) -> str:
    return max(res, key=lambda k: statistics.mean(res[k]))

def _gap(res: dict[str, list[float]]) -> float:
    vals = [statistics.mean(v) for v in res.values()]
    return max(vals) - min(vals)

w_c = _winner(cifar_results)
w_e = _winner(ecg_results)
g_c = _gap(cifar_results)
g_e = _gap(ecg_results)

print(f"\n  CIFAR-10 : meilleure config = {w_c}  (écart max={g_c:.4f})")
print(f"  ECG      : meilleure config = {w_e}  (écart max={g_e:.4f})")

NOISE = 0.005  # seuil sous lequel les différences sont considérées comme du bruit

if g_c < NOISE and g_e < NOISE:
    print("\n  → Les 4 configs sont dans le bruit sur CIFAR-10 ET ECG.")
    print("    Le gain vient du training et du CNN, pas de attn/hetero.")
    print("    CONCLUSION : EDEN peut être déployé avec la config minimale")
    print("    (no_attn_no_hetero) sans perte mesurable.")
elif g_c >= NOISE and g_e < NOISE:
    print(f"\n  → attn/hetero contribuent sur CIFAR-10 (écart={g_c:.4f} > {NOISE}).")
    print("    Sur ECG c'est dans le bruit. Garder attn_and_hetero pour CIFAR-10.")
elif g_c < NOISE and g_e >= NOISE:
    print(f"\n  → attn/hetero contribuent sur ECG (écart={g_e:.4f} > {NOISE}).")
    print("    Sur CIFAR-10 c'est dans le bruit.")
else:
    print(f"\n  → attn/hetero contribuent sur les deux datasets.")
    print(f"    CIFAR-10 écart={g_c:.4f}  ECG écart={g_e:.4f}")

print("\n===== VALIDATION COMPLÈTE TERMINÉE =====\n")
