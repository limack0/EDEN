"""
EDEN — Benchmark latence inférence
====================================
Mesure GPU + CPU, batch=1 et batch=32 sur MNIST / CIFAR-10 / ECG.
Lance ce script séparément après kaggle_deploy_validation.py.
"""
import subprocess, sys, time
from pathlib import Path

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

from eden.benchmarks.datasets import get_torchvision_loaders, get_sequence_loaders
from eden.config import set_seed
from eden.core.epigenome import HeritableEpigenome
from eden.core.genome import GeneRegulator
from eden.core.network import EDENNetwork, SequenceEDENNetwork

DATA_ROOT = Path("/kaggle/working/.data")
set_seed(0)


def make_inference_fn(model: nn.Module):
    """Return a callable(x) that handles the EDEN forward signature."""
    reg_module  = GeneRegulator().to(next(model.parameters()).device)
    epi_module  = HeritableEpigenome().to(next(model.parameters()).device)

    def _call(x: torch.Tensor):
        reg_module_d  = reg_module.to(x.device)
        epi_module_d  = epi_module.to(x.device)
        regulator     = reg_module_d()
        _, epi_mask   = epi_module_d()
        return model.to(x.device)(x, regulator, epi_mask)

    return _call


def measure(model: nn.Module, sample: torch.Tensor, label: str,
            n_warmup: int = 50, n_runs: int = 200) -> None:
    model.eval()
    bs    = sample.shape[0]
    infer = make_inference_fn(model)

    # ── GPU ───────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        m_gpu = model.cuda()
        infer_gpu = make_inference_fn(m_gpu)
        x_gpu = sample.cuda()
        with torch.no_grad():
            for _ in range(n_warmup):
                infer_gpu(x_gpu)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                infer_gpu(x_gpu)
        torch.cuda.synchronize()
        gpu_ms = (time.perf_counter() - t0) * 1000 / n_runs
        print(
            f"  {label:<28} GPU  {gpu_ms:7.2f} ms/batch"
            f"  {gpu_ms / bs * 1000:8.1f} µs/sample"
            f"  {bs / gpu_ms * 1000:9.0f} samples/s"
        )

    # ── CPU ───────────────────────────────────────────────────────────────
    m_cpu   = model.cpu()
    infer_cpu = make_inference_fn(m_cpu)
    x_cpu   = sample.cpu()
    with torch.no_grad():
        for _ in range(10):
            infer_cpu(x_cpu)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(50):
            infer_cpu(x_cpu)
    cpu_ms = (time.perf_counter() - t0) * 1000 / 50
    print(
        f"  {label:<28} CPU  {cpu_ms:7.2f} ms/batch"
        f"  {cpu_ms / bs * 1000:8.1f} µs/sample"
        f"  {bs / cpu_ms * 1000:9.0f} samples/s"
    )


# ── Load data ─────────────────────────────────────────────────────────────────
_, va_mnist,  meta_mnist  = get_torchvision_loaders("mnist",   DATA_ROOT, batch_size=64)
_, va_cifar,  meta_cifar  = get_torchvision_loaders("cifar10", DATA_ROOT, batch_size=64)
_, va_ecg,    meta_ecg    = get_sequence_loaders("ecg", batch_size=64, seed=42)

batch_mnist = next(iter(va_mnist))[0]
batch_cifar = next(iter(va_cifar))[0]
batch_ecg   = next(iter(va_ecg))[0]

model_mnist = EDENNetwork(
    int(meta_mnist["num_classes"]), int(meta_mnist["in_channels"]), tuple(meta_mnist["image_hw"])
)
model_cifar = EDENNetwork(
    int(meta_cifar["num_classes"]), int(meta_cifar["in_channels"]), tuple(meta_cifar["image_hw"])
)
model_ecg = SequenceEDENNetwork(int(meta_ecg["num_classes"]), int(meta_ecg["seq_len"]))

params_mnist = sum(p.numel() for p in model_mnist.parameters())
params_cifar = sum(p.numel() for p in model_cifar.parameters())
params_ecg   = sum(p.numel() for p in model_ecg.parameters())

# ── Benchmark ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("LATENCE INFÉRENCE — EDEN")
print(f"{'='*70}")
print(f"\n  {'label':<28} {'device':<5}  {'ms/batch':>10}  {'µs/sample':>12}  {'samples/s':>12}")
print("  " + "-"*68)

for bs in (1, 32):
    measure(model_mnist, batch_mnist[:bs], f"MNIST({params_mnist/1e6:.1f}M)  batch={bs}")

print()
for bs in (1, 32):
    measure(model_cifar, batch_cifar[:bs], f"CIFAR-10({params_cifar/1e6:.1f}M) batch={bs}")

print()
for bs in (1, 32):
    measure(model_ecg, batch_ecg[:bs],   f"ECG({params_ecg/1e6:.2f}M)    batch={bs}")

print("\n===== LATENCE TERMINÉE =====\n")
