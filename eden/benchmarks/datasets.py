"""Dataset loaders: vision, synthetic spiral, ECG-like, protein-like."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from eden.config import set_seed


def spiral_dataset(n: int = 500, noise: float = 0.03, seed: int | None = None) -> TensorDataset:
    set_seed(seed)
    theta = torch.linspace(0, 4 * math.pi, n)
    r = theta / (4 * math.pi)
    x1 = r * torch.cos(theta) + noise * torch.randn(n)
    y1 = r * torch.sin(theta) + noise * torch.randn(n)
    x2 = -r * torch.cos(theta) + noise * torch.randn(n)
    y2 = -r * torch.sin(theta) + noise * torch.randn(n)
    x = torch.stack([torch.cat([x1, x2]), torch.cat([y1, y2])], dim=1).float()
    y = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
    perm = torch.randperm(x.size(0))
    return TensorDataset(x[perm], y[perm])


class SyntheticECGDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """188-dim time series, binary arrhythmia label (synthetic, PhysioNet-like shape)."""

    def __init__(self, n: int = 2000, dim: int = 188, seed: int | None = None) -> None:
        set_seed(seed)
        self.dim = dim
        g = torch.Generator().manual_seed(seed or 0)
        self.x = torch.randn(n, dim, generator=g)
        t = torch.linspace(0, 1, dim).unsqueeze(0).expand(n, -1)
        self.x[:, : dim // 2] += 0.3 * torch.sin(8 * math.pi * t[:, : dim // 2])
        pos = (self.x[:, 10:30].sum(dim=1) > 0.5).long()
        self.y = pos

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[i], self.y[i]


class SyntheticProteinDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Random sequences (20 AA one-hot) -> Q8-like 8-class labels (CB513-shaped smoke test)."""

    def __init__(self, n: int = 1500, length: int = 128, seed: int | None = None) -> None:
        set_seed(seed)
        g = torch.Generator().manual_seed(seed or 0)
        self.seq = torch.randint(0, 20, (n, length), generator=g)
        self.y = torch.randint(0, 8, (n,), generator=g)

    def __len__(self) -> int:
        return self.seq.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.seq[i]
        oh = F.one_hot(s, num_classes=20).float().reshape(-1)
        return oh, self.y[i]


def get_torchvision_loaders(
    name: str,
    data_root: Path,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[DataLoader[Any], DataLoader[Any], dict[str, Any]]:
    try:
        from torchvision import datasets, transforms
    except ImportError as e:
        raise RuntimeError("torchvision required for vision datasets") from e

    data_root.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {"name": name, "in_channels": 1, "image_hw": (28, 28)}

    if name == "mnist":
        tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train = datasets.MNIST(data_root, train=True, download=True, transform=tfm)
        test = datasets.MNIST(data_root, train=False, download=True, transform=tfm)
        meta["num_classes"] = 10
    elif name == "fashion_mnist":
        tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
        train = datasets.FashionMNIST(data_root, train=True, download=True, transform=tfm)
        test = datasets.FashionMNIST(data_root, train=False, download=True, transform=tfm)
        meta["num_classes"] = 10
    elif name == "cifar10":
        tfm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        train = datasets.CIFAR10(data_root, train=True, download=True, transform=tfm)
        test = datasets.CIFAR10(data_root, train=False, download=True, transform=tfm)
        meta.update({"num_classes": 10, "in_channels": 3, "image_hw": (32, 32)})
    elif name == "cifar100":
        tfm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )
        train = datasets.CIFAR100(data_root, train=True, download=True, transform=tfm)
        test = datasets.CIFAR100(data_root, train=False, download=True, transform=tfm)
        meta.update({"num_classes": 100, "in_channels": 3, "image_hw": (32, 32)})
    else:
        raise ValueError(f"Unknown vision dataset: {name}")

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader, meta


def get_sequence_loaders(
    name: str,
    batch_size: int,
    seed: int | None = None,
) -> tuple[DataLoader[Any], DataLoader[Any], dict[str, Any]]:
    if name == "ecg":
        full = SyntheticECGDataset(2500, 188, seed)
        n = len(full)
        train, val = torch.utils.data.random_split(full, [int(0.85 * n), n - int(0.85 * n)], generator=torch.Generator().manual_seed(seed or 0))
        return (
            DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(val, batch_size=batch_size, shuffle=False),
            {"num_classes": 2, "seq_len": 188, "kind": "ecg"},
        )
    if name == "protein":
        full = SyntheticProteinDataset(2000, 128, seed)
        n = len(full)
        train, val = torch.utils.data.random_split(full, [int(0.85 * n), n - int(0.85 * n)], generator=torch.Generator().manual_seed(seed or 0))
        return (
            DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(val, batch_size=batch_size, shuffle=False),
            {"num_classes": 8, "seq_len": 128 * 20, "kind": "protein"},
        )
    raise ValueError(name)
