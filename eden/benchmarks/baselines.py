"""Baseline models: MLP, LeNet-5, small ResNet-8, optional NEAT."""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBaseline(nn.Module):
    def __init__(self, flat_dim: int, num_classes: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LeNet5(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(in_channels, 6, 5, padding=2 if in_channels == 1 else 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(16 * 5 * 5, 120), nn.ReLU(inplace=True), nn.Linear(120, 84), nn.ReLU(inplace=True), nn.Linear(84, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.feat(x)
        z = torch.flatten(z, 1)
        return self.fc(z)


class ResidualBlock(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.c2(F.relu(self.c1(x))) + x)


class ResNet8(nn.Module):
    """Tiny ResNet-style stack (~8 layers) for CIFAR-sized inputs."""

    def __init__(self, in_channels: int, num_classes: int, width: int = 32) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.b1 = ResidualBlock(width)
        self.b2 = ResidualBlock(width)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)
        z = self.b1(z)
        z = self.b2(z)
        z = self.pool(z).flatten(1)
        return self.fc(z)


class CNN1DBaseline(nn.Module):
    def __init__(self, in_len: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, num_classes)
        self.in_len = in_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        z = self.net(x).squeeze(-1)
        return self.fc(z)


class LSTMOnFlattenedSeq(nn.Module):
    """Interprets flat (B, L*feat) as (B, L, feat) for LSTM baseline."""

    def __init__(self, seq_len: int, feat: int, num_classes: int, hidden: int = 128) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.feat = feat
        self.lstm = nn.LSTM(feat, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        x = x.view(b, self.seq_len, self.feat)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


class LSTMBaseline(nn.Module):
    def __init__(self, input_size: int, num_classes: int, hidden: int = 128) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


def run_neat_baseline(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    generations: int = 5,
) -> float | None:
    try:
        import neat  # noqa: F401
    except ImportError:
        warnings.warn("neat-python not installed; skipping NEAT baseline", stacklevel=2)
        return None
    warnings.warn("NEAT full benchmark is heavy; returning None placeholder", stacklevel=2)
    return None
