from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from eden.benchmarks import datasets
from eden.config import AblationFlags, set_seed
from eden.core.epigenome import HeritableEpigenome
from eden.core.genome import GeneRegulator
from eden.core.network import SequenceEDENNetwork
from eden.training import train_eden


def test_spiral_train_reaches_reasonable_accuracy() -> None:
    # Fix RNG before model init and pass the same seed into training so order of
    # other tests cannot desynchronize weight init from the train loop's set_seed.
    set_seed(0)
    full = datasets.spiral_dataset(2000, noise=0.02, seed=0)
    n = len(full)
    tr, va = torch.utils.data.random_split(
        full,
        [int(0.85 * n), n - int(0.85 * n)],
        generator=torch.Generator().manual_seed(0),
    )
    tr_l = DataLoader(tr, batch_size=32, shuffle=True)
    va_l = DataLoader(va, batch_size=32, shuffle=False)
    model = SequenceEDENNetwork(2, 2, flags=AblationFlags(), max_stems=12)
    reg = GeneRegulator()
    epi = HeritableEpigenome()
    # EDEN has many non-linear biological paths; a plain MLP reaches ~0.98 on this spiral,
    # while EDEN typically lands ~0.65–0.85 in a few dozen epochs (seed-dependent).
    r = train_eden(model, reg, epi, tr_l, va_l, epochs=40, lr=3e-3, seed=0)
    assert r["final_val_accuracy"] >= 0.62
