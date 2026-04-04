from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from eden.benchmarks import datasets
from eden.config import AblationFlags
from eden.core.epigenome import HeritableEpigenome
from eden.core.genome import GeneRegulator
from eden.core.network import SequenceEDENNetwork
from eden.training import train_eden


def test_full_training_checkpoint_roundtrip(tmp_path: Path) -> None:
    full = datasets.spiral_dataset(400, noise=0.02, seed=0)
    n = len(full)
    tr, va = torch.utils.data.random_split(
        full,
        [int(0.85 * n), n - int(0.85 * n)],
        generator=torch.Generator().manual_seed(0),
    )
    tr_l = DataLoader(tr, batch_size=64, shuffle=True)
    va_l = DataLoader(va, batch_size=64, shuffle=False)
    ck = tmp_path / "t.pt"

    m1 = SequenceEDENNetwork(2, 2, flags=AblationFlags())
    train_eden(m1, GeneRegulator(), HeritableEpigenome(), tr_l, va_l, epochs=1, checkpoint_out=ck)
    assert ck.is_file()

    m2 = SequenceEDENNetwork(2, 2, flags=AblationFlags())
    rep = train_eden(
        m2,
        GeneRegulator(),
        HeritableEpigenome(),
        tr_l,
        va_l,
        epochs=1,
        resume_from=ck,
    )
    assert rep["epochs_ran"] == 2
