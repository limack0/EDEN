from __future__ import annotations

from pathlib import Path

import torch

from eden.config import AblationFlags
from eden.core.genome import GeneRegulator
from eden.core.network import EDENNetwork
from eden.training import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    m = EDENNetwork(10, 1, (28, 28), flags=AblationFlags())
    r = GeneRegulator()
    p = tmp_path / "ck.pt"
    save_checkpoint(p, m, r, {"meta": 1})
    m2 = EDENNetwork(10, 1, (28, 28), flags=AblationFlags())
    r2 = GeneRegulator()
    extra = load_checkpoint(p, m2, r2)
    assert extra.get("meta") == 1
    w0 = next(m.parameters()).data.clone()
    w1 = next(m2.parameters()).data
    assert torch.allclose(w0, w1)
