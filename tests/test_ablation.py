from __future__ import annotations

import pytest

from eden.config import AblationFlags
from eden.cli.main import _flags_from_mechanism


def test_ablation_flags_disable() -> None:
    f = _flags_from_mechanism("microglia")
    assert f.microglia is False
    f2 = _flags_from_mechanism("paracrine")
    assert f2.paracrine is False


def test_unknown_mechanism() -> None:
    with pytest.raises(Exception):
        _flags_from_mechanism("not_a_real_mechanism")
