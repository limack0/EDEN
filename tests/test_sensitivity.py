from __future__ import annotations

from eden.cli.main import REG_ATTRS


def test_reg_attrs_covers_l2() -> None:
    assert "tau" in REG_ATTRS
    assert "lambda" in REG_ATTRS
    assert len(REG_ATTRS) == 8
