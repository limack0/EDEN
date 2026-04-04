"""Optional YAML training overrides (``pip install pyyaml``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_train_yaml(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("Install PyYAML to use --config: pip install pyyaml") from e
    data = yaml.safe_load(raw)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping")
    return data


def apply_train_overrides(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Shallow merge: overrides keys replace base."""
    out = dict(base)
    for k, v in overrides.items():
        if v is not None:
            out[k] = v
    return out
