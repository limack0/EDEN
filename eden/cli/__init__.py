"""CLI subpackage. Lazy `main` so `python -m eden.cli.main` does not pre-import the module."""

from __future__ import annotations

from typing import Any

__all__ = ["main"]


def __getattr__(name: str) -> Any:
    if name == "main":
        from eden.cli.main import main as main_fn

        return main_fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
