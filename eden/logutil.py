"""Console (and optional file) logging for CLI and training."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_cli_logging(
    *,
    quiet: bool = False,
    verbose: bool = False,
    log_file: str | Path | None = None,
) -> None:
    level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"
    try:
        logging.basicConfig(level=level, format=fmt, datefmt=datefmt, stream=sys.stdout, force=True)
    except TypeError:
        logging.basicConfig(level=level, format=fmt, datefmt=datefmt, stream=sys.stdout)
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        logging.getLogger().addHandler(fh)
