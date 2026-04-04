#!/usr/bin/env python3
"""Compatibility entry point matching prompt naming (`eden benchmark` equivalent)."""

from __future__ import annotations

import sys

from eden.cli.main import main

if __name__ == "__main__":
    sys.argv = ["eden", "benchmark", *sys.argv[1:]]
    main()
