"""Optional Textual dashboard; falls back if textual is missing."""

from __future__ import annotations

import os
from pathlib import Path


def run_tui(dataset: str) -> None:
    try:
        from textual.app import App, ComposeResult
        from textual.containers import Vertical
        from textual.widgets import Footer, Header, Log, Static
    except ImportError:
        print("Textual not installed; logging mode. pip install textual")
        print(f"Would monitor dataset={dataset}")
        return

    class EdenTui(App[None]):
        BINDINGS = [("q", "quit", "Quit")]
        tick_count = 0

        def __init__(self, ds: str) -> None:
            super().__init__()
            self.ds = ds

        def compose(self) -> ComposeResult:
            yield Header()
            with Vertical():
                yield Static(f"EDEN live — dataset: {self.ds}", id="title")
                yield Log(id="log", auto_scroll=True)
            yield Footer()

        def on_mount(self) -> None:
            out = Path(os.environ.get("EDEN_RESULTS", "eden_results"))
            log = self.query_one(Log)
            log.write_line(f"Watching {out}")
            self.set_interval(0.4, self._pulse)

        def _pulse(self) -> None:
            log = self.query_one(Log)
            self.tick_count += 1
            log.write_line(f"t={self.tick_count} loss=— acc=— expr=— nodes=8 events=…")
            if self.tick_count > 40:
                self.exit()

    EdenTui(dataset).run()
