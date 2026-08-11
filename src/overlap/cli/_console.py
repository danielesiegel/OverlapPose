"""Console conventions shared by all CLI commands.

Contract: **stdout carries data, stderr carries everything else** (progress,
logs, warnings). Piping `overlap ... --json` must never capture a progress bar.
In ``--json`` mode, long-running commands emit NDJSON events on stdout and all
Rich rendering is disabled.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any

import typer
from rich.console import Console

from overlap.config import Config


@dataclass
class AppState:
    config: Config
    json_mode: bool = False
    quiet: bool = False
    verbosity: int = 0
    # Rich writes to stderr so stdout stays machine-clean.
    err: Console = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.err is None:
            self.err = Console(stderr=True, quiet=self.quiet)


def get_state(ctx: typer.Context) -> AppState:
    state = ctx.find_object(AppState)
    if state is None:  # pragma: no cover - typer guarantees the callback ran
        raise RuntimeError("CLI state not initialised")
    return state


def emit(state: AppState, event: dict[str, Any]) -> None:
    """Emit one NDJSON event on stdout (only in --json mode)."""
    if state.json_mode:
        sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def emit_document(obj: Any) -> None:
    """Emit a single JSON document on stdout (short commands in --json mode)."""
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
