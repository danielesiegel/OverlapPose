"""`overlap doctor` - environment diagnostics.

Checks that the pieces overlap depends on are importable and reports versions.
The output of `overlap doctor --json` is what bug reports ask for: versions and
environment facts only, never data.
"""

from __future__ import annotations

import platform
import shutil
import sys
from importlib import import_module
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

import overlap
from overlap.cli._console import emit_document, get_state

# (import name, human label, required)
_COMPONENTS: list[tuple[str, str, bool]] = [
    ("numpy", "numpy (array math)", True),
    ("av", "PyAV (video decode, bundles FFmpeg libs)", True),
    ("cv2", "OpenCV (image ops)", True),
    ("faiss", "FAISS (similarity search)", True),
    ("zstandard", "zstandard (manifest compression)", True),
    ("msgpack", "msgpack (manifest encoding)", True),
    ("mcap", "mcap (MCAP container)", True),
    ("fastapi", "FastAPI (web UI)", True),
    ("rosbags", "rosbags (ROS1 .bag / rosbag2) - install extra: overlap-cli[ros]", False),
]


def _probe(module: str) -> tuple[bool, str | None]:
    try:
        mod = import_module(module)
    except Exception as exc:  # noqa: BLE001 - native imports fail in odd ways
        return False, f"{type(exc).__name__}: {exc}"
    return True, str(getattr(mod, "__version__", None) or "unknown")


def doctor(ctx: typer.Context) -> None:
    """Check the environment: dependency imports, versions, index location."""
    state = get_state(ctx)
    checks: list[dict[str, Any]] = []

    for module, label, required in _COMPONENTS:
        ok, detail = _probe(module)
        checks.append({"component": label, "ok": ok, "required": required, "detail": detail})

    ffmpeg_cli = shutil.which("ffmpeg")
    checks.append(
        {
            "component": "ffmpeg CLI (optional: only used to generate test/demo media)",
            "ok": ffmpeg_cli is not None,
            "required": False,
            "detail": ffmpeg_cli,
        }
    )

    index_dir = state.config.index_dir
    index_state = "exists" if index_dir.exists() else "will be created on first index"
    checks.append(
        {
            "component": "index directory",
            "ok": True,
            "required": True,
            "detail": f"{index_dir} ({index_state})",
        }
    )

    doc = {
        "overlap": overlap.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "checks": checks,
    }

    required_failures = [c for c in checks if c["required"] and not c["ok"]]

    if state.json_mode:
        emit_document(doc)
    else:
        out = Console()
        line = f"overlap {overlap.__version__} · Python {platform.python_version()}"
        out.print(f"{line} · {platform.platform()}")
        table = Table(show_header=True)
        table.add_column("component")
        table.add_column("status")
        table.add_column("detail", style="dim", overflow="fold")
        for c in checks:
            status = (
                "[green]ok[/green]"
                if c["ok"]
                else ("[red]MISSING[/red]" if c["required"] else "[yellow]absent[/yellow]")
            )
            table.add_row(c["component"], status, str(c["detail"]))
        out.print(table)
        if required_failures:
            state.err.print("[red]Required components are missing - see above.[/red]")

    if required_failures:
        raise typer.Exit(code=1)
