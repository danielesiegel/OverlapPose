"""`overlap config` - show every effective value and where it came from."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from overlap.cli._console import emit_document, get_state
from overlap.paths import default_index_dir, reports_dir, user_config_file


def config_cmd(ctx: typer.Context) -> None:
    """Show the effective configuration, each value's source, and resolved paths."""
    state = get_state(ctx)
    cfg = state.config

    resolved_paths = {
        "index_dir": str(cfg.index_dir),
        "default_index_dir": str(default_index_dir()),
        "user_config_file": str(user_config_file()),
        "reports_dir": str(reports_dir()),
    }

    if state.json_mode:
        emit_document(
            {
                "values": {k: cfg.values[k] for k in sorted(cfg.values)},
                "sources": {k: cfg.sources[k] for k in sorted(cfg.sources)},
                "paths": resolved_paths,
            }
        )
        return

    out = Console()  # data -> stdout
    table = Table(title="Effective configuration")
    table.add_column("key", style="bold")
    table.add_column("value")
    table.add_column("source", style="dim")
    for key in sorted(cfg.values):
        table.add_row(key, repr(cfg.values[key]), cfg.sources[key])
    out.print(table)

    paths_table = Table(title="Resolved paths")
    paths_table.add_column("purpose", style="bold")
    paths_table.add_column("path")
    for name, value in resolved_paths.items():
        paths_table.add_row(name, value)
    out.print(paths_table)
