"""`overlap self-dedupe` - find duplicated footage inside the local corpus."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from overlap.cli._console import emit, get_state
from overlap.match import self_dedupe


def self_dedupe_cmd(
    ctx: typer.Context,
    output: Path = typer.Option(
        Path("self-dedupe.json"), "-o", "--output", help="Where to write the report JSON."
    ),
    min_run: float | None = typer.Option(None, "--min-run"),
) -> None:
    """Compare the corpus against itself (excluding each stream's trivial
    self-match). Vendors use this to dedupe inventory before quoting it."""
    state = get_state(ctx)
    cfg = state.config
    report = self_dedupe(
        cfg.index_dir,
        min_run_s=min_run if min_run is not None else float(cfg.get("compare.min_run_s")),
        threads=int(cfg.get("compare.threads")),
        probe_stride=int(cfg.get("compare.probe_stride")),
        progress=(lambda e: emit(state, e)) if state.json_mode else None,
    )
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = report["summary"]
    if state.json_mode:
        emit(state, {"event": "result", "report_path": str(output), "summary": summary})
    else:
        state.err.print(
            f"Internal duplication: [bold]{summary['overlap_pct']}%[/bold] of corpus hours "
            f"re-match other corpus footage ({summary['files_with_overlap']} files). "
            f"Report: {output}"
        )
