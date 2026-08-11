"""`overlap compare` - check an incoming manifest against the local corpus."""

from __future__ import annotations

import json
import webbrowser
from pathlib import Path

import typer

from overlap.cli._console import emit, get_state
from overlap.exit_codes import ExitCode
from overlap.match import compare_manifest_file, compare_two_manifests
from overlap.render import render_html


def compare_cmd(
    ctx: typer.Context,
    manifest: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Incoming .ovlm manifest, or a directory of parts.",
    ),
    output: Path = typer.Option(
        Path("report.json"), "-o", "--output", help="Where to write the report JSON."
    ),
    html: Path | None = typer.Option(
        None, "--html", help="Also render a standalone HTML report to this path."
    ),
    open_html: bool = typer.Option(False, "--open", help="Open the HTML report when done."),
    min_run: float = typer.Option(
        None, "--min-run", help="Minimum matched-run length in seconds (default 10)."
    ),
    tier: str = typer.Option(
        "probable",
        "--tier",
        help="Lowest tier to include: strong | probable | weak.",
    ),
    nprobe: int = typer.Option(
        None, "--nprobe", help="ANN recall/speed knob (default from config: 64)."
    ),
    threads: int = typer.Option(
        None,
        "--threads",
        help="Cap search threads (default from config: 0 = every core).",
    ),
    against: Path | None = typer.Option(
        None,
        "--against",
        help="Compare against another manifest instead of the local index - for two "
        "offers of footage you own neither of.",
    ),
    fail_over: float | None = typer.Option(
        None,
        "--fail-over",
        min=0.0,
        max=100.0,
        help="Exit with code 3 when overlap %% is at or above this (CI gating).",
    ),
) -> None:
    """Compare a vendor manifest against everything in the local index.

    The result answers: how much of this offering do we already own? The report
    lists matched files and segments, detected speed changes (slowed footage
    inflating billable hours), splices, and mirrored copies.

    With --against, compares two manifests instead: for when two aggregators
    offer the same footage and you own neither yet. Coverage is narrower there -
    neither side has pixels, so cropped copies cannot be found - and the report
    states it.
    """
    state = get_state(ctx)
    cfg = state.config
    if tier not in ("strong", "probable", "weak"):
        raise typer.BadParameter("--tier must be strong, probable, or weak")

    common = dict(
        min_run_s=min_run if min_run is not None else float(cfg.get("compare.min_run_s")),
        include_weak=(tier == "weak"),
        nprobe=nprobe if nprobe is not None else int(cfg.get("compare.nprobe")),
        threads=threads if threads is not None else int(cfg.get("compare.threads")),
        probe_stride=int(cfg.get("compare.probe_stride")),
        progress=(lambda e: emit(state, e)) if state.json_mode else None,
    )
    if against is not None:
        if not against.is_file():
            raise typer.BadParameter(f"--against: not found: {against}")
        report = compare_two_manifests(manifest, against, **common)  # type: ignore[arg-type]
    else:
        report = compare_manifest_file(manifest, cfg.index_dir, **common)  # type: ignore[arg-type]

    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if html is not None:
        html.write_text(render_html(report), encoding="utf-8")

    summary = report["summary"]
    if state.json_mode:
        emit(state, {"event": "result", "report_path": str(output), "summary": summary})
    else:
        pct = summary["overlap_pct"]
        state.err.print(
            f"[bold]{pct}%[/bold] of offered footage matches this corpus "
            f"({summary['matched_hours']} of {summary['offered_hours']} h, "
            f"{summary['files_with_overlap']} of {summary['files_offered']} files)."
        )
        flags = summary["flags"]
        if flags.get("weak_only_files"):
            state.err.print(
                f"[yellow]{flags['weak_only_files']} file(s) carry weak evidence that is "
                f"NOT counted above - re-run with --tier weak to see it.[/yellow]"
            )
        if flags["slowdown_files"]:
            state.err.print(
                f"[red]⚠ {flags['slowdown_files']} file(s) appear slowed down - "
                f"billable-hours inflation.[/red]"
            )
        state.err.print(f"Report: {output}" + (f" · HTML: {html}" if html else ""))

    if html is not None and open_html:
        webbrowser.open(html.resolve().as_uri())

    if fail_over is not None and summary["overlap_pct"] >= fail_over:
        raise typer.Exit(code=ExitCode.OVERLAP_FOUND)
