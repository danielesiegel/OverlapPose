"""`overlap audit-sample` - check a manifest against the footage you were shown."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from overlap.cli._console import emit, get_state
from overlap.match.audit import audit_sample


def audit_sample_cmd(
    ctx: typer.Context,
    manifest: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="The seller's .ovlm manifest, or a directory of parts.",
    ),
    sample: Path = typer.Option(
        ...,
        "--sample",
        exists=True,
        file_okay=False,
        help="Directory of the sample footage the seller shared.",
    ),
    output: Path | None = typer.Option(
        None, "-o", "--output", help="Write the full audit JSON here."
    ),
    fps: float | None = typer.Option(
        None,
        "--fps",
        min=0.5,
        max=8.0,
        help="Sampling density for the sample. Defaults to the manifest's own density, "
        "which aligns both sampling grids on the same instants.",
    ),
    workers: int | None = typer.Option(None, "--workers", help="Parallel workers (0 = auto)."),
) -> None:
    """Check that a manifest describes the same footage as the shared sample.

    A seller shows a lab one or two hours out of thousands. This fingerprints
    that sample locally and looks for it in the manifest: if the sample was drawn
    from the offered data, nearly all of it must appear. A low result means the
    manifest and the sample describe different footage.

    It cannot prove the rest of the manifest corresponds to footage the seller
    holds - only `overlap verify` against delivered bytes can do that. Run this
    before paying, and that after.
    """
    state = get_state(ctx)
    cfg = state.config
    result = audit_sample(
        manifest,
        sample,
        sample_fps=fps,
        min_run_s=float(cfg.get("compare.min_run_s")),
        nprobe=int(cfg.get("compare.nprobe")),
        threads=int(cfg.get("compare.threads")),
        workers=workers if workers is not None else int(cfg.get("index.workers")),
        progress=(lambda e: emit(state, e)) if state.json_mode else None,
    )
    if output is not None:
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if state.json_mode:
        summary = {k: v for k, v in result.items() if k != "detail"}
        emit(state, {"event": "result", **summary})
        return
    if state.quiet:
        return

    verdict = (
        "[green]consistent[/green]"
        if result["consistent"]
        else "[red]INCONSISTENT[/red]"
    )
    state.err.print(
        f"Sample: {result['sample_files']} file(s), {result['sample_hours']:,.2f} h "
        f"at {result['sample_fps']:g} fps "
        f"({result['sample_share_of_offer']:g}% of the {result['manifest_hours']:,.1f} h offered)"
    )
    state.err.print(
        f"Found in the manifest: {result['sample_found_pct']:.1f}% -> {verdict}"
    )
    if result["unmatched_sample_files"]:
        state.err.print(
            "[yellow]Sample footage absent from the manifest:[/yellow] "
            + ", ".join(result["unmatched_sample_files"][:10])
            + (
                f" (+{len(result['unmatched_sample_files']) - 10} more)"
                if len(result["unmatched_sample_files"]) > 10
                else ""
            )
        )
    if not result["consistent"]:
        state.err.print(
            "[yellow]The manifest does not appear to describe the footage you were "
            "shown. Ask the seller to re-export it from the files they intend to "
            "deliver.[/yellow]"
        )
    state.err.print(f"[dim]{result['note']}[/dim]")
