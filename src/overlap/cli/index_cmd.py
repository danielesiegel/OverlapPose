"""`overlap index` - fingerprint media files into the local index."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from overlap.cli._console import emit, get_state
from overlap.coverage import PRESETS, describe, summary_lines
from overlap.exit_codes import ExitCode
from overlap.hashing import build_crop_variants, crop_variants_spec
from overlap.ingest import index_paths


def index_cmd(
    ctx: typer.Context,
    paths: list[Path] = typer.Argument(..., help="Files or directories to fingerprint."),
    fps: float | None = typer.Option(
        None,
        "--fps",
        min=0.5,
        max=8.0,
        help=(
            "Corpus sampling density in frames/second (default from config: 4.0). "
            "Lower it for very large corpora; speed-change and arbitrary-trim "
            "detection degrade below ~2."
        ),
    ),
    preset: str | None = typer.Option(
        None,
        "--preset",
        help=(
            "Detection coverage vs cost: fast | balanced | thorough. "
            "Sets the crop geometries; see `overlap inspect` for what each catches."
        ),
    ),
    workers: int | None = typer.Option(
        None, "--workers", help="Parallel worker processes (0 = auto)."
    ),
    include: list[str] = typer.Option([], "--include", help="Only paths matching GLOB."),
    exclude: list[str] = typer.Option([], "--exclude", help="Skip paths matching GLOB."),
    follow_symlinks: bool = typer.Option(False, "--follow-symlinks"),
    crop_ladder: str | None = typer.Option(
        None,
        "--crop-ladder",
        help=(
            "Comma-separated centered-crop keep fractions indexed alongside each "
            "frame so zoom-cropped copies are detectable (default 0.94,0.88,0.82,"
            "0.76,0.70). Pass '' to disable and shrink the index ~6x."
        ),
    ),
    crop_edges: str | None = typer.Option(
        None,
        "--crop-edges",
        help=(
            "Also index one-sided edge crops, e.g. 'bottom,top' (each side gets a "
            "6%% ladder to 30%%) or 'bottom:0.08,0.16'. Catches strip crops that "
            "remove overlays, which centered rungs cannot. Off by default: every "
            "side adds 10 codes per frame to the index."
        ),
    ),
    reindex: bool = typer.Option(
        False, "--reindex", help="Re-fingerprint files even if unchanged."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="List what would be indexed, then exit."),
) -> None:
    """Fingerprint videos and robotics containers into the local index.

    Indexing is resumable: re-running the same command skips files already
    indexed (matched by path, size, and mtime). Interrupting with Ctrl+C is
    safe - completed files are committed.
    """
    state = get_state(ctx)
    cfg = state.config
    sample_fps = fps if fps is not None else float(cfg.get("index.fps"))
    shard_codes = int(cfg.get("index.shard_codes"))
    n_workers = workers if workers is not None else int(cfg.get("index.workers"))
    if preset is not None and preset not in PRESETS:
        raise typer.BadParameter(f"--preset must be one of: {', '.join(PRESETS)}")
    chosen = PRESETS[preset] if preset is not None else None
    # An explicit ladder still wins over a preset, so the two can be combined.
    ladder_spec = crop_ladder if crop_ladder is not None else (
        chosen.crop_ladder if chosen else str(cfg.get("index.crop_ladder"))
    )
    edges_spec = crop_edges if crop_edges is not None else (
        chosen.crop_edges if chosen else str(cfg.get("index.crop_edges"))
    )
    try:
        variants = build_crop_variants(ladder_spec, edges_spec)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if sample_fps < 4.0 and not state.quiet:
        # Measured: a copy cut at an arbitrary time samples instants the corpus
        # never sampled, and below 4 fps that phase gap costs enough hash bits
        # to lose trims, concatenations and speed changes.
        state.err.print(
            f"[yellow]Note: indexing at {sample_fps:g} fps. Detection of arbitrary "
            f"trims, concatenations and speed changes degrades below 4 fps "
            f"(see docs/architecture.md).[/yellow]"
        )
    if not state.quiet and not state.json_mode:
        for line in summary_lines(ladder_spec, edges_spec, sample_fps):
            state.err.print(f"[dim]{line}[/dim]")
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise typer.BadParameter(f"paths do not exist: {', '.join(map(str, missing))}")

    if dry_run:
        from overlap.ingest import discover_files

        for abspath, _root in discover_files(
            paths, list(include) or None, list(exclude) or None, follow_symlinks
        ):
            typer.echo(str(abspath))
        return

    if state.json_mode:
        emit(state, {"event": "coverage", **describe(ladder_spec, edges_spec, sample_fps)})
        stats = index_paths(
            paths,
            cfg.index_dir,
            sample_fps=sample_fps,
            crop_ladder=ladder_spec,
            crop_edges=edges_spec,
            workers=n_workers,
            include=list(include) or None,
            exclude=list(exclude) or None,
            follow_symlinks=follow_symlinks,
            reindex=reindex,
            shard_codes=shard_codes,
            progress=lambda e: emit(state, e),
        )
    else:
        progress_bar = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=state.err,
            disable=state.quiet,
        )
        task_id = None

        def render(event: dict[str, Any]) -> None:
            nonlocal task_id
            if event["event"] == "start":
                task_id = progress_bar.add_task("indexing", total=event["total_files"])
                if event.get("already_indexed"):
                    state.err.print(
                        f"Resuming: {event['already_indexed']} file(s) already indexed."
                    )
            elif event["event"] == "setting_change":
                state.err.print(
                    f"[yellow]{event['key']}: this index holds streams built with "
                    f"{event['previous']}; new files use {event['now']}. Existing "
                    f"streams keep their own setting (re-index them to change it).[/yellow]"
                )
            elif event["event"] == "file" and task_id is not None:
                if event.get("reason") != "unchanged":
                    progress_bar.advance(task_id)
                if event["status"] == "error":
                    state.err.print(f"[red]error[/red] {event['path']}: {event.get('error')}")

        with progress_bar:
            stats = index_paths(
                paths,
                cfg.index_dir,
                sample_fps=sample_fps,
                crop_ladder=ladder_spec,
                crop_edges=edges_spec,
                workers=n_workers,
                include=list(include) or None,
                exclude=list(exclude) or None,
                follow_symlinks=follow_symlinks,
                reindex=reindex,
                shard_codes=shard_codes,
                progress=render,
            )
        state.err.print(
            f"Indexed {stats.indexed} file(s) ({stats.streams} streams, "
            f"{stats.frames} frames), skipped {stats.skipped}, errors {stats.errors}."
        )
        if variants:
            state.err.print(f"Crop variants indexed per frame: {crop_variants_spec(variants)}")

    if stats.exit_partial:
        raise typer.Exit(code=ExitCode.PARTIAL_FAILURE)
