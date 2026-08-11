"""`overlap merge` - combine indexes built on separate machines."""

from __future__ import annotations

from pathlib import Path

import typer

from overlap.cli._console import emit, get_state
from overlap.errors import IndexError_
from overlap.store.annindex import AnnIndex, describe_index
from overlap.store.catalog import Catalog


def merge_cmd(
    ctx: typer.Context,
    sources: list[Path] = typer.Argument(..., help="Index directories to merge in."),
    shard: bool = typer.Option(
        True,
        "--shard/--no-shard",
        help="Build search shards for the merged footage now, or leave it for the next compare.",
    ),
) -> None:
    """Merge other index directories into this one.

    Fingerprinting is the expensive step, so a corpus too large for one machine
    is built by fingerprinting slices in parallel and merging the results: this
    moves hashes, it does not recompute them. Files already present by content
    are skipped, so a merge can be re-run safely after an interruption.

    The merged footage is sharded incrementally - adding a machine's worth of
    footage costs the search build for that footage alone.
    """
    state = get_state(ctx)
    index_dir = state.config.index_dir

    missing = [p for p in sources if not (p / "catalog.sqlite").is_file()]
    if missing:
        raise typer.BadParameter(f"not index directories: {', '.join(map(str, missing))}")
    same = [p for p in sources if p.resolve() == index_dir.resolve()]
    if same:
        raise typer.BadParameter("cannot merge an index into itself")

    totals = {"files": 0, "streams": 0, "frames": 0, "skipped": 0}
    with Catalog.open(index_dir) as target:
        for source in sources:
            with Catalog.open(source) as other:
                # Identity metadata has to be carried over on a first merge into
                # a fresh index, or the target would have no fingerprint identity.
                for key in ("algo_id", "prep_id", "sample_fps", "crop_variants"):
                    value = other.get_meta(key)
                    if value is not None and target.get_meta(key) is None:
                        target.set_meta(key, value)
                try:
                    counts = target.absorb(
                        other, progress=(lambda e: emit(state, e)) if state.json_mode else None
                    )
                except IndexError_ as exc:
                    state.err.print(f"[red]{source}: {exc}[/red]")
                    raise typer.Exit(code=1) from exc
            for name, count in counts.items():
                totals[name] += count
            if not state.quiet and not state.json_mode:
                state.err.print(
                    f"{source}: +{counts['files']} files, {counts['streams']} streams, "
                    f"{counts['frames']:,} frames ({counts['skipped']} already present)"
                )

        if shard:
            AnnIndex.build_or_load(
                target, progress=(lambda e: emit(state, e)) if state.json_mode else None
            )
        ann = describe_index(target)

    if state.json_mode:
        emit(state, {"event": "result", **totals, "search_index": ann})
    elif not state.quiet:
        state.err.print(
            f"Merged {totals['files']} file(s), {totals['frames']:,} frames. "
            f"Search index: {ann['shards']:,} shards, {ann['codes']:,} codes"
            + (
                f", {ann['streams_unsharded']:,} streams awaiting sharding."
                if ann["streams_unsharded"]
                else "."
            )
        )
