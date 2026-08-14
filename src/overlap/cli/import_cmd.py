"""`overlap import` - add a manifest's fingerprints to the local index."""

from __future__ import annotations

from pathlib import Path

import typer

from overlap.cli._console import emit, get_state
from overlap.errors import IndexError_
from overlap.store.annindex import AnnIndex, describe_index
from overlap.store.catalog import Catalog
from overlap.store.ingest_manifest import import_manifest
from overlap.store.manifest import PARTS_INDEX, read_manifest


def import_cmd(
    ctx: typer.Context,
    manifests: list[Path] = typer.Argument(
        ..., help="Manifest files (.ovlm) or part directories to import."
    ),
    label: str | None = typer.Option(
        None, "--label", help="Where this footage came from; recorded against every file."
    ),
    shard: bool = typer.Option(
        True,
        "--shard/--no-shard",
        help="Build search shards now, or leave it for the next comparison.",
    ),
) -> None:
    """Import fingerprints from a manifest, without the footage behind them.

    Use this to screen offers against data you have not bought: fingerprints of
    a published dataset, or an earlier offer you declined. The imported footage
    becomes part of what `overlap compare` checks against.

    Two limits, both reported rather than assumed away. A manifest carries no
    pixels, so no crop geometries can be built for it and a cropped copy of the
    imported footage will not be found. And a manifest carries whatever density
    the exporter chose: strided below 4 fps, recall against re-cut footage drops
    sharply, so prefer manifests exported at full density for this purpose.
    """
    state = get_state(ctx)
    index_dir = state.config.index_dir
    # An index built only from manifests never passed through `overlap index`,
    # so nothing had recorded the shard budget and every build fell back to the
    # 32M default - about 1.3 GB resident per shard. Screening offers against a
    # published dataset is exactly the case that should run on an ordinary
    # machine, so the configured budget is recorded here too.
    shard_codes = int(state.config.get("index.shard_codes"))

    def readable(path: Path) -> bool:
        # A manifest may arrive as one file or as a directory of parts; the
        # sender's packaging choice is not the importer's problem.
        return path.is_file() or (path.is_dir() and (path / PARTS_INDEX).is_file())

    missing = [p for p in manifests if not readable(p)]
    if missing:
        raise typer.BadParameter(
            f"not a manifest or part directory: {', '.join(map(str, missing))}"
        )

    added = 0
    hours = 0.0
    with Catalog.open(index_dir, expected_meta={"shard_codes": str(shard_codes)}) as catalog:
        for path in manifests:
            manifest = read_manifest(path)
            try:
                n = import_manifest(manifest, catalog, label=label or manifest.label or path.stem)
            except IndexError_ as exc:
                state.err.print(f"[red]{path}: {exc}[/red]")
                raise typer.Exit(code=1) from exc
            added += n
            hours += manifest.total_hours
            if not state.quiet and not state.json_mode:
                skipped = len(manifest.files) - n
                state.err.print(
                    f"{path.name}: +{n} files, {manifest.total_hours:,.1f} h at "
                    f"{manifest.sample_fps:g} fps"
                    + (f" ({skipped} already present)" if skipped else "")
                )
            if manifest.sample_fps < 4.0 and not state.quiet:
                state.err.print(
                    f"[yellow]{path.name} was exported at {manifest.sample_fps:g} fps. "
                    f"Measured against re-cut and transcoded footage, a 1 fps corpus "
                    f"recovers 40% of what 4 fps recovers - ask for a denser export if "
                    f"this footage matters.[/yellow]"
                )
        if shard:
            AnnIndex.build_or_load(
                catalog, progress=(lambda e: emit(state, e)) if state.json_mode else None
            )
        ann = describe_index(catalog)

    if state.json_mode:
        emit(state, {"event": "result", "files_added": added, "hours": round(hours, 2),
                     "search_index": ann})
    elif not state.quiet:
        state.err.print(
            f"Imported {added} file(s), {hours:,.1f} h of fingerprints. "
            f"Search index: {ann['shards']:,} shards, {ann['codes']:,} codes."
        )
