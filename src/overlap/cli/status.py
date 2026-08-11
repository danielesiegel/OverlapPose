"""`overlap status` - corpus statistics for the local index."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from overlap.cli._console import emit_document, get_state
from overlap.store.annindex import describe_index
from overlap.store.catalog import Catalog


def status(ctx: typer.Context) -> None:
    """Show what the local index contains."""
    state = get_state(ctx)
    index_dir = state.config.index_dir
    if not (index_dir / "catalog.sqlite").exists():
        if state.json_mode:
            emit_document({"index_dir": str(index_dir), "exists": False})
        else:
            state.err.print(
                f"No index at {index_dir} yet - run [bold]overlap index <paths>[/bold] first."
            )
        return

    with Catalog.open(index_dir) as catalog:
        s = catalog.stats()
        meta = {
            k: catalog.get_meta(k) for k in ("schema_version", "algo_id", "prep_id", "sample_fps")
        }
        # Streams may legitimately differ (settings are per stream), so report
        # the mix rather than implying the whole corpus shares one setting.
        rates: dict[float, int] = {}
        rungs: dict[int, int] = {}
        for row in catalog.iter_streams():
            rates[row.sample_fps] = rates.get(row.sample_fps, 0) + 1
            rungs[row.n_crop_rungs] = rungs.get(row.n_crop_rungs, 0) + 1
        ann = describe_index(catalog)

    hours = s.total_duration_ms / 3_600_000
    if state.json_mode:
        emit_document(
            {
                "index_dir": str(index_dir),
                "exists": True,
                "meta": meta,
                "files_done": s.files_done,
                "files_error": s.files_error,
                "files_skipped": s.files_skipped,
                "streams": s.streams,
                "frames": s.frames,
                "hours": round(hours, 2),
                "db_bytes": s.db_bytes,
                "streams_by_sample_fps": {str(k): v for k, v in sorted(rates.items())},
                "streams_by_crop_variants": {str(k): v for k, v in sorted(rungs.items())},
                "search_index": ann,
            }
        )
        return

    out = Console()
    table = Table(title=f"Index: {index_dir}")
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_row("files indexed", str(s.files_done))
    table.add_row("files errored", str(s.files_error))
    table.add_row("files skipped", str(s.files_skipped))
    table.add_row("streams", str(s.streams))
    table.add_row("fingerprinted frames", f"{s.frames:,}")
    table.add_row("footage", f"{hours:,.1f} h")
    table.add_row("catalog size", f"{s.db_bytes / 1e6:,.1f} MB")
    table.add_row("fingerprint", f"{meta['algo_id']}/{meta['prep_id']}")
    table.add_row(
        "sampling",
        ", ".join(f"{fps:g} fps x{n}" for fps, n in sorted(rates.items())) or "-",
    )
    table.add_row(
        "crop variants",
        ", ".join(f"{k} x{n}" for k, n in sorted(rungs.items())) or "-",
    )
    table.add_row(
        "search index",
        f"{ann['shards']:,} shards, {ann['codes']:,} codes, "
        f"{ann['bytes_on_disk'] / 1e9:,.2f} GB",
    )
    if ann["streams_unsharded"]:
        # Not a warning: the hashes are already durable. It only means the next
        # comparison shards these first.
        table.add_row("awaiting sharding", f"{ann['streams_unsharded']:,} streams")
    out.print(table)
