"""`overlap inspect` - describe any file overlap understands."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from overlap.cli._console import emit_document, get_state
from overlap.readers import SamplePolicy, reader_for
from overlap.store.manifest import MAGIC, read_manifest


def inspect_cmd(
    ctx: typer.Context,
    target: Path = typer.Argument(..., exists=True, readable=True),
    deep: bool = typer.Option(False, "--deep", help="For media: decode a probe frame per stream."),
) -> None:
    """Inspect a media file, a .ovlm manifest, or a report JSON."""
    state = get_state(ctx)
    out = Console()

    with target.open("rb") as f:
        head = f.read(4)

    if head == MAGIC:
        manifest = read_manifest(target)
        doc = {
            "kind": "manifest",
            "algo_id": manifest.algo_id,
            "prep_id": manifest.prep_id,
            "sample_fps": manifest.sample_fps,
            "label": manifest.label,
            "tool": manifest.tool_version,
            "files": len(manifest.files),
            "streams": len(manifest.streams),
            "hours": round(manifest.total_hours, 2),
            "frames": manifest.total_frames,
            "merkle_root": manifest.merkle_root.hex(),
            "bytes_per_hour": (
                round(target.stat().st_size / manifest.total_hours)
                if manifest.total_hours
                else 0
            ),
            # Density decides what this manifest can be used for. As a query
            # against a dense corpus, 1 fps is fine. As a corpus itself - imported,
            # or compared against another manifest - it is not: measured against
            # re-cut footage, a 1 fps corpus recovers 40% of what 4 fps recovers.
            "dense_enough_to_be_a_corpus": manifest.sample_fps >= 4.0,
        }
        _emit(state, out, doc)
        return

    if target.suffix.lower() == ".json":
        doc = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(doc, dict) and str(doc.get("schema", "")).startswith("report/"):
            _emit(state, out, {"kind": "report", "summary": doc.get("summary", {})})
            return

    reader = reader_for(target)
    if reader is None:
        raise typer.BadParameter(f"{target} is not a media file or manifest overlap understands")
    with reader.open(target) as session:
        streams = []
        for info in session.streams():
            entry = {
                "stream_key": info.stream_key,
                "codec": info.codec,
                "resolution": f"{info.width}x{info.height}",
                "fps": info.native_fps,
                "duration_s": round((info.duration_ms or 0) / 1000.0, 1),
            }
            if deep:
                sample = next(iter(session.sample(info.stream_key, SamplePolicy(fps=1.0))), None)
                entry["decodable"] = sample is not None
            streams.append(entry)
    _emit(state, out, {"kind": "media", "reader": reader.name, "streams": streams})


def _emit(state, out: Console, doc) -> None:  # type: ignore[no-untyped-def]
    if state.json_mode:
        emit_document(doc)
        return
    if doc.get("kind") == "media":
        table = Table(title=f"media ({doc['reader']})")
        for col in ("stream_key", "codec", "resolution", "fps", "duration_s"):
            table.add_column(col)
        for s in doc["streams"]:
            table.add_row(
                s["stream_key"],
                s["codec"],
                s["resolution"],
                str(s["fps"]),
                str(s["duration_s"]),
            )
        out.print(table)
        return
    out.print_json(json.dumps(doc))
    if doc.get("kind") == "manifest" and not doc["dense_enough_to_be_a_corpus"]:
        # Worth saying out loud, because the failure is silent: used as a corpus,
        # a sparse manifest reports low overlap on footage that fully matches.
        out.print(
            f"[yellow]Exported at {doc['sample_fps']:g} fps. Fine as an offer to "
            f"compare against a corpus you hold. Not dense enough to be used as "
            f"the corpus itself (overlap import, or compare --against): measured "
            f"against re-cut footage, 1 fps recovers 40% of what 4 fps recovers. "
            f"Ask the sender for --stride 1 if you need that.[/yellow]"
        )
