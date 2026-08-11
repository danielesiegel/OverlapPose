"""`overlap export` - write a shareable fingerprint manifest (no raw data)."""

from __future__ import annotations

from pathlib import Path
from time import strftime

import typer

from overlap.cli._console import emit_document, get_state
from overlap.store.catalog import Catalog
from overlap.store.manifest import (
    SUFFIX,
    export_manifest,
    export_manifest_split,
    read_manifest,
)


def export_cmd(
    ctx: typer.Context,
    output: Path | None = typer.Option(
        None, "-o", "--output", help="Manifest path or directory (default: ./corpus-<date>.ovlm)."
    ),
    label: str | None = typer.Option(
        None, "--label", help="Human-readable label embedded in the manifest."
    ),
    anonymize_paths: bool = typer.Option(
        False,
        "--anonymize-paths",
        help="Replace file names with content-derived ids (pre-sale manifests).",
    ),
    stride: int = typer.Option(
        0,
        "--stride",
        min=0,
        help=(
            "Export every Nth fingerprint. 0 = auto: stride down to ~1 fps "
            "(the corpus indexes densely at 4 fps for matching robustness, "
            "but manifests only need ~1 fps to be found)."
        ),
    ),
    split_gb: float = typer.Option(
        0.0,
        "--split-gb",
        min=0.0,
        help=(
            "Write a directory of parts, each about this many GB, instead of one "
            "file. Needed for large corpora: at 4 fps, 96,000 hours is ~44 GB, "
            "past what most hosts serve as a single object. Parts split on file "
            "boundaries, so each is a readable manifest on its own."
        ),
    ),
) -> None:
    """Export the local index as a compact .ovlm manifest to send to a counterparty.

    The manifest contains perceptual fingerprints and file metadata - never
    frames or pixels. Note that fingerprints still leak coarse visual
    structure; treat manifests as confidential business documents.
    """
    state = get_state(ctx)
    index_dir = state.config.index_dir
    if not (index_dir / "catalog.sqlite").exists():
        raise typer.BadParameter(f"no index at {index_dir}; run `overlap index` first")
    default_name = f"corpus-{strftime('%Y%m%d')}"
    out = output or Path(default_name + ("" if split_gb else SUFFIX))
    # stride 0 = auto: thin each stream to ~1 fps regardless of the rate it was
    # indexed at, so mixed-rate corpora export uniformly.
    shared = {
        "label": label,
        "anonymize_paths": anonymize_paths,
        "stride": max(1, stride),
        "target_fps": 1.0 if stride == 0 else None,
    }
    parts: list[Path] = []
    with Catalog.open(index_dir) as catalog:
        if split_gb:
            parts = export_manifest_split(
                catalog, out, part_bytes=int(split_gb * 1e9), **shared  # type: ignore[arg-type]
            )
            manifest = read_manifest(out)
        else:
            manifest = export_manifest(catalog, out, **shared)  # type: ignore[arg-type]

    info: dict[str, object] = {
        "manifest": str(out),
        "files": len(manifest.files),
        "streams": len(manifest.streams),
        "hours": round(manifest.total_hours, 2),
        "frames": manifest.total_frames,
        "bytes": (
            sum(p.stat().st_size for p in parts) if parts else out.stat().st_size
        ),
        "parts": len(parts),
        "merkle_root": manifest.merkle_root.hex(),
    }
    if state.json_mode:
        emit_document(info)
    else:
        state.err.print(
            f"Wrote [bold]{out}[/bold]: {len(manifest.files)} files, "
            f"{round(manifest.total_hours, 2)} h, {manifest.total_frames:,} fingerprints, "
            f"{out.stat().st_size / 1e6:.1f} MB."
        )
