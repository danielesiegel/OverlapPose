"""Check that a manifest describes the footage a seller actually showed.

A buyer evaluating 10,000 hours can see one or two. The sample is the only
footage it holds pixels for, so it can be fingerprinted locally and looked up
in the manifest: if it was drawn from the offered data, nearly all of it must
appear.

Proves the sampled hours are present. Does not prove the rest of the manifest
corresponds to footage the seller holds - only `overlap verify` against
delivered bytes can do that.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from overlap.match.compare import compare_manifest
from overlap.store.catalog import Catalog
from overlap.store.ingest_manifest import manifest_as_corpus
from overlap.store.manifest import build_manifest, read_manifest

AUDIT_SCHEMA = "audit/1"

# Below this share of the sample being present, the manifest and the sample are
# not describing the same footage. Set well under 100% because the evidence floor
# discards runs under min_run_s, so a short sample file can legitimately miss.
CONSISTENT_AT = 90.0


def audit_sample(
    manifest_path: Path,
    sample_dir: Path,
    *,
    sample_fps: float | None = None,
    min_run_s: float = 10.0,
    nprobe: int = 64,
    threads: int = 0,
    workers: int = 0,
    max_manifest_bytes: int | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    """Fingerprint the shared sample and look for it in the manifest."""
    from overlap.ingest.pipeline import index_paths

    emit = progress or (lambda _e: None)
    manifest = read_manifest(manifest_path, max_bytes=max_manifest_bytes)
    # Fingerprint the sample on the *seller's* grid. Both sides sample at
    # (i + 0.5) / fps from the start of the file, so matching their density makes
    # the two grids land on the same instants for footage on the same timeline -
    # which the sample is. Indexing denser instead puts most query frames between
    # the seller's samples, and the run-density gate then rejects a true match:
    # measured as 0% found on a sample that was drawn from the offer.
    if sample_fps is None:
        sample_fps = manifest.sample_fps or 4.0

    with tempfile.TemporaryDirectory(prefix="overlap-audit-") as tmp:
        sample_index = Path(tmp) / "sample.ovl"
        stats = index_paths(
            [sample_dir],
            sample_index,
            sample_fps=sample_fps,
            # The sample is compared against fingerprints only, so crop
            # geometries would have nothing on the other side to match.
            crop_ladder="",
            workers=workers,
            progress=progress,
        )
        emit({"event": "stage", "stage": "sample_indexed", "files": stats.indexed})
        with Catalog.open(sample_index) as sample_cat:
            sample_manifest, _ = build_manifest(sample_cat)
        catalog = manifest_as_corpus(manifest, Path(tmp) / "offer.ovl")
        try:
            report = compare_manifest(
                sample_manifest,
                catalog,
                min_run_s=min_run_s,
                nprobe=nprobe,
                threads=threads,
                progress=progress,
            )
        finally:
            catalog.close()

    found_pct = report["summary"]["overlap_pct"]
    unmatched = sorted(
        f["relpath"] for f in report["files"] if not f["matches"] and not f["sha256_exact"]
    )
    return {
        "schema": AUDIT_SCHEMA,
        "manifest": str(manifest_path),
        "manifest_label": manifest.label,
        "manifest_hours": round(manifest.total_hours, 2),
        "manifest_sample_fps": manifest.sample_fps,
        "sample_dir": str(sample_dir),
        "sample_files": stats.indexed,
        "sample_hours": round(sample_manifest.total_hours, 2),
        "sample_fps": sample_fps,
        "sample_found_pct": found_pct,
        "consistent": found_pct >= CONSISTENT_AT,
        "unmatched_sample_files": unmatched,
        "sample_share_of_offer": (
            round(sample_manifest.total_hours / manifest.total_hours * 100, 3)
            if manifest.total_hours
            else 0.0
        ),
        "note": (
            "Confirms the sampled hours appear in the manifest. It cannot confirm the "
            "rest of the manifest corresponds to footage the seller holds - only "
            "`overlap verify` against the delivered bytes can do that."
        ),
        "detail": report,
    }
