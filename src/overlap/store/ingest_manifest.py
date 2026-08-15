"""Load a manifest's fingerprints into a catalog, so a manifest can be a corpus.

Needed when a buyer holds two manifests and the pixels of neither: two sellers
offering the same footage, or a published fingerprint set for a public dataset.

Two limits, reported rather than assumed away. A manifest has no pixels, so no
crop geometries can be built for imported footage and cropped copies of it will
not be found. And it carries whatever density the exporter chose; below 4 fps,
recall against re-cut footage drops sharply (docs/architecture.md).

Mirror digests are derived rather than stored: mirroring negates the even DCT
columns, so inverting those bit positions reconstructs the mirror to within a
few bits of 256.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from overlap.errors import IndexError_
from overlap.hashing.base import HASH_BYTES
from overlap.hashing.pdq_numpy import mirror_of_packed
from overlap.ingest.model import FileResult, StreamResult

if TYPE_CHECKING:
    from pathlib import Path

    from overlap.store.catalog import Catalog
    from overlap.store.manifest import Manifest, ManifestStream

# Marks streams whose pixels were never available, so the report can say which
# corpus footage could only be matched by its plain fingerprints.
FINGERPRINTS_ONLY = "fingerprints-only"


def import_manifest(manifest: Manifest, catalog: Catalog, *, label: str | None = None) -> int:
    """Copy a manifest's fingerprints into ``catalog``. Returns files added.

    Idempotent by content: a file whose sha256 is already present is skipped,
    so re-importing after an interruption costs nothing and cannot double-count.
    """
    for key, value in (("algo_id", manifest.algo_id), ("prep_id", manifest.prep_id)):
        stored = catalog.get_meta(key)
        if stored is None:
            catalog.set_meta(key, value)
        elif stored != value:
            raise IndexError_(
                f"cannot import: this index uses {key}={stored}, the manifest has "
                f"{key}={value}; these fingerprints are not comparable"
            )

    have = {bytes(r["sha256"]) for r in catalog.file_rows() if r["sha256"]}
    streams_by_file: dict[int, list[ManifestStream]] = {}
    for stream in manifest.streams:
        streams_by_file.setdefault(stream.file_idx, []).append(stream)

    added = 0
    origin = label or manifest.label or "imported manifest"
    for file_idx, mfile in enumerate(manifest.files):
        if mfile.sha256 in have:
            continue
        streams = streams_by_file.get(file_idx, [])
        if not streams:
            continue
        catalog.store_file_result(
            FileResult(
                # The path records where the footage came from, not a local file:
                # nothing on this machine holds these pixels.
                abspath=f"{FINGERPRINTS_ONLY}:{origin}/{mfile.relpath}",
                root=f"{FINGERPRINTS_ONLY}:{origin}",
                relpath=mfile.relpath,
                size_bytes=mfile.size,
                mtime_ns=0,
                sha256=mfile.sha256,
                container=mfile.container,
                status="done",
                streams=[_as_stream_result(s, manifest) for s in streams],
            )
        )
        have.add(mfile.sha256)
        added += 1
    return added


def _as_stream_result(stream: ManifestStream, manifest: Manifest) -> StreamResult:
    n = stream.n_frames
    hashes = stream.hashes
    if len(hashes) != n * HASH_BYTES:
        raise IndexError_(
            f"manifest stream {stream.stream_key!r} declares {n} frames but carries "
            f"{len(hashes)} hash bytes"
        )
    return StreamResult(
        stream_key=stream.stream_key,
        codec=stream.codec,
        width=stream.width,
        height=stream.height,
        native_fps=None,
        duration_ms=stream.duration_ms,
        sample_fps=stream.sample_fps,
        algo_id=manifest.algo_id,
        prep_id=manifest.prep_id,
        border_crop="0,0,0,0",
        n_frames=n,
        hashes=hashes,
        mirrors=mirror_of_packed(hashes),
        qualities=stream.qualities,
        flags=stream.flags,
        sketch=bytes(HASH_BYTES),
        n_crop_rungs=0,
    )


def manifest_as_corpus(manifest: Manifest, index_dir: Path, *, label: str | None = None) -> Catalog:
    """Open a throwaway catalog holding one manifest, ready to be compared against."""
    from overlap.store.catalog import Catalog

    catalog = Catalog.open(index_dir)
    import_manifest(manifest, catalog, label=label)
    return catalog


def import_manifest_streaming(
    path: Path, catalog: Catalog, *, label: str | None = None
) -> dict[str, Any]:
    """Import a manifest without ever holding all of it.

    Returns ``{added, label, n_files, hours, sample_fps}`` - the header facts a
    caller wants to report, gathered during the single pass rather than by
    re-reading the file.

    Same result as :func:`import_manifest`, but the manifest is walked stream by
    stream instead of being decoded in full first. On a large published catalog
    a single part carries about 1.5 GB of frames, and decoding it whole needed
    over 3 GB - which put importing a public corpus out of reach of the machines
    the import path exists to serve.

    Streams for one file are contiguous in the manifest, so a file is written as
    soon as the next file's first stream appears, and nothing older than the
    current file is kept.
    """
    from overlap.store.manifest import PARTS_INDEX, iter_manifest_streams

    # A split export is a directory of parts. Parts break on file boundaries, so
    # each is a complete manifest and importing them in turn is equivalent to
    # importing the whole - and keeps the streaming guarantee, since only one
    # part is ever open.
    if path.is_dir():
        import json as _json

        index = _json.loads((path / PARTS_INDEX).read_text(encoding="utf-8"))
        rolled = {"added": 0, "label": index.get("label"), "n_files": 0,
                  "hours": 0.0, "sample_fps": 0.0}
        for entry in index.get("parts", []):
            part = path / str(entry["name"])
            got = import_manifest_streaming(part, catalog, label=label)
            rolled["added"] += int(got["added"])
            rolled["n_files"] += int(got["n_files"])
            rolled["hours"] += float(got["hours"])
            rolled["sample_fps"] = rolled["sample_fps"] or got["sample_fps"]
        return rolled

    have: set[bytes] = {bytes(r["sha256"]) for r in catalog.file_rows() if r["sha256"]}
    added = 0
    manifest: Manifest | None = None
    current_idx: int | None = None
    pending: list[ManifestStream] = []

    def flush() -> int:
        nonlocal pending, current_idx
        if manifest is None or current_idx is None or not pending:
            pending = []
            return 0
        mfile = manifest.files[current_idx]
        if mfile.sha256 in have:
            pending = []
            return 0
        origin = label or manifest.label or "imported manifest"
        catalog.store_file_result(
            FileResult(
                abspath=f"{FINGERPRINTS_ONLY}:{origin}/{mfile.relpath}",
                root=f"{FINGERPRINTS_ONLY}:{origin}",
                relpath=mfile.relpath,
                size_bytes=mfile.size,
                mtime_ns=0,
                sha256=mfile.sha256,
                container=mfile.container,
                status="done",
                streams=[_as_stream_result(s, manifest) for s in pending],
            )
        )
        have.add(mfile.sha256)
        pending = []
        return 1

    total_ms = 0
    sample_fps = 0.0
    for manifest, stream in iter_manifest_streams(path):
        if current_idx is None:
            _check_identity(manifest, catalog)
        if stream.file_idx != current_idx:
            added += flush()
            current_idx = stream.file_idx
        pending.append(stream)
        total_ms += stream.duration_ms
        sample_fps = sample_fps or stream.sample_fps
    added += flush()
    return {
        "added": added,
        "label": (manifest.label if manifest else None),
        "n_files": (len(manifest.files) if manifest else 0),
        "hours": total_ms / 3.6e6,
        "sample_fps": sample_fps,
    }


def _check_identity(manifest: Manifest, catalog: Catalog) -> None:
    """Refuse to mix fingerprints that mean different things."""
    for key, value in (("algo_id", manifest.algo_id), ("prep_id", manifest.prep_id)):
        stored = catalog.get_meta(key)
        if stored is None:
            catalog.set_meta(key, value)
        elif stored != value:
            raise IndexError_(
                f"cannot import: this index uses {key}={stored}, the manifest has "
                f"{key}={value}; these fingerprints are not comparable"
            )
