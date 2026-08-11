"""Indexing pipeline: walk paths -> decode/hash in workers -> single-writer catalog.

Workers are OS processes (spawn context - Windows-safe) that never touch
SQLite; the parent is the only writer and commits one transaction per file,
which is the resumability unit. Progress flows through a callback so the CLI
(Rich/NDJSON) and the web UI (SSE) render the same event stream.
"""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from overlap.hashing import (
    PREP_ID,
    CropVariant,
    PdqKernel,
    apply_crop,
    build_crop_variants,
    crop_variants_spec,
    detect_border_crop,
    to_gray,
)
from overlap.hashing.pdq_numpy import unpack_bits
from overlap.hashing.prep import BORDER_PROBE_FRAMES
from overlap.ingest.merkle import sha256_file
from overlap.ingest.model import FileResult, StreamResult
from overlap.readers import SamplePolicy, reader_for, supported_extensions
from overlap.store.catalog import Catalog

if TYPE_CHECKING:
    from collections.abc import Iterator

ProgressCb = Callable[[dict[str, Any]], None]


@dataclass
class IndexStats:
    indexed: int = 0
    skipped: int = 0
    errors: int = 0
    streams: int = 0
    frames: int = 0
    failed_files: list[tuple[str, str]] = field(default_factory=list)

    @property
    def exit_partial(self) -> bool:
        return self.errors > 0


def discover_files(
    paths: list[Path],
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    follow_symlinks: bool = False,
) -> list[tuple[Path, Path]]:
    """Resolve inputs to (abspath, root) pairs for every supported file.

    ``root`` is the directory the user named (or the file's parent for direct
    file arguments); relpaths in the index and manifests are relative to it.
    """
    exts = supported_extensions()
    found: list[tuple[Path, Path]] = []

    def matches(p: Path) -> bool:
        rel = p.as_posix()
        if include and not any(fnmatch.fnmatch(rel, g) for g in include):
            return False
        return not (exclude and any(fnmatch.fnmatch(rel, g) for g in exclude))

    for raw in paths:
        base = raw.resolve()
        if base.is_file():
            if base.suffix.lower() in exts and matches(base):
                found.append((base, base.parent))
            continue
        for dirpath, _dirnames, filenames in os.walk(base, followlinks=follow_symlinks):
            for name in sorted(filenames):
                p = Path(dirpath) / name
                if p.suffix.lower() in exts and matches(p):
                    found.append((p, base))
    # Deterministic order; dedupe files reachable via multiple arguments.
    unique = {str(p): (p, root) for p, root in found}
    return [unique[k] for k in sorted(unique)]


def process_file(
    abspath: str,
    root: str,
    sample_fps: float,
    crop_variants: tuple[CropVariant, ...] = (),
) -> FileResult:
    """Worker entry: fingerprint one file. Must stay picklable/top-level."""
    path = Path(abspath)
    stat = path.stat()
    rel = path.relative_to(root).as_posix() if Path(root) in path.parents else path.name
    base = FileResult(
        abspath=abspath,
        root=root,
        relpath=rel,
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        sha256=b"",
        container=path.suffix.lower().lstrip("."),
        status="done",
    )
    try:
        base.sha256 = sha256_file(path)
        reader = reader_for(path)
        if reader is None:
            base.status = "skipped"
            base.error = "no reader for this file type"
            return base
        kernel = PdqKernel()
        with reader.open(path) as session:
            infos = session.streams()
            if not infos:
                base.status = "skipped"
                base.error = "no visual streams found"
                return base
            for info in infos:
                stream = _fingerprint_stream(session, info, kernel, sample_fps, crop_variants)
                if stream is not None:
                    base.streams.append(stream)
        if not base.streams:
            base.status = "skipped"
            base.error = "no decodable frames in any stream"
    except Exception as exc:  # noqa: BLE001 - workers must report, not die
        base.status = "error"
        base.error = f"{type(exc).__name__}: {exc}"
        base.streams = []
    return base


def _fingerprint_stream(
    session: Any,
    info: Any,
    kernel: PdqKernel,
    sample_fps: float,
    crop_variants: tuple[CropVariant, ...] = (),
) -> StreamResult | None:
    """Decode-sample-prep-hash one stream; None when it yields no frames."""
    policy = SamplePolicy(fps=sample_fps)
    samples: Iterator[Any] = session.sample(info.stream_key, policy)

    probe: list[np.ndarray] = []
    hashes: list[bytes] = []
    mirrors: list[bytes] = []
    crop_hashes: list[bytes] = []
    crop_mirrors: list[bytes] = []
    qualities = bytearray()
    flags = bytearray()
    votes = np.zeros(256, dtype=np.int32)
    n = 0
    crop = None
    last_t_ms = 0

    def hash_gray(gray: np.ndarray) -> None:
        nonlocal n
        fh = kernel.hash_frame(gray)
        hashes.append(fh.hash)
        mirrors.append(fh.mirror)
        qualities.append(fh.quality)
        flags.append(fh.flags)
        votes[unpack_bits(fh.hash)] += 1
        # Crop variants: hashed from the same decoded frame, so a cropped
        # copy of this footage lands near one of these geometries.
        for variant in crop_variants:
            fh_c = kernel.hash_frame(variant.apply(gray))
            crop_hashes.append(fh_c.hash)
            crop_mirrors.append(fh_c.mirror)
        n += 1

    for t_ms, image in samples:
        last_t_ms = t_ms
        gray = to_gray(image)
        if crop is None:
            probe.append(gray)
            if len(probe) >= BORDER_PROBE_FRAMES:
                crop = detect_border_crop(probe)
                for g in probe:
                    hash_gray(apply_crop(g, crop))
                probe = []
            continue
        hash_gray(apply_crop(gray, crop))

    if crop is None:  # short stream: fewer samples than the probe window
        crop = detect_border_crop(probe)
        for g in probe:
            hash_gray(apply_crop(g, crop))

    if n == 0:
        return None

    sketch = np.packbits((votes * 2 > n).astype(np.uint8)).tobytes()
    duration_ms = info.duration_ms or (last_t_ms + int(500 / sample_fps))
    return StreamResult(
        stream_key=info.stream_key,
        codec=info.codec,
        width=info.width,
        height=info.height,
        native_fps=info.native_fps,
        duration_ms=duration_ms,
        sample_fps=sample_fps,
        algo_id=kernel.algo_id,
        prep_id=PREP_ID,
        border_crop=crop.as_str(),
        n_frames=n,
        hashes=b"".join(hashes),
        mirrors=b"".join(mirrors),
        qualities=bytes(qualities),
        flags=bytes(flags),
        sketch=sketch,
        n_crop_rungs=len(crop_variants),
        crop_hashes=b"".join(crop_hashes),
        crop_mirrors=b"".join(crop_mirrors),
    )


# The "mild" preset (overlap.coverage.PRESETS): one centred rung and one bottom
# rung, calibrated to the crops that leave footage still sellable. Kept in step
# with the config defaults; overlap.coverage is the single source for both.
DEFAULT_CROP_LADDER = "0.94"
DEFAULT_CROP_EDGES = "bottom:0.06"


def index_paths(
    paths: list[Path],
    index_dir: Path,
    *,
    sample_fps: float = 4.0,
    crop_ladder: str | tuple[float, ...] | None = None,
    crop_edges: str | None = None,
    workers: int = 0,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    follow_symlinks: bool = False,
    reindex: bool = False,
    shard_codes: int | None = None,
    progress: ProgressCb | None = None,
) -> IndexStats:
    """Fingerprint ``paths`` into ``index_dir``. The public indexing API."""
    emit = progress or (lambda _e: None)
    if crop_ladder is None:
        crop_ladder = DEFAULT_CROP_LADDER
    if crop_edges is None:
        crop_edges = DEFAULT_CROP_EDGES
    if not isinstance(crop_ladder, str):
        crop_ladder = ",".join(f"{k:.2f}" for k in crop_ladder)
    crop_variants = build_crop_variants(crop_ladder, crop_edges)
    kernel = PdqKernel()
    expected_meta = {
        "algo_id": kernel.algo_id,
        "prep_id": PREP_ID,
        "sample_fps": repr(sample_fps),
        "crop_variants": crop_variants_spec(crop_variants),
    }
    if shard_codes is not None:
        # Recorded so a comparison inherits the memory budget the index was
        # built for, instead of every caller having to pass it along. The default
        # does not come from overlap.store.annindex: importing it
        # pulls in FAISS, and this module is re-imported by every spawned
        # indexing worker, none of which searches anything.
        expected_meta["shard_codes"] = str(shard_codes)
    stats = IndexStats()
    work: list[tuple[str, str]] = []

    def note_setting_change(key: str, old: str, new: str) -> None:
        emit({"event": "setting_change", "key": key, "previous": old, "now": new})

    with Catalog.open(
        index_dir, expected_meta=expected_meta, on_setting_change=note_setting_change
    ) as catalog:
        for found_path, found_root in discover_files(paths, include, exclude, follow_symlinks):
            st = found_path.stat()
            if not reindex and catalog.is_done(str(found_path), st.st_size, st.st_mtime_ns):
                stats.skipped += 1
                emit(
                    {
                        "event": "file",
                        "path": str(found_path),
                        "status": "skipped",
                        "reason": "unchanged",
                    }
                )
                continue
            work.append((str(found_path), str(found_root)))

        emit({"event": "start", "total_files": len(work), "already_indexed": stats.skipped})

        def consume(result: FileResult) -> None:
            catalog.store_file_result(result)
            if result.status == "done":
                stats.indexed += 1
                stats.streams += len(result.streams)
                stats.frames += sum(s.n_frames for s in result.streams)
            elif result.status == "error":
                stats.errors += 1
                stats.failed_files.append((result.abspath, result.error or ""))
            else:
                stats.skipped += 1
            emit(
                {
                    "event": "file",
                    "path": result.abspath,
                    "status": result.status,
                    "streams": len(result.streams),
                    "frames": sum(s.n_frames for s in result.streams),
                    **({"error": result.error} if result.error else {}),
                }
            )

        n_workers = workers if workers > 0 else max(1, (os.cpu_count() or 2) - 2)
        if n_workers == 1 or len(work) <= 1:
            for work_path, work_root in work:
                consume(process_file(work_path, work_root, sample_fps, crop_variants))
        else:
            ctx = get_context("spawn")
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                pending: set[Future[FileResult]] = set()
                queue = list(reversed(work))
                while queue or pending:
                    while queue and len(pending) < n_workers * 2:
                        work_path, work_root = queue.pop()
                        pending.add(
                            pool.submit(
                                process_file, work_path, work_root, sample_fps, crop_variants
                            )
                        )
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for fut in done:
                        consume(fut.result())

    emit(
        {
            "event": "summary",
            "indexed": stats.indexed,
            "skipped": stats.skipped,
            "errors": stats.errors,
            "streams": stats.streams,
            "frames": stats.frames,
        }
    )
    return stats
