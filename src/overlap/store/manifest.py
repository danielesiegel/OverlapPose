"""The .ovlm manifest - the fingerprint file a vendor sends to a lab.

Binary layout (documented in docs/manifest-spec.md)::

    magic "OVLM" | u16 format_version | u32 header_len | header JSON (utf-8)
    section bytes, concatenated, each zstd-compressed independently

The JSON header carries schema/algo/prep identifiers, dataset metadata
(including the Merkle root binding relpaths to file sha256s), and a section
table with offsets, lengths, and per-section sha256 for integrity. Sections:

    files.msgpack    [[relpath, size, sha256, container], ...] sorted by relpath
    streams.msgpack  [[file_idx, stream_key, codec, w, h, duration_ms,
                       sample_fps, n_frames], ...]
    frames.bin       per stream: identity hashes (n*32) ‖ qualities (n) ‖ flags (n)

Manifests carry only identity digests (mirror variants live in the local
index), which keeps them at roughly 34 bytes per sampled frame - about
120 KB per hour of footage at 1 fps.

Manifests are untrusted input: reads validate magic, version, section
digests, counts, and sizes, and fail closed with ManifestError.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import msgpack
import zstandard

import overlap
from overlap.errors import ManifestError
from overlap.hashing.base import HASH_BYTES
from overlap.ingest.merkle import merkle_root, sha256_file

if TYPE_CHECKING:
    from pathlib import Path

    from overlap.store.catalog import Catalog

MAGIC = b"OVLM"
FORMAT_VERSION = 1
SCHEMA_VERSION = 1
_MAX_HEADER = 8 << 20
_MAX_SECTION = 1 << 31


@dataclass
class ManifestFile:
    relpath: str
    size: int
    sha256: bytes
    container: str


@dataclass
class ManifestStream:
    file_idx: int
    stream_key: str
    codec: str
    width: int
    height: int
    duration_ms: int
    sample_fps: float
    n_frames: int
    hashes: bytes = b""
    qualities: bytes = b""
    flags: bytes = b""


@dataclass
class Manifest:
    algo_id: str
    prep_id: str
    sample_fps: float
    label: str | None
    merkle_root: bytes
    tool_version: str
    files: list[ManifestFile] = field(default_factory=list)
    streams: list[ManifestStream] = field(default_factory=list)

    @property
    def total_hours(self) -> float:
        return sum(s.duration_ms for s in self.streams) / 3_600_000

    @property
    def total_frames(self) -> int:
        return sum(s.n_frames for s in self.streams)


def export_manifest(
    catalog: Catalog,
    out_path: Path,
    *,
    label: str | None = None,
    anonymize_paths: bool = False,
    stride: int = 1,
    target_fps: float | None = None,
) -> Manifest:
    """Build a manifest from every ``done`` file in the catalog and write it."""
    manifest, _stream_ids = build_manifest(
        catalog,
        label=label,
        anonymize_paths=anonymize_paths,
        stride=stride,
        target_fps=target_fps,
    )
    _write(manifest, out_path)
    return manifest


def build_manifest(
    catalog: Catalog,
    *,
    label: str | None = None,
    anonymize_paths: bool = False,
    stride: int = 1,
    target_fps: float | None = None,
) -> tuple[Manifest, list[int]]:
    """Build an in-memory manifest from the catalog.

    Also returns the catalog stream_id behind each manifest stream (same
    order), which self-comparison needs to exclude trivial self-matches.
    """
    algo_id = catalog.get_meta("algo_id") or ""
    prep_id = catalog.get_meta("prep_id") or ""
    # The density the *index* was built at. The header must report the density
    # the manifest actually carries after striding, not this one: a consumer that
    # trusts the header would otherwise sample against a grid no stream uses,
    # which measured as a true match being reported as 0% overlap.
    index_fps = float(catalog.get_meta("sample_fps") or "1.0")
    if stride < 1:
        raise ManifestError("stride must be >= 1")

    files: list[ManifestFile] = []
    file_idx_by_id: dict[int, int] = {}
    rows = [r for r in catalog.file_rows() if r["status"] == "done"]
    for row in sorted(rows, key=lambda r: str(r["relpath"])):
        relpath = str(row["relpath"])
        sha = bytes(row["sha256"])
        if anonymize_paths:
            ext = relpath.rsplit(".", 1)[-1] if "." in relpath else ""
            relpath = f"f{sha.hex()[:12]}" + (f".{ext}" if ext else "")
        file_idx_by_id[int(row["file_id"])] = len(files)
        files.append(
            ManifestFile(
                relpath=relpath,
                size=int(row["size_bytes"]),
                sha256=sha,
                container=str(row["container"]),
            )
        )

    streams: list[ManifestStream] = []
    stream_ids: list[int] = []
    for srow in catalog.iter_streams():
        if srow.file_id not in file_idx_by_id:
            continue
        stream_ids.append(srow.stream_id)
        hashes, _mirrors, qualities, flags = catalog.stream_hashes(srow.stream_id)
        # Per-stream stride, so a corpus holding streams sampled at different
        # rates still exports a manifest at one consistent density.
        step = stride
        if target_fps:
            step = max(1, int(round(srow.sample_fps / target_fps)))
        if step > 1:
            # Start half a stride in so the exported grid stays centred on the
            # declared timestamps; starting at index 0 shifts every reported
            # corpus timecode by up to most of a stride.
            first = (step - 1) // 2
            keep = list(range(first, srow.n_frames, step))
            hashes = b"".join(hashes[i * HASH_BYTES : (i + 1) * HASH_BYTES] for i in keep)
            qualities = bytes(qualities[i] for i in keep)
            flags = bytes(flags[i] for i in keep)
        n = len(qualities)
        codec, width, height = catalog.stream_codec_dims(srow.stream_id)
        streams.append(
            ManifestStream(
                file_idx=file_idx_by_id[srow.file_id],
                stream_key=srow.stream_key,
                codec=codec,
                width=width,
                height=height,
                duration_ms=srow.duration_ms or 0,
                sample_fps=srow.sample_fps / step,
                n_frames=n,
                hashes=hashes,
                qualities=qualities,
                flags=flags,
            )
        )

    # Report the density the streams actually carry. Where streams differ, the
    # lowest is the honest headline: it is the density a consumer can rely on
    # finding everywhere.
    carried_fps = min((st.sample_fps for st in streams), default=index_fps)
    manifest = Manifest(
        algo_id=algo_id,
        prep_id=prep_id,
        sample_fps=carried_fps,
        label=label,
        merkle_root=merkle_root([(f.relpath, f.sha256) for f in files]),
        tool_version=overlap.__version__,
        files=files,
        streams=streams,
    )
    return manifest, stream_ids


def _write(manifest: Manifest, out_path: Path) -> None:
    comp = zstandard.ZstdCompressor(level=9)

    files_payload = msgpack.packb(
        [[f.relpath, f.size, f.sha256, f.container] for f in manifest.files]
    )
    streams_payload = msgpack.packb(
        [
            [
                s.file_idx,
                s.stream_key,
                s.codec,
                s.width,
                s.height,
                s.duration_ms,
                s.sample_fps,
                s.n_frames,
            ]
            for s in manifest.streams
        ]
    )
    frames_payload = b"".join(s.hashes + s.qualities + s.flags for s in manifest.streams)

    sections = []
    blobs = []
    offset = 0
    for name, payload in (
        ("files.msgpack", files_payload),
        ("streams.msgpack", streams_payload),
        ("frames.bin", frames_payload),
    ):
        blob = comp.compress(payload)
        sections.append(
            {
                "name": name,
                "offset": offset,
                "len": len(blob),
                "raw_len": len(payload),
                "codec": "zstd",
                "sha256": hashlib.sha256(blob).hexdigest(),
            }
        )
        blobs.append(blob)
        offset += len(blob)

    header = {
        "schema": SCHEMA_VERSION,
        "algo_id": manifest.algo_id,
        "prep_id": manifest.prep_id,
        "sample_fps": manifest.sample_fps,
        "label": manifest.label,
        "tool": f"overlap {manifest.tool_version}",
        "dataset": {
            "merkle_root": manifest.merkle_root.hex(),
            "n_files": len(manifest.files),
            "n_streams": len(manifest.streams),
            "total_hours": round(manifest.total_hours, 4),
            "total_frames": manifest.total_frames,
        },
        "sections": sections,
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")

    # Write through a temporary file and rename, so a crash mid-write leaves the
    # previous manifest intact rather than a truncated one. Long catalog runs
    # re-export after every batch, which makes this failure window recurring.
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    with tmp.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<HI", FORMAT_VERSION, len(header_bytes)))
        f.write(header_bytes)
        for blob in blobs:
            f.write(blob)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(out_path)


PARTS_INDEX = "parts.json"
SUFFIX = ".ovlm"


def _join(parts: list[Path], *, max_bytes: int | None = None) -> Manifest:
    """Concatenate parts into one manifest, re-basing each part's file indices."""
    if not parts:
        raise ManifestError("part index lists no parts")
    joined: Manifest | None = None
    for part_path in parts:
        part = read_manifest(part_path, max_bytes=max_bytes)
        if joined is None:
            joined = part
            continue
        for key in ("algo_id", "prep_id"):
            if getattr(part, key) != getattr(joined, key):
                raise ManifestError(
                    f"part {part_path.name} has {key}={getattr(part, key)!r}, "
                    f"the set has {getattr(joined, key)!r}"
                )
        if part.merkle_root != joined.merkle_root:
            raise ManifestError(
                f"part {part_path.name} belongs to a different manifest set"
            )
        base = len(joined.files)
        joined.files.extend(part.files)
        joined.streams.extend(replace(st, file_idx=st.file_idx + base) for st in part.streams)
    assert joined is not None
    return joined


def export_manifest_split(
    catalog: Catalog,
    out_dir: Path,
    *,
    label: str | None = None,
    anonymize_paths: bool = False,
    stride: int = 1,
    target_fps: float | None = None,
    part_bytes: int = 1_500_000_000,
) -> list[Path]:
    """Write a manifest as a directory of parts, each a complete manifest itself.

    A single file stops being practical well before a large archive is covered:
    at 4 fps, 96,000 hours of fingerprints is about 44 GB, past what most hosts
    will serve as one object and impossible to fetch incrementally or resume.

    Parts split on *file* boundaries, never inside a stream, so each part reads
    with the ordinary reader and means something on its own - one can be
    inspected, imported, or re-fetched without the rest. ``parts.json`` lists them
    with their digests, and the Merkle root of the whole set is recorded in every
    part so a subset cannot be passed off as the whole.
    """
    manifest, _ids = build_manifest(
        catalog,
        label=label,
        anonymize_paths=anonymize_paths,
        stride=stride,
        target_fps=target_fps,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    streams_by_file: dict[int, list[ManifestStream]] = {}
    for st in manifest.streams:
        streams_by_file.setdefault(st.file_idx, []).append(st)

    groups: list[list[int]] = []
    current: list[int] = []
    current_bytes = 0
    for file_idx in range(len(manifest.files)):
        weight = sum(
            len(st.hashes) + len(st.qualities) + len(st.flags)
            for st in streams_by_file.get(file_idx, [])
        )
        # A single file larger than the target still gets its own part rather
        # than being split: a stream must stay whole to be readable.
        if current and current_bytes + weight > part_bytes:
            groups.append(current)
            current, current_bytes = [], 0
        current.append(file_idx)
        current_bytes += weight
    if current:
        groups.append(current)

    written: list[Path] = []
    entries: list[dict[str, Any]] = []
    for seq, group in enumerate(groups):
        part = _subset(manifest, group, streams_by_file)
        path = out_dir / f"part-{seq:05d}{SUFFIX}"
        _write(part, path)
        written.append(path)
        entries.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path).hex(),
                "files": len(part.files),
                "streams": len(part.streams),
                "hours": round(part.total_hours, 3),
            }
        )

    index = {
        "schema": "manifest-parts/1",
        "algo_id": manifest.algo_id,
        "prep_id": manifest.prep_id,
        "sample_fps": manifest.sample_fps,
        "label": manifest.label,
        "tool_version": manifest.tool_version,
        "merkle_root": manifest.merkle_root.hex(),
        "files": len(manifest.files),
        "streams": len(manifest.streams),
        "hours": round(manifest.total_hours, 3),
        "parts": entries,
    }
    tmp = out_dir / (PARTS_INDEX + ".part")
    tmp.write_text(json.dumps(index, indent=2), encoding="utf-8")
    tmp.replace(out_dir / PARTS_INDEX)
    return written


def _subset(
    manifest: Manifest,
    file_idxs: list[int],
    streams_by_file: dict[int, list[ManifestStream]],
) -> Manifest:
    """One part: the chosen files, their streams, and the whole set's Merkle root."""
    remap = {old: new for new, old in enumerate(file_idxs)}
    streams: list[ManifestStream] = []
    for old in file_idxs:
        for st in streams_by_file.get(old, []):
            streams.append(replace(st, file_idx=remap[old]))
    return Manifest(
        algo_id=manifest.algo_id,
        prep_id=manifest.prep_id,
        sample_fps=manifest.sample_fps,
        label=manifest.label,
        # Deliberately the *set's* root, not this part's: it is what ties a part
        # to the whole, and `verify` checks delivered bytes against the whole.
        merkle_root=manifest.merkle_root,
        tool_version=manifest.tool_version,
        files=[manifest.files[i] for i in file_idxs],
        streams=streams,
    )


def read_manifest_parts(index_path: Path) -> list[Path]:
    """The part files a parts.json refers to, verified against their digests."""
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if str(index.get("schema", "")) != "manifest-parts/1":
        raise ManifestError(f"{index_path} is not a manifest part index")
    paths = []
    for entry in index["parts"]:
        path = index_path.parent / str(entry["name"])
        if not path.is_file():
            raise ManifestError(f"part listed but missing: {path.name}")
        if sha256_file(path).hex() != entry["sha256"]:
            raise ManifestError(f"part {path.name} does not match its recorded digest")
        paths.append(path)
    return paths


def read_manifest(path: Path, *, max_bytes: int | None = None) -> Manifest:
    """Read and strictly validate a manifest. Untrusted input; fails closed.

    Accepts a single manifest file, a directory of parts, or a parts.json - so a
    caller never has to care how the sender chose to split it.
    """
    if path.is_dir():
        path = path / PARTS_INDEX
        if not path.is_file():
            raise ManifestError(f"no {PARTS_INDEX} in {path.parent}")
    if path.name == PARTS_INDEX:
        return _join(read_manifest_parts(path), max_bytes=max_bytes)
    size = path.stat().st_size
    if max_bytes is not None and size > max_bytes:
        raise ManifestError(f"manifest exceeds size limit ({size} > {max_bytes} bytes)")
    with path.open("rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ManifestError(f"{path} is not an overlap manifest (bad magic)")
        version, header_len = struct.unpack("<HI", f.read(6))
        if version != FORMAT_VERSION:
            raise ManifestError(
                f"manifest format v{version} is not supported by this build "
                f"(need v{FORMAT_VERSION})"
            )
        if header_len > _MAX_HEADER:
            raise ManifestError("manifest header implausibly large")
        try:
            header = json.loads(f.read(header_len).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ManifestError(f"corrupt manifest header: {exc}") from exc
        body = f.read()

    try:
        sections = {s["name"]: s for s in header["sections"]}
        payloads: dict[str, bytes] = {}
        decomp = zstandard.ZstdDecompressor()
        for name in ("files.msgpack", "streams.msgpack", "frames.bin"):
            sec = sections[name]
            lo, ln, raw_len = int(sec["offset"]), int(sec["len"]), int(sec["raw_len"])
            if not (0 <= lo <= lo + ln <= len(body)) or raw_len > _MAX_SECTION:
                raise ManifestError(f"section {name} out of bounds")
            blob = body[lo : lo + ln]
            if hashlib.sha256(blob).hexdigest() != sec["sha256"]:
                raise ManifestError(f"section {name} failed integrity check")
            payloads[name] = decomp.decompress(blob, max_output_size=raw_len)

        files = [
            ManifestFile(relpath=str(r), size=int(sz), sha256=bytes(sha), container=str(c))
            for r, sz, sha, c in msgpack.unpackb(payloads["files.msgpack"], raw=False)
        ]
        manifest = Manifest(
            algo_id=str(header["algo_id"]),
            prep_id=str(header["prep_id"]),
            sample_fps=float(header["sample_fps"]),
            label=header.get("label"),
            merkle_root=bytes.fromhex(header["dataset"]["merkle_root"]),
            tool_version=str(header.get("tool", "")),
            files=files,
        )
        frames = payloads["frames.bin"]
        cursor = 0
        for row in msgpack.unpackb(payloads["streams.msgpack"], raw=False):
            fi, key, codec, w, h, dur, fps, n = row
            fi, n = int(fi), int(n)
            if not (0 <= fi < len(files)) or n < 0:
                raise ManifestError("stream row out of bounds")
            need = n * (HASH_BYTES + 2)
            if cursor + need > len(frames):
                raise ManifestError("frames section truncated")
            hashes = frames[cursor : cursor + n * HASH_BYTES]
            cursor += n * HASH_BYTES
            qualities = frames[cursor : cursor + n]
            cursor += n
            flags = frames[cursor : cursor + n]
            cursor += n
            manifest.streams.append(
                ManifestStream(
                    file_idx=fi,
                    stream_key=str(key),
                    codec=str(codec),
                    width=int(w),
                    height=int(h),
                    duration_ms=int(dur),
                    sample_fps=float(fps),
                    n_frames=n,
                    hashes=hashes,
                    qualities=qualities,
                    flags=flags,
                )
            )
        if cursor != len(frames):
            raise ManifestError("frames section has trailing bytes")
    except ManifestError:
        raise
    except Exception as exc:  # noqa: BLE001 - anything else is a malformed file
        raise ManifestError(f"malformed manifest {path}: {type(exc).__name__}: {exc}") from exc
    return manifest
