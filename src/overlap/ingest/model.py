"""Result types passed from ingestion workers to the catalog writer.

Everything here must be picklable (workers run in spawned processes) and
compact (hash payloads travel as packed bytes, not object lists).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StreamResult:
    stream_key: str
    codec: str
    width: int
    height: int
    native_fps: float | None
    duration_ms: int | None
    sample_fps: float
    algo_id: str
    prep_id: str
    border_crop: str  # "top,bottom,left,right" in px
    n_frames: int
    hashes: bytes  # n_frames * 32 identity digests, concatenated
    mirrors: bytes  # n_frames * 32 mirror digests, concatenated
    qualities: bytes  # n_frames u8
    flags: bytes  # n_frames u8
    sketch: bytes  # 32-byte majority-vote sketch over identity digests
    # Crop-ladder variants (index-side only, never exported in manifests):
    # frame-major, rung-minor: frame0[rung0..rungN] frame1[rung0..rungN] ...
    n_crop_rungs: int = 0
    crop_hashes: bytes = b""  # n_frames * n_crop_rungs * 32
    crop_mirrors: bytes = b""  # n_frames * n_crop_rungs * 32


@dataclass
class FileResult:
    abspath: str
    root: str
    relpath: str
    size_bytes: int
    mtime_ns: int
    sha256: bytes
    container: str
    status: str  # done | error | skipped
    error: str | None = None
    streams: list[StreamResult] = field(default_factory=list)
