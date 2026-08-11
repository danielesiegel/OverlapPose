"""Video container reader (mp4, mkv, avi, mov, webm) via PyAV.

Sampling is sequential-decode-with-skip: every packet is decoded (h264/h265
inter-frames require it anyway), but only frames that cross the sample grid
are converted to ndarrays and handed on - conversion, prep, and hashing are
the per-frame costs worth skipping. Variable-frame-rate input is handled by
trusting presentation timestamps; frames covering several grid slots are
emitted once per slot so the grid stays regular for the matcher.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import av

from overlap.errors import ReaderError
from overlap.readers.base import FrameSample, SamplePolicy, StreamInfo, register

# PyAV renamed its base exception: `av.AVError` was an alias for the FFmpeg error
# hierarchy and was removed in PyAV 12. Catching the missing name turned every
# unreadable file into an AttributeError from inside the except clause, which
# lost the real cause and bypassed the reader's own error handling - a truncated
# download reported "module 'av' has no attribute 'AVError'" instead of being
# skipped and surfaced. Resolve it once, at import.
_AV_ERROR: type[Exception] = getattr(av, "FFmpegError", None) or getattr(av, "AVError", OSError)  # type: ignore[assignment]

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


# The class object itself implements the Reader protocol (classmethods bind
# to the protocol's method signatures); mypy cannot verify that pattern yet.
@register
class VideoFileReader:  # type: ignore[type-var]
    name = "video"
    extensions = (".mp4", ".mkv", ".avi", ".mov", ".webm")

    @classmethod
    def probe(cls, path: Path) -> bool:
        return path.suffix.lower() in cls.extensions

    @classmethod
    def open(cls, path: Path) -> VideoSession:
        return VideoSession(path)


class VideoSession:
    def __init__(self, path: Path) -> None:
        self._path = path
        try:
            self._container = av.open(str(path))
        except _AV_ERROR as exc:
            raise ReaderError(f"cannot open {path}: {exc}") from exc

    def __enter__(self) -> VideoSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self._container.close()

    def streams(self) -> list[StreamInfo]:
        infos = []
        for i, stream in enumerate(self._container.streams.video):
            duration_ms: int | None = None
            if stream.duration is not None and stream.time_base is not None:
                duration_ms = int(stream.duration * stream.time_base * 1000)
            elif self._container.duration is not None:
                duration_ms = int(self._container.duration / av.time_base * 1000)
            fps = float(stream.average_rate) if stream.average_rate else None
            infos.append(
                StreamInfo(
                    stream_key=f"v{i}",
                    codec=stream.codec_context.name,
                    width=stream.codec_context.width,
                    height=stream.codec_context.height,
                    native_fps=fps,
                    duration_ms=duration_ms,
                )
            )
        return infos

    def sample(self, stream_key: str, policy: SamplePolicy) -> Iterator[FrameSample]:
        if not stream_key.startswith("v"):
            raise ReaderError(f"unknown stream key {stream_key!r} for video container")
        index = int(stream_key[1:])
        videos = self._container.streams.video
        if index >= len(videos):
            raise ReaderError(f"no video stream {stream_key!r} in {self._path}")
        stream = videos[index]
        stream.thread_type = "AUTO"  # frame+slice parallel decode

        self._container.seek(0)
        interval = 1.0 / policy.fps
        next_t = 0.5 * interval
        fallback_fps = float(stream.average_rate) if stream.average_rate else 30.0
        n_seen = 0
        last_t = -1.0
        try:
            for frame in self._container.decode(stream):
                t = frame.time
                if t is None:
                    t = n_seen / fallback_fps
                if t < last_t:  # non-monotonic pts: clamp forward, never rewind
                    t = last_t
                last_t = t
                n_seen += 1
                if t + 1e-9 < next_t:
                    continue
                image = frame.to_ndarray(format="bgr24")
                while t + 1e-9 >= next_t:
                    yield FrameSample(t_ms=int(round((next_t) * 1000)), image=image)
                    next_t += interval
        except _AV_ERROR as exc:
            raise ReaderError(f"decode failed in {self._path} ({stream_key}): {exc}") from exc
