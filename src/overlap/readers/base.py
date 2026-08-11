"""Reader plugin interface - one reader per container format.

A reader turns any supported file into one or more *streams* of sampled,
decoded frames with timestamps. Everything downstream is format-agnostic:
fingerprints are computed on decoded pixels, which is what makes cross-format
copies (an MCAP camera topic re-delivered as .mp4) comparable.

Third-party formats plug in via the ``overlap.readers`` entry-point group
without touching this package.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np


@dataclass(frozen=True)
class StreamInfo:
    """One visual stream within a container (a video track or a camera topic)."""

    stream_key: str  # "v0" for video tracks; topic name for robotics containers
    codec: str
    width: int
    height: int
    native_fps: float | None
    duration_ms: int | None
    n_messages: int | None = None  # message count where the container knows it


class FrameSample(NamedTuple):
    t_ms: int  # stream-relative presentation time
    image: np.ndarray  # HxW uint8 gray or HxWx3 uint8 BGR


@dataclass(frozen=True)
class SamplePolicy:
    """How to sample frames from a stream.

    Frames are taken on the fixed grid ``t = (k + 0.5) / fps`` of the stream's
    presented timeline. The grid (not scene-adaptive sampling) is what lets
    the matcher reason about time linearly - speed changes become slopes.
    """

    fps: float = 1.0


class ReaderSession(Protocol):
    """An opened file. Context manager; ``sample`` may be called per stream."""

    def __enter__(self) -> ReaderSession: ...

    def __exit__(self, *exc: object) -> None: ...

    def streams(self) -> list[StreamInfo]: ...

    def sample(self, stream_key: str, policy: SamplePolicy) -> Iterator[FrameSample]: ...


@runtime_checkable
class Reader(Protocol):
    """The reader interface. Implementations are class objects with
    classmethods (see VideoFileReader); the class object itself satisfies
    this protocol structurally."""

    name: str
    extensions: tuple[str, ...]

    def probe(self, path: Path) -> bool:
        """Cheap check (extension / magic bytes) whether this reader handles path."""
        ...

    def open(self, path: Path) -> ReaderSession: ...


R = TypeVar("R", bound=Reader)


_BUILTINS: list[Reader] = []
_loaded_entry_points = False


def register(reader: R) -> R:
    """Register a Reader class (also usable as a decorator in plugins)."""
    if reader not in _BUILTINS:
        _BUILTINS.append(reader)
    return reader


def _ensure_plugins() -> None:
    global _loaded_entry_points
    if _loaded_entry_points:
        return
    _loaded_entry_points = True
    for ep in entry_points(group="overlap.readers"):
        try:
            register(ep.load())
        except Exception:  # noqa: BLE001 - a broken plugin must not kill the CLI
            continue


def all_readers() -> list[Reader]:
    _ensure_plugins()
    return list(_BUILTINS)


def reader_for(path: Path) -> Reader | None:
    """Find the reader that handles ``path``, or None if unsupported."""
    suffix = path.suffix.lower()
    for reader in all_readers():
        if suffix in reader.extensions and reader.probe(path):
            return reader
    return None


def supported_extensions() -> set[str]:
    return {ext for reader in all_readers() for ext in reader.extensions}
