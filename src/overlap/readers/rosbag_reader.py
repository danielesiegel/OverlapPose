"""ROS bag reader: ROS1 ``.bag`` and rosbag2 via the ``rosbags`` library.

No ROS installation is required - ``rosbags`` ships its own type system and
deserializers. Installed via the ``overlap-cli[ros]`` extra; without it, bag
files are reported as skipped with an actionable message.

rosbag2 note: a ``.db3`` file is one storage shard of a rosbag2 recording;
the recording is the *directory* containing ``metadata.yaml``. When pointed
at a bare ``.db3`` we open its parent directory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from overlap.errors import ReaderError
from overlap.readers.base import FrameSample, SamplePolicy, StreamInfo, register

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_COMPRESSED = "sensor_msgs/msg/CompressedImage"
_RAW = "sensor_msgs/msg/Image"


# The class object implements the Reader protocol; mypy cannot verify that.
@register
class RosbagReader:  # type: ignore[type-var]
    name = "rosbag"
    extensions = (".bag", ".db3")

    @classmethod
    def probe(cls, path: Path) -> bool:
        return path.suffix.lower() in cls.extensions

    @classmethod
    def open(cls, path: Path) -> RosbagSession:
        return RosbagSession(path)


class RosbagSession:
    def __init__(self, path: Path) -> None:
        try:
            from rosbags.highlevel import AnyReader
        except ImportError as exc:
            raise ReaderError(
                f"cannot read {path.name}: ROS bag support is not installed "
                f"(pip install 'overlap-cli[ros]')"
            ) from exc

        target = path
        if path.suffix.lower() == ".db3":
            if not (path.parent / "metadata.yaml").exists():
                raise ReaderError(
                    f"{path} is a rosbag2 shard without metadata.yaml next to it; "
                    f"point overlap at the recording directory"
                )
            target = path.parent
        self._path = path
        try:
            self._reader = AnyReader([target])
            self._reader.open()
        except Exception as exc:  # noqa: BLE001 - any parse failure is a bad file
            raise ReaderError(f"cannot open bag {path}: {exc}") from exc

    def __enter__(self) -> RosbagSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self._reader.close()

    def _image_connections(self) -> dict[str, list[Any]]:
        by_topic: dict[str, list[Any]] = {}
        for conn in self._reader.connections:
            if conn.msgtype in (_COMPRESSED, _RAW):
                by_topic.setdefault(conn.topic, []).append(conn)
        return by_topic

    def streams(self) -> list[StreamInfo]:
        duration_ms = int((self._reader.end_time - self._reader.start_time) / 1e6)
        infos = []
        for topic, conns in sorted(self._image_connections().items()):
            n_messages = sum(c.msgcount for c in conns)
            fps = round(n_messages / (duration_ms / 1000.0), 2) if duration_ms else None
            width, height, codec = self._probe_dimensions(conns)
            infos.append(
                StreamInfo(
                    stream_key=topic,
                    codec=codec,
                    width=width,
                    height=height,
                    native_fps=fps,
                    duration_ms=duration_ms,
                    n_messages=n_messages,
                )
            )
        return infos

    def _probe_dimensions(self, conns: list[Any]) -> tuple[int, int, str]:
        for conn, _timestamp, rawdata in self._reader.messages(connections=conns):
            msg = self._reader.deserialize(rawdata, conn.msgtype)
            image = _decode_image(msg, conn.msgtype)
            if image is not None:
                h, w = image.shape[:2]
                codec = getattr(msg, "format", None) or getattr(msg, "encoding", "raw")
                return w, h, str(codec)
            break
        return 0, 0, "unknown"

    def sample(self, stream_key: str, policy: SamplePolicy) -> Iterator[FrameSample]:
        conns = self._image_connections().get(stream_key)
        if not conns:
            raise ReaderError(f"no image topic {stream_key!r} in {self._path}")
        interval = 1.0 / policy.fps
        next_t = 0.5 * interval
        t_first: float | None = None
        last_t = -1.0
        for conn, timestamp, rawdata in self._reader.messages(connections=conns):
            t_abs = timestamp / 1e9
            if t_first is None:
                t_first = t_abs
            t = max(t_abs - t_first, last_t)
            last_t = t
            if t + 1e-9 < next_t:
                continue
            msg = self._reader.deserialize(rawdata, conn.msgtype)
            image = _decode_image(msg, conn.msgtype)
            if image is None:
                continue
            while t + 1e-9 >= next_t:
                yield FrameSample(t_ms=int(round(next_t * 1000)), image=image)
                next_t += interval


def _decode_image(msg: Any, msgtype: str) -> np.ndarray | None:
    if msgtype == _COMPRESSED:
        data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    encoding = str(getattr(msg, "encoding", "")).lower()
    width, height = int(msg.width), int(msg.height)
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if encoding == "rgb8":
        return cv2.cvtColor(raw.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
    if encoding == "bgr8":
        return raw.reshape(height, width, 3).copy()
    if encoding == "mono8":
        return raw.reshape(height, width).copy()
    if encoding == "mono16":
        return (raw.view(np.uint16).reshape(height, width) >> 8).astype(np.uint8)
    return None
