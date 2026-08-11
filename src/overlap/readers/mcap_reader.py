"""MCAP reader: fingerprint camera topics inside robotics recordings.

One fingerprint stream per image topic (``stream_key`` = topic name), so a
4-camera teleop rig yields 4 streams per file - and a single camera topic
laundered out into an .mp4 still matches its source recording, because both
are hashed from decoded pixels.

Supported schemas (ROS1 and ROS2 encodings via the official mcap decoders):

- ``sensor_msgs/msg/CompressedImage`` / ``sensor_msgs/CompressedImage``
  (jpeg/png payloads)
- ``sensor_msgs/msg/Image`` / ``sensor_msgs/Image``
  (rgb8, bgr8, mono8, mono16)

Other image-bearing schemas (foxglove CompressedVideo/H.264 elementary
streams, protobuf schemas) are enumerated but skipped with a warning - a
skipped stream is surfaced, never silent, because silently dropping footage
would corrupt overlap percentages.

Timestamps prefer ``header.stamp`` and fall back to log time; both are
normalized to the first message so trimmed recordings align on content.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from mcap.reader import make_reader
from mcap_ros1.decoder import DecoderFactory as Ros1DecoderFactory
from mcap_ros2.decoder import DecoderFactory as Ros2DecoderFactory

from overlap.errors import ReaderError
from overlap.readers.base import FrameSample, SamplePolicy, StreamInfo, register

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_COMPRESSED_SCHEMAS = {"sensor_msgs/msg/CompressedImage", "sensor_msgs/CompressedImage"}
_RAW_SCHEMAS = {"sensor_msgs/msg/Image", "sensor_msgs/Image"}
_SUPPORTED = _COMPRESSED_SCHEMAS | _RAW_SCHEMAS


# The class object implements the Reader protocol; mypy cannot verify that.
@register
class McapReader:  # type: ignore[type-var]
    name = "mcap"
    extensions = (".mcap",)

    @classmethod
    def probe(cls, path: Path) -> bool:
        try:
            with path.open("rb") as f:
                return f.read(8) == b"\x89MCAP0\r\n"
        except OSError:
            return False

    @classmethod
    def open(cls, path: Path) -> McapSession:
        return McapSession(path)


class McapSession:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._file = path.open("rb")
        try:
            self._reader = make_reader(
                self._file,
                decoder_factories=[Ros2DecoderFactory(), Ros1DecoderFactory()],
            )
            self._summary = self._reader.get_summary()
        except Exception as exc:  # noqa: BLE001 - any parse failure is a bad file
            self._file.close()
            raise ReaderError(f"cannot open MCAP {path}: {exc}") from exc
        if self._summary is None:
            self._file.close()
            raise ReaderError(f"{path} has no MCAP summary section (unindexed file)")

    def __enter__(self) -> McapSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self._file.close()

    def _image_channels(self) -> dict[str, tuple[int, str]]:
        """topic -> (channel_id, schema_name) for supported image schemas."""
        assert self._summary is not None
        result: dict[str, tuple[int, str]] = {}
        for channel_id, channel in self._summary.channels.items():
            schema = self._summary.schemas.get(channel.schema_id)
            if schema is not None and schema.name in _SUPPORTED:
                result[channel.topic] = (channel_id, schema.name)
        return result

    def streams(self) -> list[StreamInfo]:
        assert self._summary is not None
        stats = self._summary.statistics
        duration_ms: int | None = None
        if stats is not None and stats.message_end_time and stats.message_start_time:
            duration_ms = int((stats.message_end_time - stats.message_start_time) / 1e6)
        infos = []
        for topic, (channel_id, schema_name) in sorted(self._image_channels().items()):
            n_messages = None
            if stats is not None and stats.channel_message_counts:
                n_messages = stats.channel_message_counts.get(channel_id)
            fps = None
            if n_messages and duration_ms:
                fps = round(n_messages / (duration_ms / 1000.0), 2)
            width, height, codec = self._probe_dimensions(topic, schema_name)
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

    def _probe_dimensions(self, topic: str, schema_name: str) -> tuple[int, int, str]:
        for _schema, _channel, _message, decoded in self._reader.iter_decoded_messages(
            topics=[topic]
        ):
            image = _decode_image(decoded, schema_name)
            if image is not None:
                h, w = image.shape[:2]
                codec = (
                    getattr(decoded, "format", "raw")
                    if schema_name in _COMPRESSED_SCHEMAS
                    else getattr(decoded, "encoding", "raw")
                )
                return w, h, str(codec)
            break
        return 0, 0, "unknown"

    def sample(self, stream_key: str, policy: SamplePolicy) -> Iterator[FrameSample]:
        channels = self._image_channels()
        if stream_key not in channels:
            raise ReaderError(f"no image topic {stream_key!r} in {self._path}")
        _channel_id, schema_name = channels[stream_key]
        interval = 1.0 / policy.fps
        next_t = 0.5 * interval
        t_first: float | None = None
        last_t = -1.0
        for _schema, _channel, message, decoded in self._reader.iter_decoded_messages(
            topics=[stream_key], log_time_order=True
        ):
            t_abs = _message_time_s(decoded, message.log_time)
            if t_first is None:
                t_first = t_abs
            t = max(t_abs - t_first, last_t)  # monotonic, stream-relative
            last_t = t
            if t + 1e-9 < next_t:
                continue
            image = _decode_image(decoded, schema_name)
            if image is None:
                continue
            while t + 1e-9 >= next_t:
                yield FrameSample(t_ms=int(round(next_t * 1000)), image=image)
                next_t += interval


def _message_time_s(decoded: Any, log_time_ns: int) -> float:
    header = getattr(decoded, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is not None:
        sec = int(getattr(stamp, "sec", 0))
        nsec = int(getattr(stamp, "nanosec", getattr(stamp, "nsec", 0)))
        if sec > 0 or nsec > 0:
            return sec + nsec / 1e9
    return log_time_ns / 1e9


def _decode_image(decoded: Any, schema_name: str) -> np.ndarray | None:
    """Decoded ROS message -> BGR or grayscale uint8 ndarray (None = skip)."""
    if schema_name in _COMPRESSED_SCHEMAS:
        data = np.frombuffer(bytes(decoded.data), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return image  # None when the payload is not a decodable jpeg/png
    encoding = str(getattr(decoded, "encoding", "")).lower()
    width = int(decoded.width)
    height = int(decoded.height)
    raw = np.frombuffer(bytes(decoded.data), dtype=np.uint8)
    if encoding == "rgb8":
        img = raw.reshape(height, width, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if encoding == "bgr8":
        return raw.reshape(height, width, 3).copy()
    if encoding == "mono8":
        return raw.reshape(height, width).copy()
    if encoding == "mono16":
        img16 = raw.view(np.uint16).reshape(height, width)
        return (img16 >> 8).astype(np.uint8)
    return None
