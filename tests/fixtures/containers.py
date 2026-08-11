"""Fixture writers: wrap a video clip's frames into robotics containers.

These simulate cross-format laundering - the same footage delivered as an
MCAP camera topic or a ROS1 bag instead of an .mp4. Frames are jpeg-encoded
CompressedImage messages, exactly what real recorders produce.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import av
import cv2

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

ROS2_COMPRESSED_IMAGE_MSGDEF = """\
std_msgs/Header header
string format
uint8[] data
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""


def iter_video_frames(src: Path, *, fps: float = 12.0) -> Iterator[tuple[float, bytes]]:
    """(t_seconds, jpeg_bytes) samples from a video at the given rate."""
    with av.open(str(src)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        interval = 1.0 / fps
        next_t = 0.0
        for frame in container.decode(stream):
            t = frame.time or 0.0
            if t + 1e-9 < next_t:
                continue
            ok, jpeg = cv2.imencode(
                ".jpg", frame.to_ndarray(format="bgr24"), [cv2.IMWRITE_JPEG_QUALITY, 90]
            )
            assert ok
            yield t, jpeg.tobytes()
            next_t += interval


def video_to_mcap(
    src: Path, dst: Path, *, topic: str = "/cam_front/image/compressed", fps: float = 12.0
) -> Path:
    from mcap_ros2.writer import Writer as McapWriter

    base_ns = 1_700_000_000 * 1_000_000_000  # arbitrary wall-clock epoch
    with dst.open("wb") as f:
        writer = McapWriter(f)
        schema = writer.register_msgdef(
            "sensor_msgs/msg/CompressedImage", ROS2_COMPRESSED_IMAGE_MSGDEF
        )
        for t, jpeg in iter_video_frames(src, fps=fps):
            t_ns = base_ns + int(t * 1e9)
            writer.write_message(
                topic=topic,
                schema=schema,
                message={
                    "header": {
                        "stamp": {"sec": t_ns // 10**9, "nanosec": t_ns % 10**9},
                        "frame_id": "cam_front",
                    },
                    "format": "jpeg",
                    "data": jpeg,
                },
                log_time=t_ns,
                publish_time=t_ns,
            )
        writer.finish()
    return dst


def video_to_rosbag1(
    src: Path, dst: Path, *, topic: str = "/cam_front/image/compressed", fps: float = 12.0
) -> Path:
    import numpy as np
    from rosbags.rosbag1 import Writer as Bag1Writer
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS1_NOETIC)
    CompressedImage = typestore.types["sensor_msgs/msg/CompressedImage"]
    Header = typestore.types["std_msgs/msg/Header"]
    Time = typestore.types["builtin_interfaces/msg/Time"]
    msgtype = "sensor_msgs/msg/CompressedImage"

    base_ns = 1_700_000_000 * 1_000_000_000
    with Bag1Writer(dst) as writer:
        conn = writer.add_connection(topic, msgtype, typestore=typestore)
        for seq, (t, jpeg) in enumerate(iter_video_frames(src, fps=fps)):
            t_ns = base_ns + int(t * 1e9)
            msg = CompressedImage(
                header=Header(
                    seq=seq,
                    stamp=Time(sec=t_ns // 10**9, nanosec=t_ns % 10**9),
                    frame_id="cam_front",
                ),
                format="jpeg",
                data=np.frombuffer(jpeg, dtype=np.uint8),
            )
            writer.write(conn, t_ns, typestore.serialize_ros1(msg, msgtype))
    return dst
