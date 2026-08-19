"""Container readers. Importing this package registers the built-in readers."""

from overlap.readers import mcap_reader, pose_parquet, rosbag_reader, video  # noqa: F401
from overlap.readers.base import (
    FrameSample,
    Reader,
    ReaderSession,
    SamplePolicy,
    StreamInfo,
    all_readers,
    reader_for,
    register,
    supported_extensions,
)

__all__ = [
    "FrameSample",
    "Reader",
    "ReaderSession",
    "SamplePolicy",
    "StreamInfo",
    "all_readers",
    "reader_for",
    "register",
    "supported_extensions",
]
