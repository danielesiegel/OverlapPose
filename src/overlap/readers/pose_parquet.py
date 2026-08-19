"""Parquet reader: fingerprint dense proprioceptive streams (pose / IMU / joints).

Handles Parquet files shaped like one row per sample with a time column and
dense numeric channel columns - the common export shape for pose trajectories,
joint embeddings and IMU logs (e.g. the landmark-sim / LeRobot-style layout:
``time_ms``/``time_us`` plus ``a00_pelvis_x`` ... channel columns).

The whole dense channel matrix becomes ONE signal stream (``stream_key`` =
``"proprio"``): channels in sorted column order, so the channel set and its
ordering are part of fingerprint identity the same way an image's geometry is.
Sparse columns (mostly-NaN, e.g. 5 Hz UWB fixes inside a 1 kHz log) are
excluded; text/label columns are metadata, not evidence, and are ignored.

Prep (``sp1``, applied here because it needs whole-stream statistics, exactly
like the video path's per-stream border crop): per-channel robust scaling
``(x - median) / IQR``, residual NaNs linearly interpolated. Scaling makes the
fingerprint invariant to unit changes and calibration offsets by construction.

``sample()`` yields one window per grid tick ``t = (k + 0.5) / fps``: a
C x T float array covering ``WINDOW_S`` seconds centred on the tick (clamped
at the ends), which :class:`~overlap.hashing.sdq.SdqKernel` hashes.

pyarrow is an optional dependency (``pip install overlap-cli[pose]``); the
reader registers regardless and reports a clear error if it is missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from overlap.errors import ReaderError
from overlap.hashing.sdq import WINDOW_S
from overlap.readers.base import FrameSample, SamplePolicy, StreamInfo, register

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

STREAM_KEY = "proprio"
_TIME_COLUMNS = ("time_us", "time_ms", "timestamp_us", "timestamp_ms")
_MIN_CHANNELS = 8
_MIN_SAMPLES = 64
# A column with more NaN than this is a sparse side-channel, not a dense stream.
_MAX_NAN_FRAC = 0.10


@register
class PoseParquetReader:  # type: ignore[type-var]
    name = "pose-parquet"
    extensions = (".parquet",)

    @classmethod
    def probe(cls, path: Path) -> bool:
        try:
            with path.open("rb") as f:
                if f.read(4) != b"PAR1":
                    return False
        except OSError:
            return False
        try:
            import pyarrow.parquet as pq
        except ImportError:
            return True  # let open() report the missing-dependency hint
        try:
            names = pq.read_schema(path).names
        except Exception:  # noqa: BLE001 - unreadable parquet is not ours
            return False
        # Only claim time-series-shaped parquets; other parquet artifacts in a
        # dataset directory are skipped as unsupported, not errored.
        return any(c in names for c in _TIME_COLUMNS)

    @classmethod
    def open(cls, path: Path) -> PoseParquetSession:
        return PoseParquetSession(path)


class PoseParquetSession:
    def __init__(self, path: Path) -> None:
        self._path = path
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ReaderError(
                f"{path}: reading Parquet needs pyarrow - install overlap-cli[pose]"
            ) from exc
        try:
            table = pq.read_table(path)
        except Exception as exc:  # noqa: BLE001 - any parse failure is a bad file
            raise ReaderError(f"cannot open Parquet {path}: {exc}") from exc

        time_col = next((c for c in _TIME_COLUMNS if c in table.column_names), None)
        if time_col is None:
            raise ReaderError(
                f"{path} has no time column (looked for {', '.join(_TIME_COLUMNS)})"
            )
        t = table.column(time_col).to_numpy(zero_copy_only=False).astype(np.float64)
        self._t_s = t / (1e6 if time_col.endswith("us") else 1e3)

        import pyarrow.types as pat

        numeric = sorted(
            c
            for c, f in zip(table.column_names, table.schema, strict=True)
            if c != time_col and (pat.is_floating(f.type) or pat.is_integer(f.type))
        )
        dense: list[str] = []
        cols: list[np.ndarray] = []
        for c in numeric:
            v = table.column(c).to_numpy(zero_copy_only=False).astype(np.float64)
            if np.isnan(v).mean() <= _MAX_NAN_FRAC:
                dense.append(c)
                cols.append(v)
        if len(dense) < _MIN_CHANNELS or len(self._t_s) < _MIN_SAMPLES:
            raise ReaderError(
                f"{path} has {len(dense)} dense numeric channels x {len(self._t_s)} "
                f"samples; need at least {_MIN_CHANNELS} x {_MIN_SAMPLES}"
            )
        x = np.stack(cols, axis=0)  # C x N, sorted column order

        # sp1 prep: interpolate residual NaNs, then per-channel robust scale.
        for row in x:
            bad = np.isnan(row)
            if bad.any():
                idx = np.arange(len(row))
                row[bad] = np.interp(idx[bad], idx[~bad], row[~bad])
        med = np.median(x, axis=1, keepdims=True)
        iqr = np.percentile(x, 75, axis=1, keepdims=True) - np.percentile(
            x, 25, axis=1, keepdims=True
        )
        self._x = (x - med) / np.maximum(iqr, 1e-9)
        self._channels = dense

        dt = np.diff(self._t_s)
        dt = dt[dt > 0]
        self._rate_hz = float(1.0 / np.median(dt)) if len(dt) else 0.0
        self._duration_s = float(self._t_s[-1] - self._t_s[0]) if len(self._t_s) else 0.0
        if self._rate_hz <= 0 or self._duration_s <= 0:
            raise ReaderError(f"{path} has no usable time axis")

    def __enter__(self) -> PoseParquetSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self._x = np.empty((0, 0))

    def streams(self) -> list[StreamInfo]:
        c, n = self._x.shape
        return [
            StreamInfo(
                stream_key=STREAM_KEY,
                codec=f"f64[{c}ch@{self._rate_hz:g}Hz]",
                width=c,
                height=int(round(self._rate_hz)),
                native_fps=round(self._rate_hz, 2),
                duration_ms=int(round(self._duration_s * 1000)),
                n_messages=n,
                modality="signal",
            )
        ]

    def sample(self, stream_key: str, policy: SamplePolicy) -> Iterator[FrameSample]:
        if stream_key != STREAM_KEY:
            raise ReaderError(f"no signal stream {stream_key!r} in {self._path}")
        n = self._x.shape[1]
        half = WINDOW_S / 2.0
        interval = 1.0 / policy.fps
        t0 = float(self._t_s[0])
        k = 0
        while True:
            centre = (k + 0.5) * interval
            if centre > self._duration_s:
                break
            # Clamped window in sample indices; the kernel resamples to a fixed
            # length, so edge windows stay hashable (same as video's edge frames
            # being ordinary frames).
            lo = np.searchsorted(self._t_s, t0 + centre - half, side="left")
            hi = np.searchsorted(self._t_s, t0 + centre + half, side="right")
            window = self._x[:, max(0, lo) : min(n, max(hi, lo + 8))]
            if window.shape[1] >= 8:
                yield FrameSample(t_ms=int(round(centre * 1000)), image=window)
            k += 1
