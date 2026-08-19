"""sdq1 - the signal-domain hash kernel behind OverlapPose.

``pdq2`` fingerprints decoded pixels; ``sdq1`` applies the same recipe to
windows of multichannel proprioceptive signal (joint positions, IMU channels,
pose trajectories): normalize, project onto a small low-frequency orthonormal
basis, threshold every coefficient at the slab median, compare by Hamming
distance. The recipe - not the pixels - is what makes PDQ robust, and it is
modality-agnostic.

One window = ``WINDOW_S`` seconds of all channels, taken on the same fixed
``(k + 0.5) / fps`` grid the video path samples frames on, so the matcher's
segment/slope machinery (trims, splices, concatenation) works unchanged.

Pipeline per window (C channels x T native samples):

1. time axis resampled to a fixed 256 samples (``cv2.INTER_AREA``) - the
   analog of PDQ squashing every image to a fixed square, which makes the
   hash independent of the native sample rate,
2. 2-D orthonormal DCT: channel frequencies 0..15 x time frequencies 1..16
   (time-DC excluded, so constant per-channel offsets vanish by construction),
3. the 16x16 slab is thresholded at its median -> 256 bits.

Median-thresholding is what buys the Gaussian-noise robustness: white noise
spreads its energy over all C*256 dimensions while the slab keeps 256 of
them, and a bit only flips when the noise perturbation of its coefficient
exceeds that coefficient's distance from the median. Measured on a real
132-channel 1 kHz IMU stream: noise at 5% of per-channel signal sd flips
4 bits of 256 on average; unrelated windows sit at 128 +/- 10.

The mirror slot carries the *time-reversal* digest: reversing a window in
time multiplies time-DCT coefficient ``i`` by ``(-1)**i``, so the reversed
digest comes from the same DCT at no extra cost - exactly how pdq2 derives
its horizontal-mirror digest. A copy played backwards therefore lands on the
mirror entry and is reported through the same flip machinery.

Not covered (documented, not hidden): resampled/speed-changed signal windows
shift their DCT frequencies and land near the unrelated floor - the signal
analog of a deep crop. Detecting them needs a resample ladder, the analog of
the image crop ladder; see the README.
"""

from __future__ import annotations

import cv2
import numpy as np

from overlap.hashing.base import FLAG_LOW_QUALITY, FrameHash
from overlap.hashing.pdq_numpy import pack_bits

SIGNAL_ALGO_ID = "sdq1"
# sp1: per-channel robust scaling (median/IQR over the whole stream, computed
# once by the reader - the signal analog of the per-stream border crop),
# channels in sorted column order, NaN-dense columns dropped, 1 s windows.
SIGNAL_PREP_ID = "sp1"

WINDOW_S = 1.0  # window length; part of prep identity
_TIME_RES = 256  # fixed time resolution after resampling
_K = 16  # slab is _K x _K -> 256 bits

# Windows below this quality (near-idle signal) are hashed and stored but
# flagged and excluded from ANN candidate generation, for the same reason
# blank video frames are: near-featureless windows collide and poison matching.
QUALITY_FLOOR = 20

_DCT_TIME: np.ndarray | None = None
_DCT_CHAN_CACHE: dict[int, np.ndarray] = {}


def _dct_rows(freqs: range, n: int) -> np.ndarray:
    """Orthonormal DCT-II rows for the given frequencies over n points."""
    i = np.arange(freqs.start, freqs.stop, dtype=np.float64)[:, None]
    x = np.arange(n, dtype=np.float64)[None, :]
    d: np.ndarray = np.sqrt(2.0 / n) * np.cos((np.pi / 2.0 / n) * i * (2.0 * x + 1.0))
    if freqs.start == 0:
        d[0] = np.sqrt(1.0 / n)
    return d


def _dct_time() -> np.ndarray:
    global _DCT_TIME
    if _DCT_TIME is None:
        _DCT_TIME = _dct_rows(range(1, _K + 1), _TIME_RES)  # time-DC excluded
    return _DCT_TIME


def _dct_chan(c: int) -> np.ndarray:
    cached = _DCT_CHAN_CACHE.get(c)
    if cached is None:
        cached = _dct_rows(range(0, _K), c)  # channel-DC kept: cross-channel mean matters
        _DCT_CHAN_CACHE[c] = cached
    return cached


# Time reversal multiplies time-DCT coefficient i by (-1)**i (freqs 1.._K).
_REVERSAL_SIGNS = np.where(np.arange(1, _K + 1) % 2 == 0, 1.0, -1.0)[None, :]


def _bits(slab: np.ndarray) -> np.ndarray:
    flat = slab.reshape(-1)
    result: np.ndarray = flat > np.median(flat)
    return result


class SdqKernel:
    """Signal-window :class:`~overlap.hashing.base.HashKernel` implementation.

    ``hash_frame`` takes a C x T float window (channels x native time samples,
    already sp1-normalized by the reader) instead of a grayscale image; the
    protocol shape - ndarray in, :class:`FrameHash` out - is identical.
    """

    algo_id = SIGNAL_ALGO_ID
    hash_bits = 256

    def __init__(self, quality_floor: int = QUALITY_FLOOR) -> None:
        self.quality_floor = quality_floor

    def hash_frame(self, window: np.ndarray) -> FrameHash:
        if window.ndim != 2:
            raise ValueError(f"expected 2-D channel x time window, got shape {window.shape}")
        c, t = window.shape
        if c < 2 or t < 8:
            raise ValueError(f"window too small to hash: {window.shape}")
        w = np.ascontiguousarray(window, dtype=np.float32)
        # dsize is (width, height): resample the time axis, keep every channel.
        w = cv2.resize(w, (_TIME_RES, c), interpolation=cv2.INTER_AREA).astype(np.float64)
        slab = _dct_chan(c) @ (w @ _dct_time().T)  # _K x _K
        bits_id = _bits(slab)
        bits_rev = _bits(slab * _REVERSAL_SIGNS)
        # sp1-normalized units: rms 1.0 = signal moving about one IQR. Idle
        # stretches (a robot holding still) score near 0 and get flagged.
        rms = float(np.sqrt(np.mean(window * window)))
        quality = min(100, int(round(rms * 50)))
        flags = 0
        if quality < self.quality_floor:
            flags |= FLAG_LOW_QUALITY
        return FrameHash(
            hash=pack_bits(bits_id),
            mirror=pack_bits(bits_rev),
            quality=quality,
            flags=flags,
        )
