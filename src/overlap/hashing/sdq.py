"""sdq1 - the signal-domain hash kernel behind OverlapPose.

``pdq2`` fingerprints decoded pixels; ``sdq1`` fingerprints windows of
multichannel proprioceptive signal (joint positions, IMU channels, pose
trajectories) with the same overall recipe - normalize, project onto a small
basis, binarize against a median, compare by Hamming distance - but with a
time representation chosen for the failure mode signals actually have:
**phase**. A copy whose sampling grid is shifted by a fraction of a window
(an off-grid trim, a stream-level reversal) still contains the same motion,
so the time axis is reduced to *band energies* of the Fourier magnitude
spectrum, which shift only mildly under sub-window misalignment and are
exactly invariant to time reversal. Signed spectral coefficients (the naive
port of PDQ's DCT bits) measure phase and fall to the unrelated floor at a
28 ms shift - measured, which is why this kernel does not use them.

One window = ``WINDOW_S`` seconds of all channels, taken on the same fixed
``(k + 0.5) / fps`` grid the video path samples frames on, so the matcher's
segment/slope machinery (trims, splices, concatenation) works unchanged.

Pipeline per window (C channels x T native samples):

1. time axis resampled to a fixed 256 samples (``cv2.INTER_AREA``) - the
   analog of PDQ squashing every image to a fixed square, which makes the
   hash independent of the native sample rate,
2. per-channel window mean removed (constant offsets vanish exactly; the
   Hann window would otherwise leak them into the low bins), then a Hann
   window in time,
3. cross-channel structure: orthonormal DCT rows 0..31 over the channel
   axis - channel structure carries most of the discriminative signal and
   has no phase problem,
4. per row, the Fourier *power* in 8 two-bin bands spanning cycles 1..9 per
   window - the support where human/robot motion energy actually lives;
   wider cells higher up would be noise-dominated coin flips,
5. the 32 x 8 log-energy slab is binarized against each band's median
   across rows -> 256 bits.

Uniform scaling shifts every log energy equally and cancels in the median;
per-channel gains are handled by sp1 prep in the reader.

Measured on real data (see benchmarks/bench_sdq_noise.py): unrelated windows
sit at ~122-126 +/- 11 bits; a 125 ms grid misalignment costs ~45; a 2%
resample ~10; time reversal exactly 0. Gaussian noise margins depend on the
native rate, because only the in-band fraction of white noise lands in the
band slab: at 1 kHz, noise at 5% of per-channel signal sd flips ~7 bits; at
90 Hz (mocap-rate data) the same relative noise flips ~56 - still 6 sigma
from the floor, but thinner. Both are honest numbers, not tuning artifacts.

The FrameHash ``mirror`` slot is deliberately inert (all zero bytes): time
reversal already lands on the identity digest, and no other orientation
variant exists for generic channel sets. An all-zero code is ~128 bits from
any real digest, so the slot can never produce a hit. Whole-stream reversal
therefore produces identity hits along a slope of -1; the temporal matcher
does not fit negative slopes yet, which is a documented detection gap.
"""

from __future__ import annotations

import cv2
import numpy as np

from overlap.hashing.base import FLAG_LOW_QUALITY, HASH_BYTES, FrameHash
from overlap.hashing.pdq_numpy import pack_bits

SIGNAL_ALGO_ID = "sdq1"
# sp1: per-channel robust scaling (median/IQR over the whole stream, computed
# once by the reader - the signal analog of the per-stream border crop),
# channels in sorted column order, NaN-dense and near-constant channels
# dropped (see pose_parquet), 1 s windows.
SIGNAL_PREP_ID = "sp1"

# Window length; part of prep identity. 2 s beats 1 s on every window-level
# measure at mocap rates (5% noise: 26 vs 65 bits; 125 ms shift: 24 vs 52) -
# more content per hash and double the frequency resolution in the band slab.
WINDOW_S = 2.0
_TIME_RES = 256  # fixed time resolution after resampling
_N_ROWS = 32  # channel-DCT rows
_N_BANDS = 8  # two-bin Fourier bands over cycles 1..9 per window

# Windows below this quality (near-idle signal) are hashed and stored but
# flagged and excluded from ANN candidate generation, for the same reason
# blank video frames are: near-featureless windows collide and poison matching.
QUALITY_FLOOR = 20

_HANN = np.hanning(_TIME_RES)
_DCT_CHAN_CACHE: dict[int, np.ndarray] = {}


def _dct_chan(c: int) -> np.ndarray:
    """Orthonormal DCT-II rows 0.._N_ROWS-1 over c channel positions."""
    cached = _DCT_CHAN_CACHE.get(c)
    if cached is None:
        i = np.arange(_N_ROWS, dtype=np.float64)[:, None]
        x = np.arange(c, dtype=np.float64)[None, :]
        d = np.sqrt(2.0 / c) * np.cos((np.pi / 2.0 / c) * i * (2.0 * x + 1.0))
        d[0] = np.sqrt(1.0 / c)
        cached = d
        _DCT_CHAN_CACHE[c] = cached
    return cached


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
        w -= w.mean(axis=1, keepdims=True)
        rows = _dct_chan(c) @ (w * _HANN[None, :])  # _N_ROWS x _TIME_RES
        power: np.ndarray = np.abs(np.fft.rfft(rows, axis=1)) ** 2
        bands = np.stack(
            [power[:, b + 1] + power[:, b + 2] for b in range(_N_BANDS)], axis=1
        )
        energy = np.log(bands + 1e-12)  # _N_ROWS x _N_BANDS
        bits = energy > np.median(energy, axis=0, keepdims=True)
        # sp1-normalized units: rms 1.0 = signal moving about one IQR. Idle
        # stretches (a robot holding still) score near 0 and get flagged.
        rms = float(np.sqrt(np.mean(window * window)))
        quality = min(100, int(round(rms * 50)))
        flags = 0
        if quality < self.quality_floor:
            flags |= FLAG_LOW_QUALITY
        return FrameHash(
            hash=pack_bits(bits.astype(np.uint8).reshape(-1)),
            mirror=b"\x00" * HASH_BYTES,  # inert; see module docstring
            quality=quality,
            flags=flags,
        )
