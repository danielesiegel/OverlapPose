"""PDQ-style perceptual hash - overlap's reference implementation (``pdq2``).

The algorithm follows PDQ, designed and published by Meta in the
ThreatExchange project (https://github.com/facebook/ThreatExchange/tree/main/pdq):
a 256-bit DCT-sign hash over a smoothed 64x64 luminance downsample, with an
image-domain quality metric, thresholded at the median of a 16x16 DCT slab.

overlap deviates from the reference implementation in the downsample stage,
deliberately, to make the pipeline bit-exactly mirror-symmetric - hash(fliplr(img))
equals the derived mirror hash of img exactly, for any
input. The reference pipeline is not mirror-symmetric (even-width box windows
and truncating decimation), which makes horizontal-flip detection fragile on
low-asymmetry content. Deviations:

1. fixed pre-resize to 256x256 (``cv2.INTER_AREA``) instead of size-dependent
   filter windows,
2. two-pass Jarosz-style box filtering with a fixed *odd* window (5),
3. decimation by exact block mean (4x4 at the 256px working size) instead of
   point sampling.

No cross-implementation distance to reference PDQ is claimed: it is not
measured here, the reference library is not a dependency, and the 256px working
size makes the two differ by more than the earlier 512px version did. Only
overlap-to-overlap comparisons are bit-stable, which is what ``algo_id`` exists
to enforce - a mismatch between two sides is refused, never approximated.

What *is* pinned, by tests that fail on drift: the golden digests in
tests/unit/test_pdq.py (exact bit behaviour), exact mirror symmetry, and the
robustness margins for brightness, rescaling and unrelated frames.

The horizontal-mirror hash is derived from the same DCT at almost zero cost:
mirroring an image horizontally multiplies DCT column ``v`` by ``(-1)**(v+1)``.
"""

from __future__ import annotations

import cv2
import numpy as np

from overlap.hashing.base import HASH_BYTES

_DCT_D: np.ndarray | None = None
# Working resolution before smoothing. The output is 64x64 regardless, so this
# only sets how much memory traffic each hash costs - and the smoothing passes
# over it were 92% of the total. 512 was actively wasteful: typical robotics
# footage is smaller than that in at least one dimension (456x256 for
# Egocentric-100K), so it upsampled first and then paid 4x the bandwidth to
# filter the interpolated pixels. Measured on real frames, dropping to 256
# leaves mirror symmetry exact and the brightness, re-encode, crop-scale and
# unrelated-pair distances unchanged, at a quarter of the cost.
_PRE_SIZE = 256
_BOX_WINDOW = 5  # odd => symmetric box filter => mirror symmetry holds exactly
# float32 halves the memory traffic of the filter passes. The filtered values
# are small integers divided by tap counts, so this is a bandwidth choice, not
# a precision one; it is nonetheless part of the hash identity.
_WORK_DTYPE = np.float32


def _dct_matrix() -> np.ndarray:
    """16x64 DCT-II matrix over frequencies 1..16 (DC row excluded)."""
    global _DCT_D
    if _DCT_D is None:
        i = np.arange(1, 17, dtype=np.float64)[:, None]  # frequencies 1..16
        x = np.arange(64, dtype=np.float64)[None, :]
        _DCT_D = np.sqrt(2.0 / 64.0) * np.cos((np.pi / 2.0 / 64.0) * i * (2.0 * x + 1.0))
    return _DCT_D


def _valid_tap_counts(n: int, full_window: int) -> np.ndarray:
    """How many in-range taps each output position averages over."""
    f = full_window
    h = (f + 2) // 2
    idx = np.arange(n)
    lo = np.maximum(0, idx - (h - 1))
    hi = np.minimum(n, idx + (f - h) + 1)
    return (hi - lo).astype(_WORK_DTYPE)


_COUNT_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _counts_cached(n: int, full_window: int) -> np.ndarray:
    key = (n, full_window)
    cached = _COUNT_CACHE.get(key)
    if cached is None:
        cached = _valid_tap_counts(n, full_window)
        _COUNT_CACHE[key] = cached
    return cached


def _box_filter_axis0(a: np.ndarray, full_window: int) -> np.ndarray:
    """1-D box filter along axis 0; edge windows shrink symmetrically.

    out[i] = mean(a[lo:hi]) with lo = max(0, i - (h-1)), hi = min(n, i + f - h + 1),
    where h = (f + 2) // 2. For odd f this is a centered window, which is what
    keeps the pipeline mirror-symmetric.

    Computed as an unnormalized zero-padded box sum (SIMD, via OpenCV) divided
    by each position's valid-tap count. That is arithmetically identical to
    averaging the shrunken window, and ~40x faster than doing it with cumsum
    plus fancy indexing in numpy - which profiling showed was the entire cost
    of hashing a frame, transposes and 2 MB temporaries included.
    """
    n = a.shape[0]
    if full_window <= 1 or n == 0:
        return a.astype(_WORK_DTYPE, copy=True)
    src = np.ascontiguousarray(a, dtype=_WORK_DTYPE)
    # ksize is (width, height): a kernel 1 wide and `full_window` tall filters
    # along axis 0. Odd windows are centered, matching the definition above.
    sums = cv2.boxFilter(
        src,
        -1,
        (1, full_window),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )
    counts = _counts_cached(n, full_window)
    shape = (n,) + (1,) * (a.ndim - 1)
    result: np.ndarray = sums / counts.reshape(shape)
    return result


def _box_filter_axis1(a: np.ndarray, full_window: int) -> np.ndarray:
    """Same filter along axis 1, without transposing the array.

    The previous implementation transposed, filtered axis 0, and transposed
    back; on a 512x512 array that walks memory against its stride twice per
    pass.
    """
    n = a.shape[1]
    if full_window <= 1 or n == 0:
        return a.astype(_WORK_DTYPE, copy=True)
    src = np.ascontiguousarray(a, dtype=_WORK_DTYPE)
    sums = cv2.boxFilter(
        src,
        -1,
        (full_window, 1),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )
    counts = _counts_cached(n, full_window)
    result: np.ndarray = sums / counts.reshape((1, n))
    return result


def _downsample(gray: np.ndarray) -> np.ndarray:
    """Pre-resize -> two-pass box smoothing -> block mean -> 64x64."""
    a = cv2.resize(gray, (_PRE_SIZE, _PRE_SIZE), interpolation=cv2.INTER_AREA).astype(_WORK_DTYPE)
    for _ in range(2):
        a = _box_filter_axis1(a, _BOX_WINDOW)  # horizontal
        a = _box_filter_axis0(a, _BOX_WINDOW)  # vertical
    blk = _PRE_SIZE // 64
    result: np.ndarray = a.reshape(64, blk, 64, blk).mean(axis=(1, 3), dtype=np.float64)
    return result


def _quality(a64: np.ndarray) -> int:
    """Image-domain quality metric: normalized gradient energy, 0..100."""
    grad = np.sum(np.abs(np.diff(a64, axis=0))) + np.sum(np.abs(np.diff(a64, axis=1)))
    q = int(grad * 100.0 / 255.0) // 90
    return min(q, 100)


def _bits_from_dct(dct: np.ndarray) -> np.ndarray:
    """256 bool bits, row-major over the 16x16 DCT slab, thresholded at median."""
    median = np.median(dct)
    bits: np.ndarray = (dct > median).reshape(-1)
    return bits


# Horizontal mirror in image space multiplies DCT column v by (-1)^(v+1):
# +1 for odd v, -1 for even v.
_MIRROR_COL_SIGNS = np.where(np.arange(16) % 2 == 1, 1.0, -1.0)[None, :]


# Bit i of a packed digest is DCT cell (i // 16, i % 16). Mirroring negates the
# columns whose sign is -1 above, so inverting exactly those bit positions turns
# an identity digest into its mirror *without the pixels* - the median used for
# thresholding shifts a little, so this is close, not exact.
# Measured on 49 real frames: median 8 bits from the true mirror, worst 20, well
# inside the candidate radius of 56. Lets a manifest carry flip detection while
# shipping one digest per frame, and lets an index skip storing mirrors.
_MIRROR_BIT_FLIP = np.packbits(
    (_MIRROR_COL_SIGNS.reshape(-1)[np.arange(256) % 16] < 0).astype(np.uint8)
)


def mirror_of_packed(hashes: bytes) -> bytes:
    """Derive mirror digests for a packed run of identity digests."""
    if len(hashes) % HASH_BYTES:
        raise ValueError(f"length {len(hashes)} is not a multiple of {HASH_BYTES}")
    arr = np.frombuffer(hashes, dtype=np.uint8).reshape(-1, HASH_BYTES)
    return np.bitwise_xor(arr, _MIRROR_BIT_FLIP).tobytes()


def compute_pdq_dihedral(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute (identity_bits, mirrored_bits, quality) for a grayscale frame.

    ``gray`` is HxW uint8 (or float in 0..255). Both bit arrays are length-256
    bool, row-major. ``mirrored_bits`` equals the identity bits of the
    horizontally-flipped frame, exactly (see module docstring).
    """
    if gray.ndim != 2:
        raise ValueError(f"expected 2-D grayscale frame, got shape {gray.shape}")
    if gray.shape[0] < 2 or gray.shape[1] < 2:
        raise ValueError(f"frame too small to hash: {gray.shape}")
    a64 = _downsample(gray)
    quality = _quality(a64)
    d = _dct_matrix()
    dct = d @ a64 @ d.T
    bits_id = _bits_from_dct(dct)
    bits_mx = _bits_from_dct(dct * _MIRROR_COL_SIGNS)
    return bits_id, bits_mx, quality


def compute_pdq(gray: np.ndarray) -> tuple[np.ndarray, int]:
    """Compute (bits, quality) for a grayscale frame (identity orientation)."""
    bits_id, _, quality = compute_pdq_dihedral(gray)
    return bits_id, quality


def pack_bits(bits: np.ndarray) -> bytes:
    """Pack 256 bool bits (row-major, k = i*16 + j) into overlap's 32-byte digest.

    Byte b holds bits 8b..8b+7 MSB-first (``np.packbits`` convention). This is
    the storage and manifest format; it is fixed and versioned via algo_id.
    """
    return np.packbits(bits.astype(np.uint8)).tobytes()


def unpack_bits(digest: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(digest, dtype=np.uint8)).astype(bool)


def to_reference_hex(bits: np.ndarray) -> str:
    """Render bits in the ThreatExchange reference hex format (for test vectors).

    The reference packs bit k into 16-bit word ``k >> 4`` at position ``k & 15``
    (LSB-first) and prints words 15..0 as 4 hex digits each.
    """
    words = []
    for w in range(15, -1, -1):
        word = 0
        for t in range(16):
            if bits[w * 16 + t]:
                word |= 1 << t
        words.append(f"{word:04x}")
    return "".join(words)


def hamming(a: bytes, b: bytes) -> int:
    """Hamming distance between two packed digests."""
    x = np.frombuffer(a, dtype=np.uint8) ^ np.frombuffer(b, dtype=np.uint8)
    return int(np.unpackbits(x).sum())
