"""Unit tests for the pdq2 hash kernel.

The golden digests pin the exact bit-behavior of the implementation: if any
of these change, every existing index and manifest in the wild silently stops
matching - such a change requires a new algo_id, not a fixed test.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from overlap.hashing import FLAG_LOW_QUALITY, PdqKernel
from overlap.hashing.pdq_numpy import hamming, pack_bits, to_reference_hex, unpack_bits

GOLDEN_TEXTURED_HASH = "33ca34038065823f91519ad733c3bbc6e523d3fb4f46c7459c396b783395c351"
GOLDEN_TEXTURED_MIRROR = "99209ea1284f28953bfb307d9969116c4f81795165ec6def1293c1d2993f69cb"


def textured_img() -> np.ndarray:
    rng = np.random.default_rng(42)
    base = rng.integers(0, 256, size=(360, 480)).astype(np.int32)
    yy, xx = np.mgrid[0:360, 0:480]
    rings = (np.sin(np.hypot(yy - 180, xx - 240) / 12.0) * 60 + 120).astype(np.int32)
    return ((base + rings) // 2).astype(np.uint8)


def blob_img() -> np.ndarray:
    rng = np.random.default_rng(7)
    a = cv2.resize(rng.normal(128, 60, size=(90, 120)), (640, 480), interpolation=cv2.INTER_CUBIC)
    return np.clip(a, 0, 255).astype(np.uint8)


def test_speed_is_within_budget() -> None:
    """Hashing dominates indexing cost, so a regression here is expensive.

    Smoothing at the working resolution was once 92% of the hash and made a
    3-minute clip cost 8 minutes of CPU; this guards the fix.
    """
    import time

    k = PdqKernel()
    img = textured_img()
    k.hash_frame(img)
    t0 = time.perf_counter()
    for _ in range(20):
        k.hash_frame(img)
    ms = (time.perf_counter() - t0) / 20 * 1000
    assert ms < 8.0, f"hash took {ms:.1f} ms; it should be a few ms"


def test_golden_vector_pins_bit_behavior() -> None:
    fh = PdqKernel().hash_frame(textured_img())
    assert fh.hash.hex() == GOLDEN_TEXTURED_HASH
    assert fh.mirror.hex() == GOLDEN_TEXTURED_MIRROR
    assert fh.quality == 100
    assert fh.flags == 0


@pytest.mark.parametrize("img_factory", [textured_img, blob_img])
def test_flip_symmetry_is_exact(img_factory) -> None:  # type: ignore[no-untyped-def]
    """hash(fliplr(img)) == mirror(img), bit-exactly - the flip-detection contract."""
    k = PdqKernel()
    img = img_factory()
    fh = k.hash_frame(img)
    fh_flip = k.hash_frame(np.ascontiguousarray(np.fliplr(img)))
    assert fh_flip.hash == fh.mirror
    assert fh_flip.mirror == fh.hash


@pytest.mark.parametrize("img_factory", [textured_img, blob_img])
def test_brightness_shift_is_nearly_invariant(img_factory) -> None:  # type: ignore[no-untyped-def]
    k = PdqKernel()
    img = img_factory()
    bright = np.clip(img.astype(np.int16) + 24, 0, 255).astype(np.uint8)
    assert hamming(k.hash_frame(bright).hash, k.hash_frame(img).hash) <= 4


def test_rescale_is_nearly_invariant() -> None:
    """Downscale-then-upscale (a re-encode proxy) moves only a few bits."""
    k = PdqKernel()
    img = textured_img()
    small = cv2.resize(img, (240, 180), interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (480, 360), interpolation=cv2.INTER_LINEAR)
    assert hamming(k.hash_frame(back).hash, k.hash_frame(img).hash) <= 12


def test_featureless_frame_is_flagged_low_quality() -> None:
    fh = PdqKernel().hash_frame(np.full((480, 640), 128, dtype=np.uint8))
    assert fh.quality == 0
    assert fh.flags & FLAG_LOW_QUALITY


def test_unrelated_images_are_far_apart() -> None:
    k = PdqKernel()
    d = hamming(k.hash_frame(textured_img()).hash, k.hash_frame(blob_img()).hash)
    assert d > 80  # random 256-bit pairs center on 128


def test_pack_unpack_roundtrip() -> None:
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=256).astype(bool)
    assert np.array_equal(unpack_bits(pack_bits(bits)), bits)


def test_hamming_basics() -> None:
    a = bytes(32)
    b = bytes([0xFF]) + bytes(31)
    assert hamming(a, a) == 0
    assert hamming(a, b) == 8


def test_reference_hex_bit_order() -> None:
    bits = np.zeros(256, dtype=bool)
    bits[0] = True  # word 0, LSB -> last 4 hex chars become 0001
    assert to_reference_hex(bits) == "0" * 60 + "0001"
    bits = np.zeros(256, dtype=bool)
    bits[255] = True  # word 15, bit 15 -> leading hex digit 8
    assert to_reference_hex(bits) == "8000" + "0" * 60


def test_too_small_frame_rejected() -> None:
    with pytest.raises(ValueError, match="too small"):
        PdqKernel().hash_frame(np.zeros((1, 10), dtype=np.uint8))
