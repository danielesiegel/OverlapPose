"""Hash throughput, and whether the kernel still satisfies its invariants.

Speed is the headline, but a faster hash that broke mirror symmetry or drifted
under re-encoding would be useless, so both are checked in one place.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from overlap.hashing import FLAG_LOW_QUALITY, PdqKernel
from overlap.hashing.pdq_numpy import hamming

RADIUS = 56


def textured(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(360, 480)).astype(np.int32)
    yy, xx = np.mgrid[0:360, 0:480]
    rings = (np.sin(np.hypot(yy - 180, xx - 240) / 12.0) * 60 + 120).astype(np.int32)
    return ((base + rings) // 2).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iterations", type=int, default=200)
    args = ap.parse_args()

    kernel = PdqKernel()
    img = textured()
    kernel.hash_frame(img)  # warm up

    t0 = time.perf_counter()
    for _ in range(args.iterations):
        kernel.hash_frame(img)
    ms = (time.perf_counter() - t0) / args.iterations * 1000
    print(f"hash: {ms:.2f} ms/frame  ({1000 / ms:,.0f} hashes/s single-threaded)")
    print(f"      {ms * 12:.1f} ms per frame at 12 codes/frame\n")

    fh = kernel.hash_frame(img)
    flipped = kernel.hash_frame(np.ascontiguousarray(np.fliplr(img)))
    bright = kernel.hash_frame(np.clip(img.astype(np.int16) + 24, 0, 255).astype(np.uint8))
    small = cv2.resize(img, (240, 180), interpolation=cv2.INTER_AREA)
    rescaled = kernel.hash_frame(cv2.resize(small, (480, 360), interpolation=cv2.INTER_LINEAR))
    blank = kernel.hash_frame(np.full((480, 640), 128, dtype=np.uint8))

    checks = [
        ("mirror symmetry is exact", flipped.hash == fh.mirror, "0 bits required"),
        ("brightness +24", hamming(bright.hash, fh.hash) <= 4,
         f"{hamming(bright.hash, fh.hash)} bits"),
        ("downscale then upscale", hamming(rescaled.hash, fh.hash) <= 12,
         f"{hamming(rescaled.hash, fh.hash)} bits"),
        ("featureless frame flagged", bool(blank.flags & FLAG_LOW_QUALITY), "quality gate"),
    ]
    for name, ok, detail in checks:
        print(f"  {'ok  ' if ok else 'FAIL'} {name:28s} {detail}")
    if not all(ok for _n, ok, _d in checks):
        raise SystemExit("kernel invariants violated")


if __name__ == "__main__":
    main()
