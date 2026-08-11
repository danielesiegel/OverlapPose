"""Would cosine similarity, or keeping the DCT magnitudes, work better?

Two claims, both testable. On binary codes the metrics are the same ordering: for
+/-1 vectors dot = n - 2*hamming, so cosine, dot and Euclidean rank every pair
identically to Hamming. The real alternative is keeping the DCT magnitudes and
comparing by cosine instead of thresholding to sign bits.

What matters is separation, not raw distance: how far a manipulated copy sits from
its source relative to how far unrelated frames sit. Reported as a z-score against
each representation's own unrelated-pair distribution, so the two are comparable
despite different units.
"""

from __future__ import annotations

import cv2
import numpy as np
from _common import centre_crop, clips, edge_crop, gray_frames, parse_args

from overlap.hashing import pdq_numpy as P

_D = P._dct_matrix()


def dct_vector(gray: np.ndarray) -> np.ndarray:
    return (_D @ P._downsample(np.ascontiguousarray(gray)) @ _D.T).reshape(-1)


def bits(gray: np.ndarray) -> np.ndarray:
    v = dct_vector(gray)
    return v > np.median(v)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[1:], b[1:]  # drop DC: brightness, not structure
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    args = parse_args(__doc__)
    frames = gray_frames(clips(args.clips, args.limit), args.per_clip)
    ref_bits = [bits(f) for f in frames]
    ref_vecs = [dct_vector(f) for f in frames]
    n = len(frames)

    un_h = np.array(
        [np.sum(ref_bits[i] != ref_bits[j]) for i in range(n) for j in range(i + 1, n)], float
    )
    un_c = np.array([cosine(ref_vecs[i], ref_vecs[j]) for i in range(n) for j in range(i + 1, n)])
    print(f"{n} frames. unrelated-pair baseline:")
    print(f"  hamming mean {un_h.mean():.1f} sd {un_h.std():.1f} of 256")
    print(f"  cosine  mean {un_c.mean():.3f} sd {un_c.std():.3f}\n")
    print(f"{'manipulation':26s} {'hamming':>18s} {'cosine':>18s}")

    def row(label: str, fn) -> None:  # type: ignore[no-untyped-def]
        pairs = zip(frames, ref_bits, strict=True)
        h = np.array([np.sum(bits(fn(f)) != b) for f, b in pairs], float)
        c = np.array([cosine(dct_vector(fn(f)), v) for f, v in zip(frames, ref_vecs, strict=True)])
        zh = (un_h.mean() - h.mean()) / un_h.std()
        zc = (c.mean() - un_c.mean()) / un_c.std()
        print(f"  {label:24s} {h.mean():7.1f} (z={zh:5.1f}) {c.mean():8.3f} (z={zc:5.1f})")

    row("re-encode jpeg q40", lambda f: cv2.imdecode(
        cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 40])[1], cv2.IMREAD_GRAYSCALE))
    row("brightness +24", lambda f: np.clip(f.astype(np.int16) + 24, 0, 255).astype(np.uint8))
    for pct in (5, 10, 15, 20, 30):
        row(f"centre crop {pct}%", lambda f, p=pct: centre_crop(f, 1 - p / 100))
    row("bottom strip 15%", lambda f: edge_crop(f, 0.15))
    print("\nA z below about 2 is not a detector, whatever the raw distance looks like.")


if __name__ == "__main__":
    main()
