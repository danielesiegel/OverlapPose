"""Measure sdq1 noise margins and the unrelated-window floor.

Reproduces the numbers quoted in the OverlapPose README section: bit flips
under Gaussian noise at a ladder of amplitudes, the unrelated-pair Hamming
floor, and the (documented, negative) resample result.

Usage:
    python benchmarks/bench_sdq_noise.py [pose.parquet]

With a Parquet path the study runs on that file's dense channels via the real
reader prep; without one it runs on synthetic multichannel motion so the
harness is runnable anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from overlap.hashing.pdq_numpy import hamming
from overlap.hashing.sdq import WINDOW_S, SdqKernel

rng = np.random.default_rng(7)
HOP_S = 0.25  # 4 windows/s, the default index grid


def load_channels(path: Path) -> tuple[np.ndarray, float]:
    """Dense channel matrix (C x N, sp1-scaled) and sample rate, via the reader."""
    from overlap.readers.pose_parquet import PoseParquetReader

    session = PoseParquetReader.open(path)
    return session._x, session._rate_hz  # noqa: SLF001 - benchmark introspection


def synthetic_channels(c: int = 132, seconds: int = 120, rate: float = 1000.0) -> np.ndarray:
    n = int(seconds * rate)
    tt = np.arange(n) / rate
    x = np.zeros((c, n))
    for ch in range(c):
        for _ in range(4):
            f = rng.uniform(0.3, 8.0)
            x[ch] += rng.uniform(0.3, 1.2) * np.sin(2 * np.pi * f * tt + rng.uniform(0, 6))
    iqr = np.percentile(x, 75, axis=1, keepdims=True) - np.percentile(x, 25, axis=1, keepdims=True)
    return (x - np.median(x, axis=1, keepdims=True)) / iqr


def codes(x: np.ndarray, rate: float) -> np.ndarray:
    kernel = SdqKernel()
    win = int(round(WINDOW_S * rate))
    hop = int(round(HOP_S * rate))
    starts = np.arange(0, x.shape[1] - win + 1, hop)
    return np.array(
        [np.frombuffer(kernel.hash_frame(x[:, s : s + win]).hash, "u1") for s in starts]
    )


def dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([hamming(r.tobytes(), s.tobytes()) for r, s in zip(a, b, strict=True)])


def main() -> None:
    if len(sys.argv) > 1:
        x, rate = load_channels(Path(sys.argv[1]))
        print(f"{sys.argv[1]}: {x.shape[0]} channels @ {rate:g} Hz")
    else:
        x, rate = synthetic_channels(), 1000.0
        print(f"synthetic motion: {x.shape[0]} channels @ {rate:g} Hz")

    base = codes(x, rate)
    n_win = len(base)

    print("\nGaussian noise sweep (sigma as fraction of per-channel sd; flips of 256):")
    sd = x.std(axis=1, keepdims=True)
    for rho in (0.01, 0.05, 0.1, 0.2, 0.5, 1.0):
        noisy = codes(x + rng.normal(0.0, rho * sd, size=x.shape), rate)
        d = dist(base, noisy)
        print(f"  {rho*100:>5.0f}%  mean {d.mean():5.1f}  p95 {np.percentile(d, 95):3.0f}")

    pairs = rng.integers(0, n_win, size=(4000, 2))
    pairs = pairs[np.abs(pairs[:, 0] - pairs[:, 1]) * HOP_S > 10]
    floor = dist(base[pairs[:, 0]], base[pairs[:, 1]])
    print(f"\nunrelated floor (>10 s apart): mean {floor.mean():.1f} sd {floor.std():.1f} "
          f"min {floor.min()}")

    # Phase: an off-grid copy is misaligned by up to half a hop (125 ms).
    d_shift = int(round(0.125 * rate))
    shifted = codes(x[:, d_shift:], rate)
    m = min(len(shifted), n_win)
    print(f"125 ms grid misalignment: mean {dist(base[:m], shifted[:m]).mean():.1f}")

    # Small speed change: band energies move only slightly.
    idx = np.arange(0, int(x.shape[1] / 1.02)) * 1.02
    xr = np.stack([np.interp(idx, np.arange(x.shape[1]), row) for row in x])
    res = codes(xr, rate)
    m = min(len(res), n_win)
    aligned = np.clip(np.round(np.arange(m) * 1.02).astype(int), 0, n_win - 1)
    d_res = dist(base[aligned], res[:m])
    print(f"2% resample (speed change), content-aligned: mean {d_res.mean():.1f}")

    # Reversal is exact by construction; measure it anyway.
    rev = codes(x[:, ::-1], rate)
    m = min(len(rev), n_win)
    aligned = np.clip(n_win - 1 - np.arange(m), 0, n_win - 1)
    print(f"time reversal, content-aligned: mean {dist(base[aligned], rev[:m]).mean():.1f}")


if __name__ == "__main__":
    main()
