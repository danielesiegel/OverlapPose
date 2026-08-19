"""sdq1 kernel: bit behaviour, invariances, and noise margins.

The noise/floor margins mirror the measured study in the README: unrelated
windows sit at ~128 +/- 10 bits while Gaussian noise at 5% of signal sd flips
only a handful, so the assertions here use loose bounds well outside either
distribution's tail - they fail on a broken kernel, not on an unlucky seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from overlap.hashing.pdq_numpy import hamming
from overlap.hashing.sdq import SIGNAL_ALGO_ID, SdqKernel

rng = np.random.default_rng(11)


def _motion_window(c: int = 64, t: int = 1000, seed: int = 0) -> np.ndarray:
    """Synthetic sp1-scaled motion: smooth low-frequency multichannel signal."""
    r = np.random.default_rng(seed)
    freqs = r.uniform(0.5, 6.0, size=(c, 4))
    phases = r.uniform(0, 2 * np.pi, size=(c, 4))
    amps = r.uniform(0.3, 1.2, size=(c, 4))
    tt = np.linspace(0.0, 1.0, t)[None, None, :]
    window: np.ndarray = np.sum(
        amps[:, :, None] * np.sin(2 * np.pi * freqs[:, :, None] * tt + phases[:, :, None]),
        axis=1,
    )
    return window


def test_identity_and_determinism() -> None:
    kernel = SdqKernel()
    assert kernel.algo_id == SIGNAL_ALGO_ID
    w = _motion_window()
    a, b = kernel.hash_frame(w), kernel.hash_frame(w.copy())
    assert a.hash == b.hash
    assert a.mirror == b.mirror
    assert len(a.hash) == 32


def test_offset_and_scale_invariance() -> None:
    """Per-channel constant offsets die in the time-DCT (DC excluded); uniform
    positive scaling cancels in the median threshold."""
    kernel = SdqKernel()
    w = _motion_window(seed=1)
    base = kernel.hash_frame(w).hash
    assert kernel.hash_frame(w + 3.7).hash == base
    assert kernel.hash_frame(w * 2.5).hash == base


def test_sample_rate_independence() -> None:
    """The same 1 s of signal at 500 Hz and 1 kHz hashes nearly identically:
    the kernel resamples time to a fixed length first."""
    kernel = SdqKernel()
    hi = _motion_window(t=1000, seed=2)
    lo = hi[:, ::2]
    d = hamming(kernel.hash_frame(hi).hash, kernel.hash_frame(lo).hash)
    assert d <= 8, d


def test_time_reversal_lands_on_mirror() -> None:
    kernel = SdqKernel()
    w = _motion_window(t=1024, seed=3)  # integer decimation: reversal is exact
    fh = kernel.hash_frame(w)
    fh_rev = kernel.hash_frame(w[:, ::-1])
    assert fh_rev.hash == fh.mirror
    assert fh_rev.mirror == fh.hash


def test_gaussian_noise_margin() -> None:
    """Noise at 5% of signal sd must stay far inside the candidate radius."""
    kernel = SdqKernel()
    w = _motion_window(seed=4)
    noisy = w + rng.normal(0.0, 0.05 * w.std(), size=w.shape)
    assert hamming(kernel.hash_frame(w).hash, kernel.hash_frame(noisy).hash) <= 30


def test_unrelated_windows_near_half_bits() -> None:
    kernel = SdqKernel()
    dists = [
        hamming(
            kernel.hash_frame(_motion_window(seed=100 + i)).hash,
            kernel.hash_frame(_motion_window(seed=200 + i)).hash,
        )
        for i in range(8)
    ]
    assert min(dists) > 80, dists
    assert 100 < float(np.mean(dists)) < 156, dists


def test_idle_window_flagged_low_quality() -> None:
    from overlap.hashing.base import FLAG_LOW_QUALITY

    kernel = SdqKernel()
    idle = rng.normal(0.0, 0.05, size=(64, 1000))  # near-still robot
    assert kernel.hash_frame(idle).flags & FLAG_LOW_QUALITY
    assert not kernel.hash_frame(_motion_window(seed=5)).flags & FLAG_LOW_QUALITY


def test_rejects_bad_shapes() -> None:
    kernel = SdqKernel()
    with pytest.raises(ValueError):
        kernel.hash_frame(np.zeros(100))
    with pytest.raises(ValueError):
        kernel.hash_frame(np.zeros((1, 100)))
    with pytest.raises(ValueError):
        kernel.hash_frame(np.zeros((10, 4)))
