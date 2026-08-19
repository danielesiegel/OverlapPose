"""sdq1 kernel: bit behaviour, invariances, and margins.

The margins mirror the measured study in the README (reproduce with
benchmarks/bench_sdq_noise.py): unrelated windows sit at ~122-126 +/- 11 bits,
a 125 ms grid misalignment costs ~45, time reversal is exact. Assertions use
loose bounds well outside either distribution's tail - they fail on a broken
kernel, not on an unlucky seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from overlap.hashing.base import HASH_BYTES
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


def _motion_stream(
    c: int = 64, seconds: float = 4.0, rate: int = 1000, seed: int = 0
) -> np.ndarray:
    r = np.random.default_rng(seed)
    n = int(seconds * rate)
    tt = np.arange(n) / rate
    x = np.zeros((c, n))
    for ch in range(c):
        for _ in range(4):
            x[ch] += r.uniform(0.3, 1.2) * np.sin(
                2 * np.pi * r.uniform(0.5, 6.0) * tt + r.uniform(0, 2 * np.pi)
            )
    return x


def test_identity_and_determinism() -> None:
    kernel = SdqKernel()
    assert kernel.algo_id == SIGNAL_ALGO_ID
    w = _motion_window()
    a, b = kernel.hash_frame(w), kernel.hash_frame(w.copy())
    assert a.hash == b.hash
    assert len(a.hash) == HASH_BYTES


def test_mirror_slot_is_inert() -> None:
    """No orientation variant exists for generic channel sets; the slot is
    all-zero so it can never be within matching radius of a real digest."""
    fh = SdqKernel().hash_frame(_motion_window())
    assert fh.mirror == b"\x00" * HASH_BYTES
    assert hamming(fh.hash, fh.mirror) > 80


def test_offset_and_scale_invariance() -> None:
    """Per-channel constant offsets die in the window-mean removal; uniform
    positive scaling shifts every log energy equally and cancels in the
    per-band median."""
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


def test_time_reversal_is_identity() -> None:
    """Fourier band powers are exactly invariant to reversal, so a reversed
    window is the *same* fingerprint - reversed copies match without a
    dedicated variant slot."""
    kernel = SdqKernel()
    w = _motion_window(t=1024, seed=3)  # integer decimation: exact
    assert kernel.hash_frame(w[:, ::-1]).hash == kernel.hash_frame(w).hash
    w2 = _motion_window(t=1000, seed=3)  # non-integer resample: near-exact
    assert hamming(kernel.hash_frame(w2[:, ::-1]).hash, kernel.hash_frame(w2).hash) <= 4


def test_phase_shift_stays_inside_radius() -> None:
    """An off-grid copy is misaligned by up to half a hop (125 ms). Band
    energies move a little; signed spectral bits would be at the floor (~120)."""
    kernel = SdqKernel()
    x = _motion_stream(seed=5)
    win = 1000
    dists = [
        hamming(
            kernel.hash_frame(x[:, s : s + win]).hash,
            kernel.hash_frame(x[:, s + 125 : s + 125 + win]).hash,
        )
        for s in range(0, x.shape[1] - win - 125, 500)
    ]
    assert float(np.mean(dists)) < 56, dists


def test_gaussian_noise_margin() -> None:
    """Noise at 5% of signal sd must stay far inside the candidate radius."""
    kernel = SdqKernel()
    w = _motion_window(seed=4)
    noisy = w + rng.normal(0.0, 0.05 * w.std(), size=w.shape)
    assert hamming(kernel.hash_frame(w).hash, kernel.hash_frame(noisy).hash) <= 30


def test_small_resample_stays_inside_radius() -> None:
    """A 2% speed change moves band energies only slightly - small resamples
    are now detectable (deep ones are still a documented gap)."""
    kernel = SdqKernel()
    x = _motion_stream(seed=6)
    idx = np.arange(0, int((x.shape[1] - 1) / 1.02)) * 1.02
    xr = np.stack([np.interp(idx, np.arange(x.shape[1]), row) for row in x])
    d = hamming(kernel.hash_frame(x[:, :1000]).hash, kernel.hash_frame(xr[:, :1000]).hash)
    assert d <= 40, d


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
