"""Unit tests for the Hough diagonal-run matcher on synthetic hit sets."""

from __future__ import annotations

import numpy as np

from overlap.match.chain import ChainParams, HoughChainMatcher


def _hits(pairs: list[tuple[int, int]], dist: int = 10, mirrored: bool = False):  # type: ignore[no-untyped-def]
    q = np.array([p[0] for p in pairs], dtype=np.int64)
    c = np.array([p[1] for p in pairs], dtype=np.int64)
    d = np.full(len(pairs), dist, dtype=np.int32)
    m = np.full(len(pairs), mirrored, dtype=bool)
    return q, c, d, m


def grid(n: int, fps: float = 1.0) -> list[int]:
    return [int((i + 0.5) * 1000 / fps) for i in range(n)]


def test_perfect_diagonal_found() -> None:
    times = grid(30)
    runs = HoughChainMatcher().find_runs(*_hits([(t, t + 5000) for t in times]))
    assert len(runs) == 1
    run = runs[0]
    assert abs(run.slope - 1.0) < 0.01
    assert run.n_inliers == 30
    assert run.density > 0.9
    assert abs(run.c_start_ms - run.q_start_ms - 5000) < 1500


def test_half_speed_diagonal_yields_slope_and_ratio() -> None:
    # Query is a 0.5x-slowed copy: 60 query seconds map to 30 corpus seconds.
    times = grid(60)
    runs = HoughChainMatcher().find_runs(*_hits([(t, t // 2) for t in times]))
    assert len(runs) == 1
    assert abs(runs[0].slope - 0.5) < 0.02
    assert abs(runs[0].speed_ratio - 2.0) < 0.1


def test_splice_yields_two_runs() -> None:
    part_a = [(t, t + 2000) for t in grid(15)]
    part_b = [(t + 15000, t + 90000) for t in grid(15)]
    runs = HoughChainMatcher().find_runs(*_hits(part_a + part_b))
    assert len(runs) == 2
    offsets = sorted(round((r.c_start_ms - r.q_start_ms) / 1000) for r in runs)
    assert offsets[0] in (1, 2, 3)
    assert offsets[1] in (74, 75, 76)


def test_random_noise_produces_no_runs() -> None:
    rng = np.random.default_rng(0)
    q = rng.integers(0, 600_000, size=100).astype(np.int64)
    c = rng.integers(0, 3_600_000, size=100).astype(np.int64)
    d = rng.integers(20, 42, size=100).astype(np.int32)
    m = np.zeros(100, dtype=bool)
    runs = HoughChainMatcher().find_runs(q, c, d, m)
    # Uniform random hits over a huge plane should not form dense linear runs.
    assert all(r.density < 0.5 for r in runs)


def test_run_survives_noise_hits() -> None:
    times = grid(25)
    real = [(t, t + 10000) for t in times]
    rng = np.random.default_rng(1)
    noise = [(int(rng.integers(0, 25_000)), int(rng.integers(0, 300_000))) for _ in range(40)]
    runs = HoughChainMatcher().find_runs(*_hits(real + noise))
    best = max(runs, key=lambda r: r.n_inliers)
    assert best.n_inliers >= 25
    assert abs(best.slope - 1.0) < 0.05


def test_too_few_hits_no_run() -> None:
    runs = HoughChainMatcher().find_runs(*_hits([(t, t) for t in grid(5)]))
    assert runs == []


def test_mirrored_majority_propagates() -> None:
    times = grid(20)
    runs = HoughChainMatcher().find_runs(*_hits([(t, t) for t in times], mirrored=True))
    assert runs[0].mirrored is True


def test_parallel_runs_both_survive() -> None:
    """Two pieces of one source concatenated: two parallel diagonals must
    both be found. A compromise slope used to eat both."""
    part_a = [(t, t + 2000) for t in grid(20)]
    part_b = [(t + 20000, t + 75000) for t in grid(20)]
    runs = HoughChainMatcher().find_runs(*_hits(part_a + part_b))
    assert len(runs) == 2, [(r.q_start_ms, r.q_end_ms, r.density) for r in runs]
    assert all(r.density > 0.8 for r in runs)


def test_a_wide_hole_is_reported_as_two_segments() -> None:
    """Two stretches with a long gap between them are two segments, not one run.

    Both clusters sit on the same diagonal, so a single line fits them - but 45 s
    of the offer matched nothing, and reporting one run across that would credit
    footage nobody owns. They are split instead, which also makes the count of
    matched stretches mean what a buyer assumes it means.
    """
    cluster_a = [(t, t) for t in grid(15)]
    cluster_b = [(t + 60000, t + 60000) for t in grid(15)]
    runs = HoughChainMatcher().find_runs(*_hits(cluster_a + cluster_b))

    assert len(runs) == 2, [(r.q_start_ms, r.q_end_ms) for r in runs]
    spanned = sum(r.q_span_ms for r in runs)
    assert spanned <= 40_000, "the 45 s hole is being counted somewhere"
    for r in runs:
        assert r.covered_ms
        assert sum(b - a for a, b in r.covered_ms) == r.covered_ms_total
        # No individual run may straddle the hole.
        assert r.q_span_ms < 30_000, r.q_span_ms


def test_a_narrow_hole_stays_one_run_but_is_not_credited() -> None:
    """Below the split threshold, covered_ms is what keeps the accounting honest.

    Short dropouts inside a genuine match are normal - a few unusable frames, a
    brief occlusion - and splitting on those would shred real runs below the
    evidence floor. They stay one run, and the gap is excluded from the matched
    total rather than from the run.

    Two thresholds are in play and they are not the same number: a gap of at
    least min_gap_ms (5 s) is excluded from covered_ms, while a split needs
    max(min_gap_ms, 8000/fps), which at 1 fps is 8 s. This uses a 6 s gap, which
    sits between them. At 4 fps the two coincide and this window closes.
    """
    before = [(t, t) for t in grid(12)]
    after = [(t + 18_000, t + 18_000) for t in grid(12)]
    gap_ms = after[0][0] - before[-1][0]
    assert gap_ms - 1_000 > 5_000 and gap_ms < 8_000, (
        f'the {gap_ms} ms gap must sit between the covering and splitting thresholds'
    )
    runs = HoughChainMatcher().find_runs(*_hits(before + after))

    assert len(runs) == 1, [(r.q_start_ms, r.q_end_ms) for r in runs]
    run = runs[0]
    assert run.q_span_ms > 25_000  # the span reaches across the short gap
    assert run.covered_ms_total < run.q_span_ms  # but the gap is not credited
    assert sum(b - a for a, b in run.covered_ms) == run.covered_ms_total


def test_continuous_run_covers_its_whole_span() -> None:
    times = grid(30)
    runs = HoughChainMatcher().find_runs(*_hits([(t, t + 5000) for t in times]))
    assert len(runs) == 1
    assert runs[0].covered_ms_total == runs[0].q_span_ms


def test_min_inliers_configurable() -> None:
    times = grid(6)
    hits = _hits([(t, t) for t in times])
    assert HoughChainMatcher().find_runs(*hits) == []
    lax = HoughChainMatcher(ChainParams(min_inliers=4, min_votes=3))
    assert len(lax.find_runs(*hits)) == 1
