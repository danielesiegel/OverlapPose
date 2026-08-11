"""A wrong-slope peak must not bury the diagonals it sits between.

Built from hit patterns directly, with no crop variants involved, so this
cannot come to depend on corpus configuration.
"""

from __future__ import annotations

import numpy as np

from overlap.match.chain import ChainParams, HoughChainMatcher


def _two_pieces_of_one_master() -> tuple[np.ndarray, ...]:
    """An offer built from corpus footage at 0-18 s and 36-54 s, joined end to end.

    The offer is continuous, so the discontinuity exists only on the corpus
    timeline - which is exactly why splitting on query-time gaps cannot see it.
    """
    q1 = np.arange(0, 18_000, 1000, dtype=np.int64)
    c1 = q1 + 2_000
    q2 = np.arange(18_000, 36_000, 1000, dtype=np.int64)
    c2 = (q2 - 18_000) + 36_000
    q = np.concatenate([q1, q2])
    c = np.concatenate([c1, c2])
    dist = np.full(q.size, 20, dtype=np.int32)
    return q, c, dist, np.zeros(q.size, bool), np.zeros(q.size, np.int16)


def test_both_segments_are_found() -> None:
    q, c, dist, mirrored, variant = _two_pieces_of_one_master()
    runs = HoughChainMatcher(
        ChainParams(sample_fps=1.0, corpus_fps=1.0, min_inliers=8)
    ).find_runs(q, c, dist, mirrored, variant)

    assert len(runs) >= 2, f"expected two segments, got {[r.q_span_ms for r in runs]}"
    # Both must be at true speed. A compromise slope would land near 2.0 here,
    # and would be reported to a buyer as a speed-changed copy.
    for r in runs[:2]:
        assert 0.95 <= r.slope <= 1.05, f"slope {r.slope} is a compromise, not the truth"
        assert r.q_span_ms >= 10_000, f"segment too short to clear the evidence floor: {r}"

    covered = sum(r.q_span_ms for r in runs)
    assert covered >= 30_000, f"only {covered / 1000:.1f}s of 36s accounted for"


def test_the_corpus_side_jump_is_what_separates_them() -> None:
    """Pin the axis. The offer is continuous, so query-gap splitting sees nothing."""
    q, c, _d, _m, _v = _two_pieces_of_one_master()
    assert np.diff(q).max() == 1000, "the offer must be continuous for this to be the case"
    assert np.diff(c).max() > 15_000, "the corpus timeline must be where the jump is"


def test_a_genuine_slowdown_is_not_chopped_up() -> None:
    """The corpus-side gap threshold scales with slope, so a slowed copy survives.

    A copy at half speed advances twice as far through the corpus per offer
    second. Treating that as a discontinuity would fragment the single most
    important manipulation this tool reports.
    """
    q = np.arange(0, 40_000, 500, dtype=np.int64)
    c = (q * 2).astype(np.int64)
    dist = np.full(q.size, 18, dtype=np.int32)
    runs = HoughChainMatcher(
        ChainParams(sample_fps=2.0, corpus_fps=2.0, min_inliers=8)
    ).find_runs(q, c, dist, np.zeros(q.size, bool), np.zeros(q.size, np.int16))

    assert runs, "a clean slowed copy produced no run at all"
    assert runs[0].q_span_ms >= 35_000, f"slowdown was fragmented: {runs[0].q_span_ms}ms"
    assert 0.45 <= runs[0].speed_ratio <= 0.55, runs[0].speed_ratio
