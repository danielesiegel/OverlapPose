"""Run density is a fraction of what was attainable, not of a dense corpus.

A query sampled 4x denser than the corpus cannot exceed 25% density, so
without this a perfect match is rejected before it is scored.
"""

from __future__ import annotations

import numpy as np

from overlap.match.chain import ChainParams, HoughChainMatcher


def _perfect_diagonal(seconds: float, corpus_fps: float, query_fps: float):  # type: ignore[no-untyped-def]
    """Hits for a flawless match: every corpus sample found by its query frame.

    The query grid is denser, so each corpus sample is claimed by the nearest
    query frame - the best a real comparison could possibly do.
    """
    corpus_t = (np.arange(int(seconds * corpus_fps)) + 0.5) / corpus_fps
    q_ms = np.round(corpus_t * 1000.0).astype(np.int64)
    c_ms = q_ms.copy()
    dist = np.full(q_ms.size, 4, dtype=np.int32)
    return q_ms, c_ms, dist, np.zeros(q_ms.size, bool), np.zeros(q_ms.size, np.int16)


def test_sparse_corpus_against_dense_query_is_accepted() -> None:
    q, c, d, m, v = _perfect_diagonal(40.0, corpus_fps=1.0, query_fps=4.0)
    matcher = HoughChainMatcher(ChainParams(sample_fps=4.0, corpus_fps=1.0, min_inliers=8))
    runs = matcher.find_runs(q, c, d, m, v)
    assert runs, "a flawless match against a 4x sparser corpus was rejected"
    assert runs[0].density >= 0.9, runs[0].density


def test_unaware_density_would_have_rejected_it() -> None:
    """Pin the bug: without the corpus rate, the same hits fall under the gate."""
    q, c, d, m, v = _perfect_diagonal(40.0, corpus_fps=1.0, query_fps=4.0)
    naive = HoughChainMatcher(ChainParams(sample_fps=4.0, min_inliers=8))
    runs = naive.find_runs(q, c, d, m, v)
    assert not runs or runs[0].density < 0.30, (
        "this test no longer demonstrates the failure it guards against"
    )


def test_equal_density_is_unchanged() -> None:
    """The common case must behave exactly as before."""
    q, c, d, m, v = _perfect_diagonal(40.0, corpus_fps=4.0, query_fps=4.0)
    with_rate = HoughChainMatcher(
        ChainParams(sample_fps=4.0, corpus_fps=4.0, min_inliers=8)
    ).find_runs(q, c, d, m, v)
    without = HoughChainMatcher(ChainParams(sample_fps=4.0, min_inliers=8)).find_runs(
        q, c, d, m, v
    )
    assert with_rate and without
    assert round(with_rate[0].density, 6) == round(without[0].density, 6)


def test_slowed_copy_still_prices_the_corpus_rate() -> None:
    """A halved-speed copy covers half the corpus per query second.

    Density must account for the slope, or a slowdown - the manipulation this
    tool most wants to catch - looks half as dense as it is.
    """
    # Corpus at 2 fps; the copy runs at half speed, so query time is doubled.
    corpus_t = (np.arange(60) + 0.5) / 2.0
    c_ms = np.round(corpus_t * 1000.0).astype(np.int64)
    q_ms = c_ms * 2
    dist = np.full(q_ms.size, 6, dtype=np.int32)
    zeros_b = np.zeros(q_ms.size, bool)
    zeros_v = np.zeros(q_ms.size, np.int16)
    matcher = HoughChainMatcher(ChainParams(sample_fps=4.0, corpus_fps=2.0, min_inliers=8))
    runs = matcher.find_runs(q_ms, c_ms, dist, zeros_b, zeros_v)
    assert runs, "a slowed copy against a sparser corpus was rejected"
    assert runs[0].density >= 0.9, runs[0].density
    assert 1.8 <= runs[0].speed_ratio <= 2.2, runs[0].speed_ratio
