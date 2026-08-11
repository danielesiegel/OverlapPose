"""Diagonal-run detection over candidate hits.

Hits between one query stream and one corpus stream are points in the
(query_time, corpus_time) plane. Shared footage lies on a line c = a*q + b: the
slope is the speed ratio, several lines mean a splice, a partial line means a
trim. Peaks are found by a Hough vote over (log-slope, offset), then refit to
their inliers.

Rationale and the calibration behind each threshold: docs/architecture.md.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

# Slope grid: 37 log2-spaced bins covering speed ratios 0.125x .. 8x.
_SLOPES = 2.0 ** (np.arange(37) / 6.0 - 3.0)


@dataclass(frozen=True)
class RunGeometry:
    """One matched diagonal run between a query stream and a corpus stream."""

    q_start_ms: int
    q_end_ms: int
    c_start_ms: int
    c_end_ms: int
    slope: float  # c-seconds per q-second
    mirrored: bool
    crop_variant: int  # index of the matched corpus crop variant (0 = uncropped)
    n_inliers: int  # distinct matched query samples (best hit per slot)
    density: float  # n_inliers / expected samples in the query span
    mean_dist: float  # mean of the best per-slot Hamming distances
    # Stretches of query time this run actually covers. A run with a hole in
    # the middle spans more time than it matched, and crediting the whole span
    # would over-report overlap, so accounting uses these, not q_start/q_end.
    covered_ms: tuple[tuple[int, int], ...] = ()

    @property
    def q_span_ms(self) -> int:
        return self.q_end_ms - self.q_start_ms

    @property
    def covered_ms_total(self) -> int:
        """Query milliseconds matched (never more than the span)."""
        if not self.covered_ms:
            return self.q_span_ms
        return sum(end - start for start, end in self.covered_ms)

    @property
    def speed_ratio(self) -> float:
        """Query seconds per corpus second. > 1 means the copy was slowed down."""
        return 1.0 / self.slope if self.slope > 0 else float("inf")


@dataclass
class ChainParams:
    sample_fps: float = 1.0
    # The corpus stream's own sampling rate. Density is the fraction of an
    # *attainable* match that was found, and neither side can produce more
    # matched moments than the sparser of the two supplies - so a dense query
    # against a sparse corpus must not be judged as if both were dense. None
    # means "same as the query", which is the common case.
    corpus_fps: float | None = None
    offset_bin_ms: int = 2000
    residual_ms: int = 1500
    min_votes: int = 6
    min_inliers: int = 8
    max_runs: int = 64
    # A candidate peak must look like a real diagonal before it is allowed to
    # consume hits. Without this gate a wrong-slope peak - which can outvote a
    # true one when hits are dense - swallows the hits a genuine parallel run
    # needed, so a file reassembled from two pieces of one master can report
    # no overlap even though a plain trim of the same master matches.
    min_density: float = 0.30
    # A run this dense is taken at face value. Below it, the run is *also* tested
    # for being a compromise slope smeared across separate segments, because a
    # peak can clear min_density and still be a smear: measured at density 0.34
    # on two 18 s pieces of one master whose true diagonals score 0.72 each.
    rescue_density: float = 0.60
    # A piece shorter than the scorer's evidence floor is not worth keeping, and
    # keeping it does real harm: its hits are consumed, so the genuine diagonal
    # they belong to can no longer be found.
    min_piece_ms: float = 10_000.0
    # Jump that separates two segments rather than one sparse run.
    min_gap_ms: float = 5000.0


class HoughChainMatcher:
    """Default temporal matcher. See module docstring for the algorithm."""

    def __init__(self, params: ChainParams | None = None) -> None:
        self.params = params or ChainParams()

    def find_runs(
        self,
        q_ms: np.ndarray,
        c_ms: np.ndarray,
        dist: np.ndarray,
        mirrored: np.ndarray,
        crop_variant: np.ndarray | None = None,
    ) -> list[RunGeometry]:
        p = self.params
        n = len(q_ms)
        if n < p.min_inliers:
            return []
        q = q_ms.astype(np.float64)
        c = c_ms.astype(np.float64)
        if crop_variant is None:
            crop_variant = np.zeros(n, dtype=np.int16)

        votes: Counter[tuple[int, int]] = Counter()
        for si, slope in enumerate(_SLOPES):
            bins = np.round((c - slope * q) / p.offset_bin_ms).astype(np.int64)
            for b in bins:
                votes[(si, int(b))] += 1
                votes[(si, int(b) - 1)] += 1
                votes[(si, int(b) + 1)] += 1

        consumed = np.zeros(n, dtype=bool)
        runs: list[RunGeometry] = []
        for (si, b_bin), count in votes.most_common():
            if count < p.min_votes or len(runs) >= p.max_runs:
                break
            slope = float(_SLOPES[si])
            offset = float(b_bin * p.offset_bin_ms)
            inliers = self._gather(q, c, consumed, slope, offset, p.residual_ms)
            if inliers.sum() < p.min_inliers:
                continue
            for _ in range(2):
                slope, offset = self._refit(q, c, inliers, slope)
                inliers = self._gather(q, c, consumed, slope, offset, p.residual_ms)
            if inliers.sum() < p.min_inliers:
                continue
            run = self._to_run(q, c, dist, mirrored, crop_variant, inliers, slope)
            if run.density >= p.rescue_density:
                consumed |= inliers
                runs.append(run)
                continue
            # Not convincingly dense, so this line may be a compromise slope
            # smeared across separate segments: a peak fitted between
            # two parallel diagonals can outvote either of them and then consume
            # the hits they needed. Try to recover the real structure, and keep
            # the split only if it explains the hits better than the smear did.
            pieces: list[tuple[RunGeometry, np.ndarray]] = []
            for group in self._split_on_gaps(q, c, inliers, slope):
                if group.sum() < p.min_inliers:
                    continue
                # Refit each piece. A compromise slope is wrong for every piece
                # it spans, so scoring one with it would misreport the speed
                # ratio and understate density.
                g_slope, _g_offset = self._refit(q, c, group, slope)
                sub = self._to_run(q, c, dist, mirrored, crop_variant, group, g_slope)
                if sub.density >= p.min_density and sub.q_span_ms >= p.min_piece_ms:
                    pieces.append((sub, group))
            if len(pieces) > 1:
                for sub, group in pieces:
                    consumed |= group
                    runs.append(sub)
                    if len(runs) >= p.max_runs:
                        break
            elif run.density >= p.min_density and run.q_span_ms >= p.min_piece_ms:
                consumed |= inliers
                runs.append(run)
            # Otherwise the peak is discarded *without consuming its hits*. That
            # matters: a compromise slope fitted between two parallel
            # diagonals can outvote either of them, and if it takes their hits
            # with it neither can ever be found. Leaving them free lets the peaks
            # that actually fit gather their own full diagonals.

        return _non_max_suppress(runs)

    @staticmethod
    def _gather(
        q: np.ndarray,
        c: np.ndarray,
        consumed: np.ndarray,
        slope: float,
        offset: float,
        residual_ms: int,
    ) -> np.ndarray:
        residual = np.abs(c - (slope * q + offset))
        mask: np.ndarray = (residual <= residual_ms) & ~consumed
        return mask

    def _split_on_gaps(
        self, q: np.ndarray, c: np.ndarray, inliers: np.ndarray, slope: float
    ) -> list[np.ndarray]:
        """Cut an inlier mask wherever *either* timeline jumps by more than a gap.

        Splitting on query time alone misses the case this matters most for. A
        file reassembled from two cuts of one master is continuous in the offer -
        the pieces sit end to end - and the jump appears only on the corpus
        timeline. The corpus-side threshold is scaled by the slope so a
        legitimately slowed copy, whose corpus steps are proportionally larger,
        is not chopped up for it.
        """
        idx = np.nonzero(inliers)[0]
        if idx.size == 0:
            return []
        order = idx[np.argsort(q[idx], kind="stable")]
        gap_ms = max(self.params.min_gap_ms, 8000.0 / self.params.sample_fps)
        jump_q = np.abs(np.diff(q[order])) > gap_ms
        jump_c = np.abs(np.diff(c[order])) > gap_ms * max(abs(slope), 1e-6)
        breaks = np.nonzero(jump_q | jump_c)[0]
        if breaks.size == 0:
            return [inliers]
        groups = []
        for chunk in np.split(order, breaks + 1):
            mask = np.zeros_like(inliers)
            mask[chunk] = True
            groups.append(mask)
        return groups

    @staticmethod
    def _refit(
        q: np.ndarray, c: np.ndarray, inliers: np.ndarray, fallback_slope: float
    ) -> tuple[float, float]:
        qi, ci = q[inliers], c[inliers]
        if len(qi) < 2 or float(np.ptp(qi)) < 1e-6:
            return fallback_slope, float(np.median(ci - fallback_slope * qi))
        slope, offset = np.polyfit(qi, ci, 1)
        # A fitted slope outside the physical grid means the fit chased noise.
        if not (float(_SLOPES[0]) / 1.5 <= slope <= float(_SLOPES[-1]) * 1.5):
            return fallback_slope, float(np.median(ci - fallback_slope * qi))
        return float(slope), float(offset)

    def _to_run(
        self,
        q: np.ndarray,
        c: np.ndarray,
        dist: np.ndarray,
        mirrored: np.ndarray,
        crop_variant: np.ndarray,
        inliers: np.ndarray,
        slope: float,
    ) -> RunGeometry:
        p = self.params
        qi, ci, di = q[inliers], c[inliers], dist[inliers]
        half = 500.0 / p.sample_fps  # half a sample interval on each side
        q_start, q_end = float(qi.min()) - half, float(qi.max()) + half
        c_lo, c_hi = float(ci.min()) - half * slope, float(ci.max()) + half * slope

        # A query frame legitimately hits several adjacent corpus samples (the
        # corpus is indexed densely); the honest per-moment statistic is the
        # BEST hit per distinct query slot - otherwise the far neighbors
        # inflate mean_dist and double-count density.
        unique_q, inverse = np.unique(qi, return_inverse=True)
        best = np.full(len(unique_q), 1 << 10, dtype=np.int32)
        np.minimum.at(best, inverse, di.astype(np.int32))
        n_slots = len(unique_q)
        covered = _cover_intervals(unique_q, half, self.params.min_gap_ms)

        # The crop geometry a run sits on is the one its closest hits agree
        # on: take the variant of the best hit in each query slot, then the
        # most common of those.
        best_pos = np.full(len(unique_q), -1, dtype=np.int64)
        order = np.argsort(-di.astype(np.int64), kind="stable")
        best_pos[inverse[order]] = order
        variants = crop_variant[inliers][best_pos]
        run_variant = int(np.bincount(variants.astype(np.int64)).argmax()) if len(variants) else 0
        # Corpus samples land on the query timeline at corpus_fps * slope: a copy
        # slowed to half speed covers half as much corpus per query second.
        corpus_rate = (p.corpus_fps if p.corpus_fps else p.sample_fps) * max(slope, 1e-9)
        attainable = min(p.sample_fps, corpus_rate)
        expected = max(1.0, (q_end - q_start) / 1000.0 * attainable)
        return RunGeometry(
            q_start_ms=int(q_start),
            q_end_ms=int(q_end),
            c_start_ms=int(max(0.0, c_lo)),
            c_end_ms=int(c_hi),
            slope=slope,
            mirrored=bool(np.mean(mirrored[inliers]) > 0.5),
            crop_variant=run_variant,
            n_inliers=n_slots,
            density=min(1.0, n_slots / expected),
            mean_dist=float(best.mean()),
            covered_ms=covered,
        )


def _cover_intervals(
    sample_times: np.ndarray, half: float, gap_ms: float
) -> tuple[tuple[int, int], ...]:
    """Contiguous stretches covered by matched samples, merging small holes."""
    if len(sample_times) == 0:
        return ()
    intervals: list[tuple[int, int]] = []
    start = float(sample_times[0]) - half
    end = float(sample_times[0]) + half
    for t in sample_times[1:]:
        t = float(t)
        if t - half - end > gap_ms:
            intervals.append((int(max(0.0, start)), int(end)))
            start = t - half
        end = t + half
    intervals.append((int(max(0.0, start)), int(end)))
    return tuple(intervals)


def _overlap_frac(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = min(a_end, b_end) - max(a_start, b_start)
    shorter = max(1, min(a_end - a_start, b_end - b_start))
    return max(0.0, inter / shorter)


def _non_max_suppress(runs: list[RunGeometry]) -> list[RunGeometry]:
    """Drop runs that re-explain the same query and corpus interval as a
    stronger run; keep genuine duplicates (same query region matching two
    different corpus regions)."""
    accepted: list[RunGeometry] = []
    for run in sorted(runs, key=lambda r: r.n_inliers, reverse=True):
        redundant = any(
            _overlap_frac(run.q_start_ms, run.q_end_ms, a.q_start_ms, a.q_end_ms) > 0.5
            and _overlap_frac(run.c_start_ms, run.c_end_ms, a.c_start_ms, a.c_end_ms) > 0.5
            for a in accepted
        )
        if not redundant:
            accepted.append(run)
    return sorted(accepted, key=lambda r: r.q_start_ms)
