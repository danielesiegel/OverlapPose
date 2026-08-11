"""Run acceptance, tiers, confidence, and overlap accounting.

Tier semantics (documented in the README and report):

- ``exact`` - byte-identical file (sha256 join); never reaches the matcher
- ``strong`` - long, dense, close runs; safe to act on individually
- ``probable`` - meets all acceptance thresholds; the headline tier
- ``weak`` - borderline evidence; excluded from the headline number,
                 reported only when explicitly requested

Overlap percentages are computed on the *query* (offered) timeline via
interval union, so overlapping runs never double-count a second of footage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from overlap.match.chain import RunGeometry

TIER_ORDER = {"exact": 0, "strong": 1, "probable": 2, "weak": 3}


@dataclass
class ScoreParams:
    # mean_dist thresholds are calibrated to the measured hash drift of
    # temporally phase-misaligned copies on realistic footage (up to ~40 of
    # 256 bits at the worst 125 ms misalignment a 4 fps corpus grid allows);
    # random-pair distances concentrate at 128 +/- 8, so these acceptance
    # bands stay ~10 sigma away from noise.
    min_run_s: float = 10.0
    min_inliers: int = 8
    accept_density: float = 0.5
    accept_mean_dist: float = 42.0
    weak_density: float = 0.35
    weak_mean_dist: float = 50.0
    strong_run_s: float = 30.0
    strong_density: float = 0.7
    strong_mean_dist: float = 26.0
    # Structural evidence can stand in for per-frame crispness. A long, dense,
    # straight run is far stronger evidence than any single frame's distance:
    # unrelated footage does not produce 15-second diagonals at 60% density.
    # Blurry-but-structural matches are what temporal phase misalignment
    # produces - a corpus indexed at low fps cannot sample the same instants a
    # cut-at-arbitrary-time copy did, so every frame is a near-miss even though
    # the footage is identical. Without this, an 18 s run at 75% density was
    # discarded for a mean distance of 56.
    structural_run_s: float = 15.0
    structural_density: float = 0.6
    structural_mean_dist: float = 64.0
    slowdown_ratio: float = 1.1  # speed_ratio above this flags billable-hours inflation


def assign_tier(run: RunGeometry, p: ScoreParams | None = None) -> str | None:
    """Tier for a run, or None when the evidence does not clear the floor."""
    p = p or ScoreParams()
    dur_s = run.q_span_ms / 1000.0
    if dur_s < p.min_run_s or run.n_inliers < p.min_inliers:
        return None
    if (
        dur_s >= p.strong_run_s
        and run.density >= p.strong_density
        and run.mean_dist <= p.strong_mean_dist
    ):
        return "strong"
    # Long, dense runs earn a higher distance ceiling (see ScoreParams).
    structural = dur_s >= p.structural_run_s and run.density >= p.structural_density
    accept_ceiling = p.structural_mean_dist if structural else p.accept_mean_dist
    if run.density >= p.accept_density and run.mean_dist <= accept_ceiling:
        return "probable"
    weak_ceiling = p.structural_mean_dist if structural else p.weak_mean_dist
    if run.density >= p.weak_density and run.mean_dist <= weak_ceiling:
        return "weak"
    return None


def confidence(run: RunGeometry, p: ScoreParams | None = None) -> float:
    """Sorting/UI confidence in (0, 1). Not used for gating."""
    p = p or ScoreParams()
    dur_s = max(run.q_span_ms / 1000.0, 0.001)
    z_dur = math.log2(dur_s / p.min_run_s) if dur_s > 0 else -3.0
    z_density = (run.density - 0.5) / 0.15
    z_dist = (p.accept_mean_dist - run.mean_dist) / 6.0
    x = 0.9 * z_dur + 1.4 * z_density + 1.1 * z_dist
    value = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))
    return min(0.9999, max(0.0001, round(value, 4)))


def union_ms(intervals: list[tuple[int, int]]) -> int:
    """Total covered milliseconds of a set of [start, end] intervals."""
    if not intervals:
        return 0
    total = 0
    cur_start, cur_end = None, None
    for start, end in sorted(intervals):
        if cur_end is None or start > cur_end:
            if cur_end is not None and cur_start is not None:
                total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_end is not None and cur_start is not None:
        total += cur_end - cur_start
    return total
