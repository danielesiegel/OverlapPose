from __future__ import annotations

from overlap.match.chain import RunGeometry
from overlap.match.score import ScoreParams, assign_tier, confidence, union_ms


def run(
    dur_s: float = 20.0,
    density: float = 0.8,
    mean_dist: float = 20.0,
    n: int = 20,
    slope: float = 1.0,
    crop_variant: int = 0,
) -> RunGeometry:
    return RunGeometry(
        q_start_ms=0,
        q_end_ms=int(dur_s * 1000),
        c_start_ms=0,
        c_end_ms=int(dur_s * 1000 * slope),
        slope=slope,
        mirrored=False,
        crop_variant=crop_variant,
        n_inliers=n,
        density=density,
        mean_dist=mean_dist,
    )


def test_crop_variant_recorded_on_run() -> None:
    assert run().crop_variant == 0
    assert run(crop_variant=3).crop_variant == 3


def test_structural_ceiling_accepts_long_dense_blurry_runs() -> None:
    """A phase-misaligned copy of identical footage is blurry per frame but
    unmistakable structurally; judging it on distance alone lost real matches."""
    blurry_long = run(dur_s=18.0, density=0.75, mean_dist=56.0, n=27)
    assert assign_tier(blurry_long) == "probable"
    # the same distance without the structure stays rejected
    blurry_short = run(dur_s=11.0, density=0.45, mean_dist=56.0, n=11)
    assert assign_tier(blurry_short) is None


def test_tier_probable() -> None:
    assert assign_tier(run()) == "probable"


def test_tier_strong_needs_duration_density_distance() -> None:
    assert assign_tier(run(dur_s=35.0, density=0.8, mean_dist=20.0)) == "strong"
    assert assign_tier(run(dur_s=25.0, density=0.8, mean_dist=20.0)) == "probable"
    assert assign_tier(run(dur_s=35.0, density=0.6, mean_dist=20.0)) == "probable"


def test_tier_weak_band() -> None:
    # short runs (below the structural bar) fall back to the tight ceilings
    assert assign_tier(run(dur_s=12.0, density=0.4, n=5)) is None  # too few inliers
    assert assign_tier(run(dur_s=12.0, density=0.4)) == "weak"
    assert assign_tier(run(dur_s=12.0, mean_dist=46.0)) == "weak"


def test_below_floor_rejected() -> None:
    assert assign_tier(run(dur_s=5.0)) is None
    assert assign_tier(run(n=4)) is None
    assert assign_tier(run(density=0.2)) is None
    # a short run gets no structural allowance, so 55 bits is still too far
    assert assign_tier(run(dur_s=12.0, mean_dist=55.0)) is None
    # and nothing survives beyond the structural ceiling either
    assert assign_tier(run(dur_s=40.0, density=0.9, mean_dist=70.0)) is None


def test_confidence_orders_sensibly() -> None:
    good = confidence(run(dur_s=60.0, density=0.95, mean_dist=8.0))
    poor = confidence(run(dur_s=11.0, density=0.5, mean_dist=41.0))
    assert 0.0 < poor < good < 1.0


def test_speed_ratio_from_slope() -> None:
    slowed = run(slope=0.5)  # query runs twice as long as corpus footage
    assert abs(slowed.speed_ratio - 2.0) < 1e-9


def test_union_ms_merges_overlaps() -> None:
    assert union_ms([]) == 0
    assert union_ms([(0, 1000)]) == 1000
    assert union_ms([(0, 1000), (500, 1500)]) == 1500
    assert union_ms([(0, 1000), (2000, 3000)]) == 2000
    assert union_ms([(2000, 3000), (0, 1000), (900, 2100)]) == 3000


def test_custom_min_run() -> None:
    p = ScoreParams(min_run_s=5.0)
    assert assign_tier(run(dur_s=6.0), p) == "probable"
