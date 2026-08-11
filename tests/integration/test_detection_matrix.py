"""THE executable detection matrix.

Every row of tests/detection_matrix.toml runs here against generated
fixtures. This suite is the contract behind every detection claim in the
README:

- tier = "detect"       -> the manipulated copy MUST match (hard failure)
- tier = "best-effort"  -> recorded via non-strict xfail, never blocks CI
- tier = "none"         -> the case MUST NOT match; if it starts matching,
the failure message says to promote the row - under-claiming is caught mechanically too.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from tests.fixtures import manipulations
from tests.fixtures.ffmpeg_factory import make_teleop_clip

from overlap.ingest import index_paths
from overlap.match import compare_manifest_file
from overlap.store.catalog import Catalog
from overlap.store.manifest import export_manifest

# The five-rung centred ladder ("balanced"), for rows testing crops deeper than
# the default reaches.
DEEP_LADDER = "0.94,0.88,0.82,0.76,0.70"

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

pytestmark = pytest.mark.integration

MATRIX_PATH = Path(__file__).parent.parent / "detection_matrix.toml"
CASES = tomllib.loads(MATRIX_PATH.read_text(encoding="utf-8"))["case"]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The lab side, indexed three ways so a row can state what it needs.

    The default is the "mild" preset - one centred rung, one bottom rung - so
    deeper coverage is opt-in and rows that rely on it say so via index_opts.
    Building all three here keeps that explicit instead of letting a row silently
    inherit whatever the default happens to be.
    """
    root = tmp_path_factory.mktemp("matrix")
    data = root / "corpus"
    data.mkdir()
    clip_a = make_teleop_clip(data / "corpus_clip.mp4", duration=60.0, seed=11)
    clip_b = make_teleop_clip(data / "corpus_clip_b.mp4", duration=60.0, seed=23)
    default_dir = root / "corpus.ovl"
    stats = index_paths([data], default_dir, workers=1)
    assert stats.indexed == 2 and stats.errors == 0
    edges_dir = root / "corpus-edges.ovl"
    stats = index_paths([data], edges_dir, crop_edges="bottom,top", workers=1)
    assert stats.indexed == 2 and stats.errors == 0
    deep_dir = root / "corpus-deep.ovl"
    stats = index_paths(
        [data], deep_dir, crop_ladder=DEEP_LADDER, crop_edges="", workers=1
    )
    assert stats.indexed == 2 and stats.errors == 0
    return {
        "index_dir": default_dir,
        "index_dir_edges": edges_dir,
        "index_dir_deep": deep_dir,
        "clip_a": clip_a,
        "clip_b": clip_b,
        "root": root,
    }


def _apply_transform(case: dict[str, Any], corpus: dict[str, Any], out_dir: Path) -> Path:
    transform = case["transform"]
    params = case.get("params", {})
    src: Path = corpus["clip_a"]
    ext = params.get("ext", ".mp4")
    dst = out_dir / f"{case['id']}{ext}"
    if transform == "copy":
        dst.write_bytes(src.read_bytes())
        return dst
    if transform == "unrelated":
        return make_teleop_clip(dst, duration=40.0, seed=47)
    if transform == "launder_mcap":
        from tests.fixtures.containers import video_to_mcap

        return video_to_mcap(src, dst.with_suffix(".mcap"))
    if transform == "merge_two_masters":
        pa = manipulations.trim(src, out_dir / "pa.mp4", start=5.0, duration=18.0)
        pb = manipulations.trim(corpus["clip_b"], out_dir / "pb.mp4", start=20.0, duration=18.0)
        result = manipulations.splice(pa, pb, dst, a_seconds=18.0)
        pa.unlink()
        pb.unlink()
        return result
    if transform == "merge_owned_plus_new":
        pa = manipulations.trim(src, out_dir / "pa.mp4", start=5.0, duration=18.0)
        fresh = make_teleop_clip(out_dir / "fresh.mp4", duration=18.0, seed=91)
        result = manipulations.splice(pa, fresh, dst, a_seconds=18.0)
        pa.unlink()
        fresh.unlink()
        return result
    if transform == "same_scene_other_view":
        # same background and rig, camera panning a different region
        return make_teleop_clip(dst, duration=40.0, seed=11, pan_start=1500)
    if transform == "merge_segments":
        return manipulations.merge_segments(
            src, dst, segments=[tuple(seg) for seg in params["segments"]]
        )
    if transform == "splice":
        other = make_teleop_clip(out_dir / f"{case['id']}-other.mp4", duration=25.0, seed=31)
        result = manipulations.splice(src, other, dst, a_seconds=params["a_seconds"])
        other.unlink()  # only the spliced file goes into the vendor set
        return result
    fn = getattr(manipulations, transform)
    kwargs = {k: v for k, v in params.items() if k != "ext"}
    return fn(src, dst, **kwargs)


def _compare_one(case: dict[str, Any], corpus: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    vendor_dir = tmp_path / "vendor"
    vendor_dir.mkdir()
    _apply_transform(case, corpus, vendor_dir)

    vendor_index = tmp_path / "vendor.ovl"
    stats = index_paths([vendor_dir], vendor_index, workers=1)
    assert stats.errors == 0, f"fixture for {case['id']} failed to index"
    manifest_path = tmp_path / "vendor.olm"
    with Catalog.open(vendor_index) as cat:
        export_manifest(cat, manifest_path)
    opts = case.get("index_opts", {})
    if opts.get("crop_edges"):
        corpus_key = "index_dir_edges"
    elif opts.get("crop_ladder"):
        # Rows asking for a ladder must get the deep one, not silently fall back
        # to the default and then fail as if the manipulation were undetectable.
        assert opts["crop_ladder"] == DEEP_LADDER, (
            f"{case['id']}: only {DEEP_LADDER!r} is prebuilt; add a corpus variant"
        )
        corpus_key = "index_dir_deep"
    else:
        corpus_key = "index_dir"
    return compare_manifest_file(manifest_path, corpus[corpus_key])


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_detection_matrix(
    case: dict[str, Any], corpus: dict[str, Any], tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    report = _compare_one(case, corpus, tmp_path)
    summary = report["summary"]
    tier = case["tier"]

    if tier == "none":
        assert summary["overlap_pct"] == 0.0, (
            f"matrix row {case['id']} is tiered 'none' but now DETECTS "
            f"({summary['overlap_pct']}% overlap). If this is robust, promote the row "
            f"in tests/detection_matrix.toml and regenerate the docs - the matrix "
            f"must never under-claim either."
        )
        return

    if tier == "best-effort" and summary["overlap_pct"] < case.get("min_overlap", 50):
        pytest.xfail(f"best-effort case {case['id']} not detected (allowed)")

    min_overlap = case.get("min_overlap", 50)
    assert summary["overlap_pct"] >= min_overlap, (
        f"matrix row {case['id']} (tier {tier}) fell below its floor: "
        f"{summary['overlap_pct']}% < {min_overlap}%"
    )
    if "max_overlap" in case:
        assert summary["overlap_pct"] <= case["max_overlap"], (
            f"matrix row {case['id']} over-reported: {summary['overlap_pct']}% > "
            f"{case['max_overlap']}% (only part of this offer is owned footage)"
        )

    if case.get("expect_segments"):
        matched = [f for f in report["files"] if f["overlap_pct"] > 0]
        assert matched, f"{case['id']}: nothing matched"
        assert matched[0]["matched_segments"] >= case["expect_segments"], (
            f"{case['id']}: expected >= {case['expect_segments']} matched segments, "
            f"got {matched[0]['matched_segments']}"
        )
        assert summary["flags"]["spliced_files"] >= 1, (
            f"{case['id']}: reassembly/concatenation was not flagged"
        )

    if case.get("expect_slowdown"):
        assert summary["flags"]["slowdown_files"] >= 1, f"{case['id']}: slowdown was not flagged"
        matches = [m for f in report["files"] for m in f["matches"]]
        assert any(m["speed_ratio"] > 1.5 for m in matches), (
            f"{case['id']}: speed_ratio not estimated (matches: {matches})"
        )

    if case.get("expect_mirrored"):
        assert summary["flags"]["flipped_files"] >= 1, f"{case['id']}: flip not flagged"

    if case.get("expect_cropped"):
        assert summary["flags"]["cropped_files"] >= 1, f"{case['id']}: crop not flagged"
        matches = [m for f in report["files"] for m in f["matches"]]
        params = case.get("params", {})
        # centered rows give a keep fraction, edge rows give the removed fraction
        expected_pct = (
            100.0 * params["frac"] if "frac" in params else 100.0 * (1.0 - params["keep"])
        )
        assert any(abs(m["crop_pct"] - expected_pct) <= 8.0 for m in matches), (
            f"{case['id']}: crop depth misestimated (expected ~{expected_pct:.0f}%, "
            f"got {[(m['crop_pct'], m['crop_geometry']) for m in matches]})"
        )
        if "side" in params:
            assert any(params["side"] in m["crop_geometry"] for m in matches), (
                f"{case['id']}: crop side misreported: {[m['crop_geometry'] for m in matches]}"
            )
