"""Merging slices must equal indexing the whole thing on one machine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.fixtures import manipulations
from tests.fixtures.ffmpeg_factory import make_teleop_clip
from typer.testing import CliRunner

from overlap.cli import app
from overlap.store.catalog import Catalog

pytestmark = pytest.mark.integration

runner = CliRunner()


@pytest.fixture(scope="module")
def scenario(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Two machines' worth of lab footage, and an offer overlapping both."""
    root = tmp_path_factory.mktemp("merge")
    for name in ("slice_a", "slice_b", "offer", "whole"):
        (root / name).mkdir()
    a = make_teleop_clip(root / "slice_a" / "ep_a.mp4", duration=30.0, seed=11)
    b = make_teleop_clip(root / "slice_b" / "ep_b.mp4", duration=30.0, seed=23)
    # The single-machine reference sees the same bytes in one place.
    (root / "whole" / "ep_a.mp4").write_bytes(a.read_bytes())
    (root / "whole" / "ep_b.mp4").write_bytes(b.read_bytes())

    manipulations.reencode(a, root / "offer" / "offer_a.mp4", codec="libx264", crf=30)
    manipulations.trim(b, root / "offer" / "offer_b.mp4", start=4.0, duration=20.0)
    make_teleop_clip(root / "offer" / "offer_new.mp4", duration=30.0, seed=97)
    return {"root": root}


def _index(paths: list[Path], index_dir: Path) -> None:
    result = runner.invoke(
        app, ["--index", str(index_dir), "index", *[str(p) for p in paths], "--workers", "1"]
    )
    assert result.exit_code == 0, result.output


def _compare(manifest: Path, index_dir: Path, out: Path) -> dict:
    result = runner.invoke(
        app, ["--index", str(index_dir), "compare", str(manifest), "-o", str(out)]
    )
    assert result.exit_code == 0, result.output
    return json.loads(out.read_text(encoding="utf-8"))


def _verdict(report: dict) -> object:
    return sorted(
        (
            f["relpath"],
            round(f["overlap_pct"], 3),
            sorted(Path(m["corpus_file"]).name for m in f["matches"]),
        )
        for f in report["files"]
    )


def test_merged_slices_match_a_single_machine_index(scenario: dict[str, Path]) -> None:
    root = scenario["root"]
    slice_a, slice_b = root / "a.ovl", root / "b.ovl"
    merged, whole, offer_index = root / "merged.ovl", root / "whole.ovl", root / "offer.ovl"

    _index([root / "slice_a"], slice_a)
    _index([root / "slice_b"], slice_b)
    _index([root / "whole"], whole)
    _index([root / "offer"], offer_index)

    manifest = root / "offer.olm"
    result = runner.invoke(
        app, ["--index", str(offer_index), "export", "-o", str(manifest)]
    )
    assert result.exit_code == 0, result.output

    result = runner.invoke(
        app, ["--index", str(merged), "merge", str(slice_a), str(slice_b)]
    )
    assert result.exit_code == 0, result.output

    with Catalog.open(merged) as cat, Catalog.open(whole) as ref:
        assert cat.stats().frames == ref.stats().frames
        assert cat.stats().streams == ref.stats().streams

    merged_report = _compare(manifest, merged, root / "merged.json")
    whole_report = _compare(manifest, whole, root / "whole.json")
    assert _verdict(merged_report) == _verdict(whole_report)
    # The comparison has to be finding something, or this proves nothing.
    assert merged_report["summary"]["files_with_overlap"] == 2


def test_merge_is_idempotent(scenario: dict[str, Path]) -> None:
    """Re-running after an interruption must not double-count footage."""
    root = scenario["root"]
    target = root / "twice.ovl"
    for _ in range(2):
        result = runner.invoke(
            app, ["--index", str(target), "merge", str(root / "a.ovl"), str(root / "b.ovl")]
        )
        assert result.exit_code == 0, result.output
    with Catalog.open(target) as cat, Catalog.open(root / "whole.ovl") as ref:
        assert cat.stats().frames == ref.stats().frames
        assert cat.stats().files_done == ref.stats().files_done


def test_merge_refuses_incomparable_fingerprints(scenario: dict[str, Path]) -> None:
    root = scenario["root"]
    foreign = root / "foreign.ovl"
    with Catalog.open(foreign, expected_meta={"algo_id": "pdq1", "prep_id": "prep-v1"}):
        pass
    result = runner.invoke(
        app, ["--index", str(root / "refuse.ovl"), "merge", str(root / "a.ovl"), str(foreign)]
    )
    assert result.exit_code == 1
    assert "not comparable" in result.output or "not comparable" in str(result.stderr)
