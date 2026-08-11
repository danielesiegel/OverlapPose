"""Two sellers offering the same footage, when the buyer owns neither.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from tests.fixtures import manipulations
from tests.fixtures.ffmpeg_factory import make_teleop_clip
from typer.testing import CliRunner

from overlap.cli import app

pytestmark = pytest.mark.integration

runner = CliRunner()


def _index(paths: list[Path], index_dir: Path, *, fps: float = 4.0) -> None:
    result = runner.invoke(
        app,
        ["--index", str(index_dir), "index", *[str(p) for p in paths],
         "--workers", "1", "--fps", str(fps)],
    )
    assert result.exit_code == 0, result.output


def _export(index_dir: Path, out: Path, *, stride: int = 1, label: str = "offer") -> None:
    result = runner.invoke(
        app,
        ["--index", str(index_dir), "export", "-o", str(out), "--label", label,
         "--stride", str(stride)],
    )
    assert result.exit_code == 0, result.output


def _compare_manifests(offer: Path, against: Path, out: Path) -> dict:
    result = runner.invoke(
        app, ["compare", str(offer), "--against", str(against), "-o", str(out)]
    )
    assert result.exit_code == 0, result.output
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def offers(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Two aggregators, one underlying dataset, plus a genuinely different one."""
    root = tmp_path_factory.mktemp("offers")
    master = root / "master"
    master.mkdir()
    for i, seed in enumerate((11, 23, 31)):
        make_teleop_clip(master / f"ep_{i}.mp4", duration=40.0, seed=seed)

    # Aggregator A ships it transcoded and renamed.
    agg_a = root / "agg-a"
    agg_a.mkdir()
    for i in range(3):
        manipulations.reencode(
            master / f"ep_{i}.mp4", agg_a / f"A-{i:03d}.mp4", codec="libx264", crf=30
        )

    # Aggregator B ships the same footage concatenated and re-cut on boundaries
    # that align with nothing, then transcoded to mkv. Pure repackaging.
    agg_b = root / "agg-b"
    agg_b.mkdir()
    listing = root / "concat.txt"
    listing.write_text(
        "".join(f"file '{(master / f'ep_{i}.mp4').as_posix()}'\n" for i in range(3)),
        encoding="utf-8",
    )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-an",
         "-f", "segment", "-segment_time", "37", "-reset_timestamps", "1",
         str(agg_b / "B-%03d.mkv")],
        check=True,
    )

    # Aggregator C offers unrelated footage: the control.
    agg_c = root / "agg-c"
    agg_c.mkdir()
    for i, seed in enumerate((71, 83)):
        make_teleop_clip(agg_c / f"C-{i:03d}.mp4", duration=40.0, seed=seed)

    for name in ("agg-a", "agg-b", "agg-c"):
        _index([root / name], root / f"{name}.ovl")
        _export(root / f"{name}.ovl", root / f"{name}.olm", label=name)
    return {"root": root}


def test_two_aggregators_selling_the_same_data_are_caught(offers: dict[str, Path]) -> None:
    """The headline case: same footage, different packaging, no pixels available."""
    root = offers["root"]
    report = _compare_manifests(
        root / "agg-a.olm", root / "agg-b.olm", root / "a-vs-b.json"
    )
    assert report["mode"] == "manifest-vs-manifest"
    assert report["summary"]["overlap_pct"] >= 60.0, report["summary"]
    assert report["summary"]["files_with_overlap"] == 3
    # The report must say what it could not check, since neither side had pixels.
    assert "crop" in report["coverage_note"].lower()
    assert report["against"]["files"] > 0


def test_unrelated_offers_do_not_collide(offers: dict[str, Path]) -> None:
    root = offers["root"]
    report = _compare_manifests(
        root / "agg-a.olm", root / "agg-c.olm", root / "a-vs-c.json"
    )
    assert report["summary"]["overlap_pct"] == 0.0, report["summary"]
    assert report["summary"]["files_with_overlap"] == 0


def test_comparison_is_symmetric(offers: dict[str, Path]) -> None:
    """Whichever manifest a lab happens to load first must not change the verdict."""
    root = offers["root"]
    forward = _compare_manifests(root / "agg-a.olm", root / "agg-b.olm", root / "f.json")
    reverse = _compare_manifests(root / "agg-b.olm", root / "agg-a.olm", root / "r.json")
    assert forward["summary"]["files_with_overlap"] == 3

    # Re-cutting 120 s into 37 s segments leaves a 9 s tail, which cannot produce
    # a run at the 10 s evidence floor. Every segment long enough to clear the
    # floor must match; the short tail is correctly not counted.
    long_enough = [
        f for f in reverse["files"] if f.get("duration_s", 0) >= 15.0
    ]
    matched = {f["relpath"] for f in reverse["files"] if f["matches"]}
    assert long_enough, reverse["files"]
    assert {f["relpath"] for f in long_enough} <= matched, (
        f"segments over 15 s that went unmatched: "
        f"{ {f['relpath'] for f in long_enough} - matched }"
    )
    # Both directions must agree this is substantially the same footage.
    assert abs(forward["summary"]["overlap_pct"] - reverse["summary"]["overlap_pct"]) < 25.0


def test_imported_manifest_becomes_part_of_the_corpus(offers: dict[str, Path]) -> None:
    """A declined offer, kept as fingerprints, screens the next one."""
    root = offers["root"]
    lab = root / "lab-from-imports.ovl"
    result = runner.invoke(
        app, ["--index", str(lab), "import", str(root / "agg-b.olm"), "--label", "declined-B"]
    )
    assert result.exit_code == 0, result.output

    # Importing the same manifest twice must not double-count it.
    before = json.loads(
        runner.invoke(app, ["--index", str(lab), "--json", "status"]).stdout
    )
    result = runner.invoke(app, ["--index", str(lab), "import", str(root / "agg-b.olm")])
    assert result.exit_code == 0, result.output
    after = json.loads(runner.invoke(app, ["--index", str(lab), "--json", "status"]).stdout)
    assert after["files_done"] == before["files_done"]
    assert after["frames"] == before["frames"]

    # Now aggregator A's offer is checkable against footage the lab never owned.
    out = root / "a-vs-imported.json"
    result = runner.invoke(app, ["--index", str(lab), "compare", str(root / "agg-a.olm"),
                                "-o", str(out)])
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["summary"]["files_with_overlap"] == 3, report["summary"]


def test_flip_is_caught_without_stored_mirrors(offers: dict[str, Path]) -> None:
    """A manifest ships one digest per frame, yet mirrored resale still matches.

    The mirror digest is derived from the identity digest, so flip detection
    survives into manifest-only comparison.
    """
    root = offers["root"]
    flipped_dir = root / "agg-d"
    flipped_dir.mkdir(exist_ok=True)
    for i in range(3):
        manipulations.hflip(root / "master" / f"ep_{i}.mp4", flipped_dir / f"D-{i:03d}.mp4")
    _index([flipped_dir], root / "agg-d.ovl")
    _export(root / "agg-d.ovl", root / "agg-d.olm", label="agg-d")

    report = _compare_manifests(
        root / "agg-d.olm", root / "agg-a.olm", root / "d-vs-a.json"
    )
    assert report["summary"]["files_with_overlap"] == 3, report["summary"]
    mirrored = [
        m
        for f in report["files"]
        for m in f["matches"]
        if m["mirrored"]
    ]
    assert mirrored, "flip was not reported even though the footage was mirrored"
