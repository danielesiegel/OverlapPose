"""The shared sample must be able to prove something about the manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.fixtures import manipulations
from tests.fixtures.ffmpeg_factory import make_teleop_clip
from typer.testing import CliRunner

from overlap.cli import app

pytestmark = pytest.mark.integration

runner = CliRunner()


@pytest.fixture(scope="module")
def seller(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A seller's catalogue, a truthful sample from it, and a misleading one."""
    root = tmp_path_factory.mktemp("audit")
    offered = root / "offered"
    offered.mkdir()
    for i, seed in enumerate((11, 23, 31, 43)):
        make_teleop_clip(offered / f"ep_{i}.mp4", duration=30.0, seed=seed)

    # A real sample: two of the offered episodes, transcoded for delivery as a
    # seller would do rather than handing over masters.
    honest = root / "sample-honest"
    honest.mkdir()
    for i in (1, 2):
        manipulations.reencode(
            offered / f"ep_{i}.mp4", honest / f"sample_{i}.mp4", codec="libx264", crf=30
        )

    # A sample of footage that is not in the offer at all.
    dishonest = root / "sample-other"
    dishonest.mkdir()
    for i, seed in enumerate((91, 97)):
        make_teleop_clip(dishonest / f"other_{i}.mp4", duration=30.0, seed=seed)

    result = runner.invoke(
        app, ["--index", str(root / "seller.ovl"), "index", str(offered), "--workers", "1"]
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        app,
        ["--index", str(root / "seller.ovl"), "export", "-o", str(root / "offer.olm"),
         "--label", "Q3 offer"],
    )
    assert result.exit_code == 0, result.output
    return {"root": root}


def _audit(manifest: Path, sample: Path, out: Path) -> dict:
    result = runner.invoke(
        app,
        ["--json", "audit-sample", str(manifest), "--sample", str(sample),
         "-o", str(out), "--workers", "1"],
    )
    assert result.exit_code == 0, result.output
    return json.loads(out.read_text(encoding="utf-8"))


def test_a_truthful_sample_is_found_in_the_manifest(seller: dict[str, Path]) -> None:
    root = seller["root"]
    audit = _audit(root / "offer.olm", root / "sample-honest", root / "honest.json")
    assert audit["consistent"] is True, audit["sample_found_pct"]
    assert audit["sample_found_pct"] >= 90.0
    assert audit["unmatched_sample_files"] == []
    assert audit["sample_files"] == 2
    # The audit should say how little of the offer was actually seen.
    assert 0 < audit["sample_share_of_offer"] < 60


def test_a_sample_from_different_footage_is_flagged(seller: dict[str, Path]) -> None:
    """The check that makes the manifest worth trusting at all."""
    root = seller["root"]
    audit = _audit(root / "offer.olm", root / "sample-other", root / "other.json")
    assert audit["consistent"] is False
    assert audit["sample_found_pct"] == 0.0
    assert len(audit["unmatched_sample_files"]) == 2


def test_audit_reports_its_own_limits(seller: dict[str, Path]) -> None:
    """Under-claiming matters as much as over-claiming for a trust tool."""
    root = seller["root"]
    audit = _audit(root / "offer.olm", root / "sample-honest", root / "limits.json")
    note = audit["note"].lower()
    assert "cannot" in note and "verify" in note
    assert audit["manifest_hours"] > audit["sample_hours"]
