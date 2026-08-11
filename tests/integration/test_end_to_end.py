"""Vendor <-> lab round trip over the real CLI surface."""

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
def scenario(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Lab owns clips A+B. Vendor offers: exact copy of A, a re-encoded trim
    of B, and unrelated footage C."""
    root = tmp_path_factory.mktemp("e2e")
    lab_data = root / "lab-data"
    vendor_data = root / "vendor-data"
    lab_data.mkdir()
    vendor_data.mkdir()

    clip_a = make_teleop_clip(lab_data / "episode_a.mp4", duration=40.0, seed=11)
    clip_b = make_teleop_clip(lab_data / "episode_b.mp4", duration=40.0, seed=23)

    (vendor_data / "offer_1.mp4").write_bytes(clip_a.read_bytes())  # exact dup
    trimmed = manipulations.trim(clip_b, root / "tmp_trim.mp4", start=5.0, duration=20.0)
    manipulations.reencode(trimmed, vendor_data / "offer_2.mp4", codec="libx264", crf=32)
    make_teleop_clip(vendor_data / "offer_3_new.mp4", duration=40.0, seed=7)  # novel

    return {"root": root, "lab_data": lab_data, "vendor_data": vendor_data}


def test_full_round_trip(scenario: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    root = scenario["root"]
    lab_index = root / "lab.ovl"
    vendor_index = root / "vendor.ovl"
    manifest = root / "offer.olm"
    report_json = root / "report.json"
    report_html = root / "report.html"

    # Vendor side: index + export manifest.
    monkeypatch.setenv("OVERLAP_INDEX", str(vendor_index))
    result = runner.invoke(app, ["index", str(scenario["vendor_data"]), "--workers", "1"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["export", "-o", str(manifest), "--label", "Q3 teleop offer"])
    assert result.exit_code == 0, result.output
    assert manifest.exists()
    # Manifests are compact: ~34 bytes/frame + metadata, so well under 100KB here.
    assert manifest.stat().st_size < 100_000

    # Lab side: index own corpus, compare the incoming manifest.
    monkeypatch.setenv("OVERLAP_INDEX", str(lab_index))
    result = runner.invoke(app, ["index", str(scenario["lab_data"]), "--workers", "1"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        app,
        ["compare", str(manifest), "-o", str(report_json), "--html", str(report_html)],
    )
    assert result.exit_code == 0, result.output

    report = json.loads(report_json.read_text(encoding="utf-8"))
    files = {f["relpath"]: f for f in report["files"]}

    assert files["offer_1.mp4"]["sha256_exact"] is True
    assert files["offer_1.mp4"]["overlap_pct"] == 100.0
    assert files["offer_2.mp4"]["overlap_pct"] >= 60.0, files["offer_2.mp4"]
    assert files["offer_2.mp4"]["matches"][0]["corpus_file"] == "episode_b.mp4"
    assert files["offer_3_new.mp4"]["overlap_pct"] == 0.0

    summary = report["summary"]
    assert 30.0 <= summary["overlap_pct"] <= 80.0
    assert summary["files_with_overlap"] == 2

    html = report_html.read_text(encoding="utf-8")
    assert "offer_2.mp4" in html
    assert "<script src=" not in html  # self-contained: no external assets

    # CI gating: overlap above threshold exits 3.
    result = runner.invoke(
        app, ["compare", str(manifest), "-o", str(report_json), "--fail-over", "10"]
    )
    assert result.exit_code == 3

    # Markdown rendering from the saved report.
    result = runner.invoke(app, ["report", str(report_json), "--format", "md"])
    assert result.exit_code == 0
    md = report_json.with_suffix(".md").read_text(encoding="utf-8")
    assert "% of offered footage matches" in md


def test_verify_delivery_roundtrip(
    scenario: dict[str, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = scenario["root"]
    manifest = root / "offer.olm"  # exported by the previous test
    assert manifest.exists()

    delivered = tmp_path / "delivered"
    delivered.mkdir()
    for p in scenario["vendor_data"].iterdir():
        (delivered / f"renamed_{p.name}").write_bytes(p.read_bytes())

    result = runner.invoke(app, ["--json", "verify", str(manifest), "--data", str(delivered)])
    assert result.exit_code == 0, result.output
    doc = json.loads(result.stdout)
    assert doc["ok"] is True
    assert doc["files_matched"] == 3

    # Tamper with one delivered file -> verification fails.
    victim = next(delivered.iterdir())
    victim.write_bytes(victim.read_bytes() + b"\x00")
    result = runner.invoke(app, ["--json", "verify", str(manifest), "--data", str(delivered)])
    assert result.exit_code == 1
    doc = json.loads(result.stdout)
    assert doc["ok"] is False
    assert doc["files_missing"] == 1


def test_self_dedupe_finds_internal_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "inventory"
    data.mkdir()
    original = make_teleop_clip(data / "take_one.mp4", duration=40.0, seed=11)
    manipulations.reencode(original, data / "take_one_reexport.mp4", codec="libx264", crf=30)
    make_teleop_clip(data / "fresh.mp4", duration=40.0, seed=43)

    index_dir = tmp_path / "inv.ovl"
    monkeypatch.setenv("OVERLAP_INDEX", str(index_dir))
    result = runner.invoke(app, ["index", str(data), "--workers", "1"])
    assert result.exit_code == 0, result.output

    out = tmp_path / "dupes.json"
    result = runner.invoke(app, ["self-dedupe", "-o", str(out)])
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text(encoding="utf-8"))
    dup_files = [f["relpath"] for f in report["files"] if f["overlap_pct"] > 0]
    assert "take_one.mp4" in dup_files or "take_one_reexport.mp4" in dup_files
    assert "fresh.mp4" not in dup_files
