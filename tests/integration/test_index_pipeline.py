from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from overlap.cli import app
from overlap.ingest import index_paths
from overlap.store.catalog import Catalog

pytestmark = pytest.mark.integration

runner = CliRunner()


def test_index_two_clips(base_clips, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    index_dir = tmp_path / "idx.ovl"
    stats = index_paths([base_clips["dir"]], index_dir, workers=1)
    assert stats.indexed == 2
    assert stats.errors == 0
    assert stats.streams == 2
    # 40 s at the default 4 fps grid = 160 samples per clip
    assert stats.frames == 320

    with Catalog.open(index_dir) as cat:
        s = cat.stats()
        assert s.files_done == 2
        assert s.frames == 320
        rows = list(cat.iter_streams())
        assert len(rows) == 2
        hashes, mirrors, qualities, flags = cat.stream_hashes(rows[0].stream_id)
        assert len(hashes) == rows[0].n_frames * 32
        assert len(mirrors) == len(hashes)
        assert len(qualities) == rows[0].n_frames
        assert len(flags) == rows[0].n_frames
        # synthetic testsrc2/mandelbrot frames are feature-rich
        assert max(qualities) == 100


def test_resume_skips_done_files(base_clips, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    index_dir = tmp_path / "idx.ovl"
    first = index_paths([base_clips["dir"]], index_dir, workers=1)
    assert first.indexed == 2
    second = index_paths([base_clips["dir"]], index_dir, workers=1)
    assert second.indexed == 0
    assert second.skipped == 2
    third = index_paths([base_clips["dir"]], index_dir, workers=1, reindex=True)
    assert third.indexed == 2


def test_identical_content_different_names_share_sha256(base_clips, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    src = Path(str(base_clips["a"]))
    copy = data_dir / "renamed_copy.mp4"
    copy.write_bytes(src.read_bytes())
    index_dir = tmp_path / "idx.ovl"
    index_paths([src, copy], index_dir, workers=1)
    with Catalog.open(index_dir) as cat:
        rows = cat.file_rows()
        assert len(rows) == 2
        assert rows[0]["sha256"] == rows[1]["sha256"]


def test_corrupt_file_reports_error_not_crash(base_clips, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "broken.mp4").write_bytes(b"this is not a video" * 100)
    good = Path(str(base_clips["a"]))
    (data_dir / "good.mp4").write_bytes(good.read_bytes())
    stats = index_paths([data_dir], tmp_path / "idx.ovl", workers=1)
    assert stats.indexed == 1
    assert stats.errors == 1
    assert stats.exit_partial


def test_parallel_workers_produce_same_result(base_clips, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    seq_dir = tmp_path / "seq.ovl"
    par_dir = tmp_path / "par.ovl"
    index_paths([base_clips["dir"]], seq_dir, workers=1)
    index_paths([base_clips["dir"]], par_dir, workers=2)
    with Catalog.open(seq_dir) as a, Catalog.open(par_dir) as b:
        rows_a = {r.stream_id: a.stream_hashes(r.stream_id)[0] for r in a.iter_streams()}
        rows_b = {r.stream_id: b.stream_hashes(r.stream_id)[0] for r in b.iter_streams()}
        assert sorted(rows_a.values()) == sorted(rows_b.values())


def test_cli_index_json_events_and_status(base_clips, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    index_dir = tmp_path / "cli.ovl"
    monkeypatch.setenv("OVERLAP_INDEX", str(index_dir))
    result = runner.invoke(app, ["--json", "index", str(base_clips["dir"]), "--workers", "1"])
    assert result.exit_code == 0, result.output
    events = [json.loads(line) for line in result.stdout.strip().splitlines()]
    # A coverage preamble states what this configuration can detect and what it
    # costs, before any work happens; then the run itself.
    assert events[0]["event"] == "coverage"
    # The mild default: uncropped + one centred rung + one bottom rung, each
    # upright and mirrored.
    assert events[0]["codes_per_frame"] == 6
    assert events[0]["index_mb_per_hour"] > 0
    assert events[1]["event"] == "start"
    assert events[-1]["event"] == "summary"
    assert events[-1]["indexed"] == 2

    result = runner.invoke(app, ["--json", "status"])
    doc = json.loads(result.stdout)
    assert doc["files_done"] == 2
    assert doc["frames"] == 320
    assert doc["meta"]["algo_id"] == "pdq2"


def test_dry_run_lists_without_indexing(base_clips, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    index_dir = tmp_path / "dry.ovl"
    monkeypatch.setenv("OVERLAP_INDEX", str(index_dir))
    result = runner.invoke(app, ["index", str(base_clips["dir"]), "--dry-run"])
    assert result.exit_code == 0
    assert "clip_a.mp4" in result.stdout
    assert not (index_dir / "catalog.sqlite").exists()
