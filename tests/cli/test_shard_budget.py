"""An index built without `overlap index` must still get a shard budget.

`index.shard_codes` is what bounds peak memory during a comparison: one shard
is resident at a time, and at the 32M default that is roughly 1.3 GB. The
budget is carried on the index itself, recorded in catalog metadata, so a later
comparison inherits it without every caller passing it along.

Only `overlap index` used to record it. An index assembled purely by `import`
(screening offers against a published dataset) or by `merge` (slices from
several machines) therefore had no budget recorded and silently fell back to
the 32M default, with no flag or environment variable able to change it -
which put exactly the "look something up on an ordinary machine" case out of
reach.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from overlap.cli import app
from overlap.ingest.model import FileResult, StreamResult
from overlap.store.catalog import Catalog
from overlap.store.manifest import export_manifest

runner = CliRunner()

DEFAULT_SHARD_CODES = 32_000_000


def fake_stream(n_frames: int) -> StreamResult:
    return StreamResult(
        stream_key="v0",
        codec="h264",
        width=320,
        height=180,
        native_fps=24.0,
        duration_ms=n_frames * 1000,
        sample_fps=1.0,
        algo_id="pdq2",
        prep_id="p1",
        border_crop="0,0,0,0",
        n_frames=n_frames,
        hashes=bytes(range(256))[:32] * n_frames,
        mirrors=bytes(reversed(range(256)))[-32:] * n_frames,
        qualities=bytes([90]) * n_frames,
        flags=bytes([0]) * n_frames,
        sketch=bytes(32),
    )


def fake_file(relpath: str, n_frames: int = 12) -> FileResult:
    return FileResult(
        abspath=f"C:/data/{relpath}",
        root="C:/data",
        relpath=relpath,
        size_bytes=1000,
        mtime_ns=1,
        sha256=hashlib.sha256(relpath.encode()).digest(),
        container="mp4",
        status="done",
        streams=[fake_stream(n_frames)],
    )


@pytest.fixture()
def manifest(tmp_path: Path) -> Path:
    """A manifest to import, built without touching ffmpeg."""
    source = tmp_path / "source.ovl"
    cat = Catalog.open(
        source,
        expected_meta={"algo_id": "pdq2", "prep_id": "p1", "sample_fps": "1.0"},
    )
    cat.store_file_result(fake_file("clip.mp4"))
    out = tmp_path / "offer.ovlm"
    export_manifest(cat, out, label="published catalog")
    cat.close()
    return out


def shard_budget(index_dir: Path) -> str | None:
    cat = Catalog.open(index_dir)
    try:
        return cat.get_meta("shard_codes")
    finally:
        cat.close()


def test_import_records_the_configured_shard_budget(
    manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_dir = tmp_path / "lab.ovl"
    monkeypatch.setenv("OVERLAP_INDEX_SHARD_CODES", "250000")

    result = runner.invoke(
        app, ["--index", str(index_dir), "import", str(manifest), "--no-shard"]
    )

    assert result.exit_code == 0, result.output
    assert shard_budget(index_dir) == "250000"


def test_import_records_the_default_when_nothing_is_configured(
    manifest: Path, tmp_path: Path
) -> None:
    """Recorded either way, so the budget is a property of the index."""
    index_dir = tmp_path / "lab.ovl"

    result = runner.invoke(
        app, ["--index", str(index_dir), "import", str(manifest), "--no-shard"]
    )

    assert result.exit_code == 0, result.output
    assert shard_budget(index_dir) == str(DEFAULT_SHARD_CODES)


def test_a_smaller_budget_survives_a_second_import(
    manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Importing block after block must not quietly restore the 32M default."""
    index_dir = tmp_path / "lab.ovl"
    monkeypatch.setenv("OVERLAP_INDEX_SHARD_CODES", "250000")

    for _ in range(2):
        result = runner.invoke(
            app, ["--index", str(index_dir), "import", str(manifest), "--no-shard"]
        )
        assert result.exit_code == 0, result.output

    assert shard_budget(index_dir) == "250000"


def test_merge_records_the_configured_shard_budget(
    manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A target assembled from slices has never been through `overlap index` either."""
    source = tmp_path / "slice.ovl"
    runner.invoke(app, ["--index", str(source), "import", str(manifest), "--no-shard"])

    target = tmp_path / "merged.ovl"
    monkeypatch.setenv("OVERLAP_INDEX_SHARD_CODES", "125000")
    result = runner.invoke(
        app, ["--index", str(target), "merge", str(source), "--no-shard"]
    )

    assert result.exit_code == 0, result.output
    assert shard_budget(target) == "125000"
