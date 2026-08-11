"""Unreadable files are reported and skipped, never fatal to the run.

A large delivery reliably contains a few files that will not decode, so
aborting at file 90,000 would lose days of work.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.fixtures.ffmpeg_factory import make_teleop_clip
from typer.testing import CliRunner

from overlap.cli import app
from overlap.errors import ReaderError
from overlap.readers.video import VideoSession

pytestmark = pytest.mark.integration

runner = CliRunner()


@pytest.fixture(scope="module")
def truncated(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("badmedia")
    data = root / "data"
    data.mkdir()
    good = make_teleop_clip(data / "good.mp4", duration=12.0, seed=5)
    # Half a file, as an interrupted copy leaves behind.
    (data / "truncated.mp4").write_bytes(good.read_bytes()[: len(good.read_bytes()) // 2])
    (data / "empty.mp4").write_bytes(b"")
    (data / "garbage.mp4").write_bytes(b"not a container, just bytes" * 100)
    return {"root": root, "data": data}


def test_reader_raises_its_own_error_type(truncated: dict[str, Path]) -> None:
    """The reader's error type is the contract the pipeline handles."""
    for name in ("empty.mp4", "garbage.mp4"):
        with pytest.raises(ReaderError):
            VideoSession(truncated["data"] / name)


def test_index_reports_bad_files_and_keeps_the_good_one(truncated: dict[str, Path]) -> None:
    index_dir = truncated["root"] / "corpus.ovl"
    result = runner.invoke(
        app,
        ["--index", str(index_dir), "--json", "index", str(truncated["data"]), "--workers", "1"],
    )
    assert result.exit_code in (0, 4), result.output

    events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    files = {
        Path(e["path"]).name: e for e in events if e.get("event") == "file" and "path" in e
    }
    assert files["good.mp4"]["status"] == "done"
    for name in ("empty.mp4", "garbage.mp4"):
        assert files[name]["status"] == "error", files[name]
        # The message has to name a cause; the bug replaced it with the
        # reader's own missing attribute.
        assert "AVError" not in str(files[name].get("error", ""))

    # The good file is indexed and searchable regardless of its neighbours.
    result = runner.invoke(app, ["--index", str(index_dir), "--json", "status"])
    assert result.exit_code == 0, result.output
    doc = json.loads(result.stdout)
    assert doc["files_done"] == 1
    assert doc["frames"] > 0
