"""A split manifest must read back identically to an unsplit one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.fixtures.ffmpeg_factory import make_teleop_clip
from typer.testing import CliRunner

from overlap.cli import app
from overlap.errors import ManifestError
from overlap.store.manifest import PARTS_INDEX, read_manifest

pytestmark = pytest.mark.integration

runner = CliRunner()


@pytest.fixture(scope="module")
def indexed(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("split")
    data = root / "data"
    data.mkdir()
    for i, seed in enumerate((11, 23, 31, 43, 57)):
        make_teleop_clip(data / f"ep_{i}.mp4", duration=20.0, seed=seed)
    index = root / "corpus.ovl"
    result = runner.invoke(
        app, ["--index", str(index), "index", str(data), "--workers", "1"]
    )
    assert result.exit_code == 0, result.output
    return {"root": root, "index": index, "data": data}


def _export(index: Path, out: Path, *extra: str) -> None:
    result = runner.invoke(
        app, ["--index", str(index), "export", "-o", str(out), "--stride", "1", *extra]
    )
    assert result.exit_code == 0, result.output


def test_split_reads_back_identically_to_one_file(indexed: dict[str, Path]) -> None:
    """The whole point: splitting must be a packaging choice, not a data change."""
    root, index = indexed["root"], indexed["index"]
    single = root / "one.ovlm"
    parts_dir = root / "parts"
    _export(index, single)
    # Five 20 s clips at 4 fps are ~2.7 KB of fingerprints each, so the part
    # budget has to be smaller than that to force a split.
    _export(index, parts_dir, "--split-gb", "0.000004")

    whole = read_manifest(single)
    joined = read_manifest(parts_dir)

    assert len(list(parts_dir.glob("part-*.ovlm"))) > 1, "did not actually split"
    assert joined.merkle_root == whole.merkle_root
    assert [f.relpath for f in joined.files] == [f.relpath for f in whole.files]
    assert [f.sha256 for f in joined.files] == [f.sha256 for f in whole.files]
    assert len(joined.streams) == len(whole.streams)
    for a, b in zip(joined.streams, whole.streams, strict=True):
        assert a.stream_key == b.stream_key
        assert a.file_idx == b.file_idx, "file indices were not re-based on join"
        assert a.hashes == b.hashes
        assert a.n_frames == b.n_frames


def test_each_part_is_a_manifest_on_its_own(indexed: dict[str, Path]) -> None:
    """So a part can be inspected, imported, or re-fetched without the rest."""
    parts_dir = indexed["root"] / "parts"
    parts = sorted(parts_dir.glob("part-*.ovlm"))
    assert parts
    for part in parts:
        m = read_manifest(part)
        assert m.files and m.streams
        # Every stream must point at a file present in this part.
        assert max(s.file_idx for s in m.streams) < len(m.files)
        for s in m.streams:
            assert len(s.hashes) == s.n_frames * 32


def test_the_index_pins_every_part_by_digest(indexed: dict[str, Path]) -> None:
    parts_dir = indexed["root"] / "parts"
    index = json.loads((parts_dir / PARTS_INDEX).read_text(encoding="utf-8"))
    assert index["schema"] == "manifest-parts/1"
    assert len(index["parts"]) == len(sorted(parts_dir.glob("part-*.ovlm")))
    assert index["hours"] > 0

    # A tampered part must not read.
    victim = parts_dir / str(index["parts"][0]["name"])
    original = victim.read_bytes()
    try:
        victim.write_bytes(original[:-64] + b"\x00" * 64)
        with pytest.raises(ManifestError, match="digest"):
            read_manifest(parts_dir)
    finally:
        victim.write_bytes(original)
    read_manifest(parts_dir)  # restored


def test_a_split_manifest_compares_like_any_other(indexed: dict[str, Path]) -> None:
    """End to end: the lab should not care how the sender packaged it."""
    root, index = indexed["root"], indexed["index"]
    report_single = root / "r-single.json"
    report_parts = root / "r-parts.json"
    for manifest, out in ((root / "one.ovlm", report_single), (root / "parts", report_parts)):
        result = runner.invoke(
            app, ["--index", str(index), "compare", str(manifest), "-o", str(out)]
        )
        assert result.exit_code == 0, result.output

    a = json.loads(report_single.read_text(encoding="utf-8"))
    b = json.loads(report_parts.read_text(encoding="utf-8"))
    assert a["summary"]["overlap_pct"] == b["summary"]["overlap_pct"]
    assert a["summary"]["files_with_overlap"] == b["summary"]["files_with_overlap"] == 5


def test_importing_a_part_directory_works(indexed: dict[str, Path]) -> None:
    root = indexed["root"]
    lab = root / "from-parts.ovl"
    result = runner.invoke(app, ["--index", str(lab), "import", str(root / "parts")])
    assert result.exit_code == 0, result.output
    status = json.loads(
        runner.invoke(app, ["--index", str(lab), "--json", "status"]).stdout
    )
    assert status["files_done"] == 5
