"""Manifest write/read roundtrip and untrusted-input hardening (no ffmpeg)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from overlap.errors import ManifestError
from overlap.ingest.model import FileResult, StreamResult
from overlap.store.catalog import Catalog
from overlap.store.manifest import export_manifest, read_manifest


def fake_stream(n_frames: int, key: str = "v0") -> StreamResult:
    return StreamResult(
        stream_key=key,
        codec="h264",
        width=320,
        height=180,
        native_fps=24.0,
        duration_ms=n_frames * 1000,
        sample_fps=1.0,
        algo_id="pdq1",
        prep_id="p1",
        border_crop="0,0,0,0",
        n_frames=n_frames,
        hashes=bytes(range(256))[:32] * n_frames,
        mirrors=bytes(reversed(range(256)))[-32:] * n_frames,
        qualities=bytes([90]) * n_frames,
        flags=bytes([0]) * n_frames,
        sketch=bytes(32),
    )


def fake_file(relpath: str, n_frames: int = 30) -> FileResult:
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
def catalog(tmp_path: Path) -> Catalog:
    cat = Catalog.open(
        tmp_path / "idx.ovl",
        expected_meta={
            "algo_id": "pdq1",
            "prep_id": "p1",
            "sample_fps": "1.0",
        },
    )
    cat.store_file_result(fake_file("b_second.mp4", 20))
    cat.store_file_result(fake_file("a_first.mp4", 30))
    yield cat
    cat.close()


def test_roundtrip(catalog: Catalog, tmp_path: Path) -> None:
    out = tmp_path / "m.olm"
    written = export_manifest(catalog, out, label="test batch")
    loaded = read_manifest(out)
    assert loaded.algo_id == "pdq1"
    assert loaded.label == "test batch"
    assert loaded.merkle_root == written.merkle_root
    assert [f.relpath for f in loaded.files] == ["a_first.mp4", "b_second.mp4"]
    assert loaded.total_frames == 50
    assert loaded.streams[0].hashes == written.streams[0].hashes
    assert loaded.streams[0].qualities == written.streams[0].qualities


def test_anonymize_paths_hides_names_keeps_root_stable(catalog: Catalog, tmp_path: Path) -> None:
    plain = export_manifest(catalog, tmp_path / "p.olm")
    anon = export_manifest(catalog, tmp_path / "a.olm", anonymize_paths=True)
    assert all(f.relpath.startswith("f") for f in anon.files)
    assert not any("first" in f.relpath for f in anon.files)
    # Roots differ because relpaths are part of the leaves - that is intended:
    # the root binds whatever names the manifest promises.
    assert anon.merkle_root != plain.merkle_root


def test_stride_halves_frames(catalog: Catalog, tmp_path: Path) -> None:
    m = export_manifest(catalog, tmp_path / "s.olm", stride=2)
    assert m.total_frames == 25
    assert m.streams[0].sample_fps == 0.5


def test_bad_magic_rejected(tmp_path: Path) -> None:
    bogus = tmp_path / "x.olm"
    bogus.write_bytes(b"NOPE" + bytes(100))
    with pytest.raises(ManifestError, match="bad magic"):
        read_manifest(bogus)


def test_truncated_manifest_rejected(catalog: Catalog, tmp_path: Path) -> None:
    out = tmp_path / "m.olm"
    export_manifest(catalog, out)
    data = out.read_bytes()
    out.write_bytes(data[: len(data) - 40])
    with pytest.raises(ManifestError):
        read_manifest(out)


def test_corrupted_section_rejected(catalog: Catalog, tmp_path: Path) -> None:
    out = tmp_path / "m.olm"
    export_manifest(catalog, out)
    data = bytearray(out.read_bytes())
    data[-10] ^= 0xFF  # flip a bit inside the frames section
    out.write_bytes(bytes(data))
    with pytest.raises(ManifestError, match="integrity"):
        read_manifest(out)


def test_size_cap_enforced(catalog: Catalog, tmp_path: Path) -> None:
    out = tmp_path / "m.olm"
    export_manifest(catalog, out)
    with pytest.raises(ManifestError, match="size limit"):
        read_manifest(out, max_bytes=10)
