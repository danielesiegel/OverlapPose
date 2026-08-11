"""Cross-format laundering: the same frames must match across containers.

The core anti-laundering property: fingerprints are computed on decoded
pixels, so an MCAP camera topic, a ROS1 bag, and an .mp4 of the same footage
all land on (nearly) the same hashes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.fixtures.containers import video_to_mcap, video_to_rosbag1

from overlap.ingest import index_paths
from overlap.match import compare_manifest_file
from overlap.readers import SamplePolicy, reader_for
from overlap.store.catalog import Catalog
from overlap.store.manifest import export_manifest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def mp4_corpus(base_clips, tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    """Lab corpus: the original clip as plain .mp4."""
    root = tmp_path_factory.mktemp("xfmt")
    index_dir = root / "corpus.ovl"
    stats = index_paths([Path(str(base_clips["a"]))], index_dir, workers=1)
    assert stats.indexed == 1
    return {"index_dir": index_dir, "root": root}


def test_mcap_reader_enumerates_topic(base_clips, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    mcap_file = video_to_mcap(Path(str(base_clips["a"])), tmp_path / "recording.mcap")
    reader = reader_for(mcap_file)
    assert reader is not None and reader.name == "mcap"
    with reader.open(mcap_file) as session:
        streams = session.streams()
        assert [s.stream_key for s in streams] == ["/cam_front/image/compressed"]
        info = streams[0]
        assert info.width == 320 and info.height == 180
        assert info.n_messages and info.n_messages > 400
        samples = list(session.sample(info.stream_key, SamplePolicy(fps=1.0)))
        assert len(samples) == pytest.approx(40, abs=2)
        assert samples[0].image.shape[:2] == (180, 320)


def test_mcap_launder_detected_against_mp4_corpus(base_clips, mp4_corpus, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """Vendor wraps the lab's footage into an MCAP recording -> still caught."""
    vendor_dir = tmp_path / "vendor"
    vendor_dir.mkdir()
    video_to_mcap(Path(str(base_clips["a"])), vendor_dir / "teleop_session.mcap")

    vendor_index = tmp_path / "vendor.ovl"
    stats = index_paths([vendor_dir], vendor_index, workers=1)
    assert stats.indexed == 1 and stats.errors == 0
    manifest = tmp_path / "vendor.olm"
    with Catalog.open(vendor_index) as cat:
        export_manifest(cat, manifest)

    report = compare_manifest_file(manifest, mp4_corpus["index_dir"])
    assert report["summary"]["overlap_pct"] >= 60.0, report["summary"]
    match = report["files"][0]["matches"][0]
    assert match["corpus_file"] == "clip_a.mp4"
    assert 0.9 <= match["speed_ratio"] <= 1.1


def test_rosbag1_launder_detected_against_mp4_corpus(
    base_clips, mp4_corpus, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("rosbags", reason="needs the overlap-cli[ros] extra")
    vendor_dir = tmp_path / "vendor"
    vendor_dir.mkdir()
    video_to_rosbag1(Path(str(base_clips["a"])), vendor_dir / "teleop_session.bag")

    vendor_index = tmp_path / "vendor.ovl"
    stats = index_paths([vendor_dir], vendor_index, workers=1)
    assert stats.indexed == 1 and stats.errors == 0, stats.failed_files
    manifest = tmp_path / "vendor.olm"
    with Catalog.open(vendor_index) as cat:
        export_manifest(cat, manifest)

    report = compare_manifest_file(manifest, mp4_corpus["index_dir"])
    assert report["summary"]["overlap_pct"] >= 60.0, report["summary"]
