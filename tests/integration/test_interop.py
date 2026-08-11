"""Interop: what two sides must agree on, and what they may choose freely.

The vendor and the lab configure their own indexes independently. This suite
pins which parts of that configuration are part of the *interchange contract*
(and enforced) versus a private tuning choice (and must never break a
comparison):

- ``algo_id`` / ``prep_id`` - the hash function and its normalization. These
  MUST match; comparing across them is meaningless, and it is refused.
- sampling rate and crop variants - private. Frame hashes depend on the frame,
  not on how often you sampled, and the matcher maps both sides onto real
  milliseconds. A denser corpus simply guarantees that whatever instants the
  vendor happened to sample, some corpus sample sits close to them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.fixtures.ffmpeg_factory import make_teleop_clip

from overlap.errors import IndexError_
from overlap.ingest import index_paths
from overlap.match import compare_manifest_file
from overlap.store.catalog import Catalog
from overlap.store.manifest import export_manifest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def shared(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("interop")
    data = root / "data"
    data.mkdir()
    make_teleop_clip(data / "master.mp4", duration=40.0, seed=11)
    corpus = root / "corpus4.ovl"
    index_paths([data], corpus, sample_fps=4.0, workers=1)
    return {"root": root, "data": data, "corpus4": corpus}


def _manifest_at(root: Path, data: Path, fps: float, name: str) -> Path:
    idx = root / f"vendor{name}.ovl"
    index_paths([data], idx, sample_fps=fps, workers=1)
    out = root / f"vendor{name}.olm"
    with Catalog.open(idx) as cat:
        export_manifest(cat, out)
    return out


@pytest.mark.parametrize("vendor_fps", [1.0, 2.0, 4.0, 8.0])
def test_vendor_sampling_rate_does_not_break_comparison(
    shared: dict[str, Path], vendor_fps: float
) -> None:
    """A vendor may index at any rate; the lab's 4 fps corpus absorbs it."""
    manifest = _manifest_at(shared["root"], shared["data"], vendor_fps, f"_{vendor_fps:g}")
    report = compare_manifest_file(manifest, shared["corpus4"])
    assert report["summary"]["overlap_pct"] >= 80.0, (
        f"a vendor manifest sampled at {vendor_fps} fps should still match a 4 fps "
        f"corpus; got {report['summary']['overlap_pct']}%"
    )


def test_corpus_crop_variants_are_private_to_the_lab(shared: dict[str, Path]) -> None:
    """Enabling extra corpus variants must not change how manifests compare."""
    manifest = _manifest_at(shared["root"], shared["data"], 2.0, "_priv")
    plain = compare_manifest_file(manifest, shared["corpus4"])

    edges = shared["root"] / "corpus-edges.ovl"
    index_paths([shared["data"]], edges, sample_fps=4.0, crop_edges="bottom", workers=1)
    with_edges = compare_manifest_file(manifest, edges)

    assert plain["summary"]["overlap_pct"] >= 80.0
    assert with_edges["summary"]["overlap_pct"] >= 80.0


def test_mismatched_prep_or_algo_is_refused(shared: dict[str, Path]) -> None:
    """The one thing that genuinely must match is refused loudly, not guessed."""
    manifest = _manifest_at(shared["root"], shared["data"], 4.0, "_algo")
    with Catalog.open(shared["corpus4"]) as cat:
        cat.set_meta("algo_id", "pdq-something-else")
    try:
        with pytest.raises(IndexError_, match="algo_id"):
            compare_manifest_file(manifest, shared["corpus4"])
    finally:
        with Catalog.open(shared["corpus4"]) as cat:
            cat.set_meta("algo_id", "pdq1")


def test_index_can_be_extended_with_different_settings(tmp_path: Path) -> None:
    """A lab may deepen coverage on new files without re-indexing everything.

    Per-stream settings are recorded, so a corpus can hold streams indexed
    with different rates and variant sets; refusing that would make enabling
    a new variant cost a full re-index of the archive.
    """
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()
    make_teleop_clip(first / "one.mp4", duration=20.0, seed=11)
    make_teleop_clip(second / "two.mp4", duration=20.0, seed=23)

    index_dir = tmp_path / "mixed.ovl"
    index_paths([first], index_dir, sample_fps=2.0, workers=1)
    stats = index_paths([second], index_dir, sample_fps=4.0, crop_edges="bottom", workers=1)
    assert stats.indexed == 1 and stats.errors == 0

    with Catalog.open(index_dir) as cat:
        rows = list(cat.iter_streams())
        assert {r.sample_fps for r in rows} == {2.0, 4.0}
        assert max(r.n_crop_rungs for r in rows) > min(r.n_crop_rungs for r in rows)

    # and the mixed corpus still answers a comparison
    manifest = tmp_path / "q.olm"
    vendor = tmp_path / "v.ovl"
    index_paths([first], vendor, sample_fps=1.0, workers=1)
    with Catalog.open(vendor) as cat:
        export_manifest(cat, manifest)
    report = compare_manifest_file(manifest, index_dir)
    assert report["summary"]["overlap_pct"] >= 80.0
