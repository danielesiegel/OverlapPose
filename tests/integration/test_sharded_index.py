"""Sharding must be invisible to results, and growth must be incremental.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from tests.fixtures import manipulations
from tests.fixtures.ffmpeg_factory import make_teleop_clip

from overlap.ingest.pipeline import index_paths
from overlap.store.annindex import AnnIndex
from overlap.store.catalog import Catalog
from overlap.store.manifest import build_manifest

pytestmark = pytest.mark.integration


def _fingerprint(report: dict) -> object:
    """The parts of a report a lab would act on."""
    return (
        round(report["summary"]["overlap_pct"], 6),
        report["summary"]["files_with_overlap"],
        sorted((f["relpath"], round(f["overlap_pct"], 6)) for f in report["files"]),
    )


def _shard_state(ann_dir: Path) -> dict[str, tuple[int, int]]:
    """Per shard, enough to prove it was not rewritten."""
    state: dict[str, tuple[int, int]] = {}
    for path in sorted(ann_dir.glob("shard-*.faiss")):
        payload = path.stat()
        mapping = path.with_suffix(".npz").stat()
        state[path.stem] = (payload.st_size + mapping.st_size, payload.st_mtime_ns)
    return state


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A lab corpus of four clips, plus an offer overlapping two of them."""
    root = tmp_path_factory.mktemp("shards")
    lab = root / "lab"
    offer = root / "offer"
    lab.mkdir()
    offer.mkdir()
    for i, seed in enumerate((11, 23, 31, 47)):
        make_teleop_clip(lab / f"ep_{i}.mp4", duration=30.0, seed=seed)
    # An offer built from footage the lab owns: one re-encode, one flip.
    manipulations.reencode(lab / "ep_0.mp4", offer / "offer_a.mp4", codec="libx264", crf=30)
    manipulations.hflip(lab / "ep_2.mp4", offer / "offer_b.mp4")
    make_teleop_clip(offer / "offer_c_new.mp4", duration=30.0, seed=99)
    return {"root": root, "lab": lab, "offer": offer}


@pytest.fixture(scope="module")
def indexes(corpus: dict[str, Path]) -> dict[str, Path]:
    lab_index = corpus["root"] / "lab.ovl"
    offer_index = corpus["root"] / "offer.ovl"
    index_paths([corpus["lab"]], lab_index, workers=1)
    index_paths([corpus["offer"]], offer_index, workers=1)
    return {"lab": lab_index, "offer": offer_index}


def _report(lab_index: Path, offer_index: Path) -> dict:
    from overlap.match.compare import compare_manifest

    with Catalog.open(offer_index) as offer_cat:
        manifest, _ = build_manifest(offer_cat)
    with Catalog.open(lab_index) as lab_cat:
        return compare_manifest(manifest, lab_cat)


def test_shard_count_does_not_change_the_answer(indexes: dict[str, Path]) -> None:
    """A many-shard index and a one-shard index must report the same overlap.

    This is the claim that makes the corpus size limit a disk question instead
    of a memory question, so it is asserted on the report a lab would read.
    """
    lab_index, offer_index = indexes["lab"], indexes["offer"]
    ann_dir = lab_index / "ann"

    shutil.rmtree(ann_dir, ignore_errors=True)
    with Catalog.open(lab_index) as cat:
        single = AnnIndex.build_or_load(cat, shard_codes=10**9)
        assert single.n_shards == 1
        total_codes = single.n_codes
    one_shard_report = _report(lab_index, offer_index)

    # A budget below one clip's worth of codes forces a shard per clip.
    shutil.rmtree(ann_dir, ignore_errors=True)
    with Catalog.open(lab_index) as cat:
        many = AnnIndex.build_or_load(cat, shard_codes=500)
        assert many.n_shards >= 4, "shard budget was not honored"
        assert many.n_codes == total_codes, "sharding lost or duplicated codes"
    many_shard_report = _report(lab_index, offer_index)

    assert _fingerprint(many_shard_report) == _fingerprint(one_shard_report)
    # And the answer has to be non-trivial, or this asserts nothing.
    assert one_shard_report["summary"]["files_with_overlap"] == 2


def test_reopening_an_unchanged_index_rebuilds_nothing(indexes: dict[str, Path]) -> None:
    lab_index = indexes["lab"]
    ann_dir = lab_index / "ann"
    shutil.rmtree(ann_dir, ignore_errors=True)
    with Catalog.open(lab_index) as cat:
        AnnIndex.build_or_load(cat, shard_codes=500)
    before = _shard_state(ann_dir)

    with Catalog.open(lab_index) as cat:
        events: list[dict] = []
        ann = AnnIndex.build_or_load(cat, shard_codes=500, progress=events.append)
    assert _shard_state(ann_dir) == before
    assert ann.n_shards == len(before)
    assert not [e for e in events if e.get("status") == "building"]


def test_added_footage_only_builds_new_shards(
    corpus: dict[str, Path], indexes: dict[str, Path]
) -> None:
    """The property that makes a 500 TB index maintainable rather than frozen."""
    lab_index = indexes["lab"]
    ann_dir = lab_index / "ann"
    shutil.rmtree(ann_dir, ignore_errors=True)
    with Catalog.open(lab_index) as cat:
        AnnIndex.build_or_load(cat, shard_codes=500)
    before = _shard_state(ann_dir)

    late = corpus["root"] / "late"
    late.mkdir(exist_ok=True)
    make_teleop_clip(late / "ep_late.mp4", duration=30.0, seed=71)
    index_paths([late], lab_index, workers=1)

    with Catalog.open(lab_index) as cat:
        ann = AnnIndex.build_or_load(cat, shard_codes=500)
    after = _shard_state(ann_dir)

    assert all(after[name] == state for name, state in before.items()), (
        "existing shards were rewritten; growth is not incremental"
    )
    assert len(after) > len(before)
    assert ann.n_shards == len(after)


def test_reindexed_file_invalidates_only_its_own_shard(
    corpus: dict[str, Path], indexes: dict[str, Path]
) -> None:
    lab_index = indexes["lab"]
    ann_dir = lab_index / "ann"
    shutil.rmtree(ann_dir, ignore_errors=True)
    # One stream per shard, so "only its own shard" is a sharp assertion.
    with Catalog.open(lab_index) as cat:
        AnnIndex.build_or_load(cat, shard_codes=1)
        assignments = cat.shard_assignments()
        target_sid = min(assignments)
        target_shard = assignments[target_sid][0]
    before = _shard_state(ann_dir)

    # Re-index one clip as a shorter file: the stream keeps its identity but
    # its frame count changes, which is what invalidation keys on.
    victim = next(
        Path(row[1]) for row in _files(lab_index) if Path(row[1]).name == "ep_0.mp4"
    )
    manipulations.trim(victim, corpus["root"] / "shorter.mp4", start=0.0, duration=15.0)
    shutil.copyfile(corpus["root"] / "shorter.mp4", victim)
    index_paths([corpus["lab"]], lab_index, workers=1)

    with Catalog.open(lab_index) as cat:
        AnnIndex.build_or_load(cat, shard_codes=1)
    after = _shard_state(ann_dir)

    survivors = {n for n in before if n in after and after[n] == before[n]}
    assert target_shard not in survivors, "the stale shard was not rebuilt"
    assert len(survivors) >= len(before) - 2, (
        f"re-indexing one file disturbed {len(before) - len(survivors)} shards"
    )


def _files(index_dir: Path) -> list[tuple[int, str, str]]:
    with Catalog.open(index_dir) as cat:
        return cat.files_by_status("done")
