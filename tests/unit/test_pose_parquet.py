"""Pose parquet reader: probe, stream discovery, sampling, end-to-end index."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402

from overlap.errors import ReaderError  # noqa: E402
from overlap.hashing.sdq import SIGNAL_ALGO_ID, SIGNAL_PREP_ID  # noqa: E402
from overlap.readers import SamplePolicy, reader_for  # noqa: E402
from overlap.readers.pose_parquet import PoseParquetReader  # noqa: E402

RATE = 200  # Hz
SECONDS = 12
N = RATE * SECONDS


def _write_pose_parquet(path: Path, *, n: int = N, seed: int = 0, noise: float = 0.0) -> None:
    r = np.random.default_rng(seed)
    # Separate stream for the additive noise, so clean and noisy variants of
    # the same seed share identical underlying motion.
    rn = np.random.default_rng(seed + 1000)
    t_ms = (np.arange(n) * (1000 / RATE)).astype(np.int64)
    cols: dict[str, object] = {"time_ms": t_ms, "label": np.array(["walk"] * n)}
    tt = np.arange(n) / RATE
    for j in range(12):
        for ax in "xyz":
            f = 0.5 + 0.37 * j
            v = np.sin(2 * np.pi * f * tt + j) + 0.4 * np.cos(2 * np.pi * 2 * f * tt)
            # A smooth aperiodic drift so windows are unambiguous in time -
            # pure sinusoids are self-similar and let matchers latch wrong lags.
            drift = np.cumsum(r.normal(0, 1, n))
            v = v + 3.0 * drift / max(1.0, np.abs(drift).max())
            if noise:
                v = v + rn.normal(0, noise * v.std(), n)
            cols[f"p{j:02d}_joint_{ax}"] = v.astype(np.float32)
    # A sparse side-channel (like 5 Hz UWB fixes): must be excluded, not fatal.
    sparse = np.full(n, np.nan)
    sparse[::RATE] = 1.0
    cols["g00_fix_x"] = sparse
    pq.write_table(pa.table(cols), path)


def test_probe_and_reader_selection(tmp_path: Path) -> None:
    p = tmp_path / "pose.parquet"
    _write_pose_parquet(p)
    assert PoseParquetReader.probe(p)
    assert reader_for(p) is PoseParquetReader
    not_parquet = tmp_path / "fake.parquet"
    not_parquet.write_bytes(b"not a parquet file")
    assert not PoseParquetReader.probe(not_parquet)
    # A parquet without a time column is someone else's artifact: skipped.
    other = tmp_path / "labels.parquet"
    pq.write_table(pa.table({"name": ["a", "b"], "score": [1.0, 2.0]}), other)
    assert not PoseParquetReader.probe(other)


def test_streams_and_sampling(tmp_path: Path) -> None:
    p = tmp_path / "pose.parquet"
    _write_pose_parquet(p)
    with PoseParquetReader.open(p) as session:
        (info,) = session.streams()
        assert info.modality == "signal"
        assert info.stream_key == "proprio"
        assert info.width == 36  # 12 joints x 3 axes; the sparse channel dropped
        assert info.height == RATE
        assert info.duration_ms is not None and abs(info.duration_ms - SECONDS * 1000) < 100
        samples = list(session.sample("proprio", SamplePolicy(fps=4.0)))
    assert len(samples) == pytest.approx(SECONDS * 4, abs=3)
    t_ms, window = samples[10]
    assert window.shape[0] == 36
    assert window.shape[1] == pytest.approx(2 * RATE, abs=2)  # WINDOW_S at native rate
    assert t_ms == int(round((10 + 0.5) / 4.0 * 1000))
    # sp1 scaling happened: channels are median-centred, IQR-scaled.
    assert abs(float(np.median(window))) < 1.0


def test_locked_channels_floored_and_noise_robust(tmp_path: Path) -> None:
    """Mocap-rig shape (found on HiPHI BVH data): half the channels are locked
    joint axes carrying only quantization jitter. sp1 must scale-floor them -
    scaling by their ~zero IQR would amplify a noised copy's noise to full
    amplitude on channels with no motion, and the hashes would diverge; but
    the channel *set* must stay schema-determined so trims stay comparable."""
    from overlap.hashing.pdq_numpy import hamming
    from overlap.hashing.sdq import SdqKernel

    r = np.random.default_rng(4)
    n = RATE * 10
    tt = np.arange(n) / RATE

    def write(path: Path, noise: float) -> None:
        cols: dict[str, object] = {"time_ms": (np.arange(n) * (1000 / RATE)).astype(np.int64)}
        for j in range(20):
            v = np.sin(2 * np.pi * (0.4 + 0.3 * j) * tt + j)
            cols[f"m{j:02d}_move"] = (v + (r.normal(0, noise * v.std(), n) if noise else 0)).astype(
                np.float32
            )
        for j in range(20):  # locked axes: constant + float32-level jitter
            v = np.full(n, 37.5) + r.normal(0, 1e-5, n)
            cols[f"z{j:02d}_locked"] = (v + (r.normal(0, noise * 1e-5, n) if noise else 0)).astype(
                np.float32
            )
        pq.write_table(pa.table(cols), path)

    clean, noisy = tmp_path / "clean.parquet", tmp_path / "noisy.parquet"
    write(clean, 0.0)
    write(noisy, 0.1)
    kernel = SdqKernel()
    dists = []
    with PoseParquetReader.open(clean) as a, PoseParquetReader.open(noisy) as b:
        assert a.streams()[0].width == 40  # locked channels kept, scale-floored
        for (_, wa), (_, wb) in zip(
            a.sample("proprio", SamplePolicy(fps=4.0)),
            b.sample("proprio", SamplePolicy(fps=4.0)),
            strict=True,
        ):
            dists.append(hamming(kernel.hash_frame(wa).hash, kernel.hash_frame(wb).hash))
    assert float(np.mean(dists)) < 30, dists


def test_open_rejects_thin_files(tmp_path: Path) -> None:
    p = tmp_path / "thin.parquet"
    pq.write_table(
        pa.table({"time_ms": np.arange(100, dtype=np.int64), "a": np.ones(100)}), p
    )
    with pytest.raises(ReaderError, match="dense numeric channels"):
        PoseParquetReader.open(p)


def test_index_parquet_end_to_end(tmp_path: Path) -> None:
    """A parquet indexes through the real pipeline into the catalog with the
    signal identity, and a noisy copy of the same data stays hash-close."""
    from overlap.hashing.pdq_numpy import hamming
    from overlap.ingest.pipeline import index_paths
    from overlap.store.catalog import Catalog

    data = tmp_path / "data"
    data.mkdir()
    _write_pose_parquet(data / "session.parquet")
    _write_pose_parquet(data / "session_noisy.parquet", noise=0.05)
    stats = index_paths([data], tmp_path / "index", sample_fps=4.0, workers=1)
    assert stats.indexed == 2 and stats.errors == 0
    assert stats.streams == 2

    with Catalog.open(tmp_path / "index") as catalog:
        rows = list(catalog.iter_streams())
        assert len(rows) == 2
        for row in rows:
            assert catalog.stream_identity(row.stream_id) == (SIGNAL_ALGO_ID, SIGNAL_PREP_ID)
        (h0, _, q0, _), (h1, _, _, _) = (catalog.stream_hashes(r.stream_id) for r in rows)
        n = min(len(h0), len(h1)) // 32
        dists = [hamming(h0[i * 32 : i * 32 + 32], h1[i * 32 : i * 32 + 32]) for i in range(n)]
        assert float(np.mean(dists)) < 40, dists  # 5% noise: well inside radius
        assert max(q0) > 20  # moving signal is not flagged wholesale


def test_compare_detects_noisy_trimmed_copy(tmp_path: Path) -> None:
    """The lab-side flow on signal data: a copy with 10% Gaussian noise and a
    trimmed start still matches its source, reported uncropped at speed 1.0."""
    from overlap.ingest.pipeline import index_paths
    from overlap.match.compare import compare_manifest
    from overlap.store.catalog import Catalog
    from overlap.store.manifest import export_manifest, read_manifest

    n = RATE * 40
    corpus_data = tmp_path / "corpus"
    corpus_data.mkdir()
    _write_pose_parquet(corpus_data / "master.parquet", n=n)
    index_paths([corpus_data], tmp_path / "corpus_index", sample_fps=4.0, workers=1)

    offer_data = tmp_path / "offer"
    offer_data.mkdir()
    r = np.random.default_rng(9)
    src = pq.read_table(corpus_data / "master.parquet")
    trim = RATE * 5  # drop the first 5 s
    cols: dict[str, object] = {}
    for c in src.column_names:
        v = src.column(c).to_numpy(zero_copy_only=False)[trim:]
        if c.startswith("p"):
            v = v + r.normal(0, 0.1 * np.nanstd(v), len(v)).astype(v.dtype)
        if c == "time_ms":
            v = v - v[0]
        cols[c] = v
    pq.write_table(pa.table(cols), offer_data / "resold.parquet")
    index_paths([offer_data], tmp_path / "offer_index", sample_fps=4.0, workers=1)

    out = tmp_path / "offer.ovlm"
    with Catalog.open(tmp_path / "offer_index") as offer_catalog:
        export_manifest(offer_catalog, out)
    with Catalog.open(tmp_path / "corpus_index") as corpus_catalog:
        report = compare_manifest(read_manifest(out), corpus_catalog)
    (match_file,) = [f for f in report["files"] if f.get("matches")]
    assert match_file["overlap_pct"] > 80
    m = match_file["matches"][0]
    assert m["corpus_file"] == "master.parquet"
    assert m["crop_geometry"] == "uncropped"  # signal streams index no crops
    assert m["speed_ratio"] == pytest.approx(1.0, abs=0.02)
    assert m["c"][0] == pytest.approx(5.0, abs=1.0)  # the trim, localized


def test_export_signal_manifest(tmp_path: Path) -> None:
    """A pose-only manifest carries the signal identity in its header, so a
    consumer without sdq1 refuses it instead of mis-reading it as pdq2."""
    from overlap.ingest.pipeline import index_paths
    from overlap.store.catalog import Catalog
    from overlap.store.manifest import export_manifest, read_manifest

    data = tmp_path / "data"
    data.mkdir()
    _write_pose_parquet(data / "session.parquet")
    index_paths([data], tmp_path / "index", sample_fps=4.0, workers=1)
    out = tmp_path / "offer.ovlm"
    with Catalog.open(tmp_path / "index") as catalog:
        export_manifest(catalog, out)
    manifest = read_manifest(out)
    assert manifest.algo_id == SIGNAL_ALGO_ID
    assert manifest.prep_id == SIGNAL_PREP_ID
    assert manifest.total_frames > 0
