"""Compare orchestration: manifest vs local index -> report document.

Stages: sha256 join (exact) -> ANN candidates -> diagonal-run chaining ->
scoring -> report. The report is a plain JSON-serializable dict with schema
``report/1``; renderers (JSON, HTML, Markdown) all consume this one model.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

import faiss

from overlap.errors import IndexError_
from overlap.match.candidates import DEFAULT_PROBE_STRIDE, generate_candidates
from overlap.match.chain import ChainParams, HoughChainMatcher
from overlap.match.score import ScoreParams, assign_tier, confidence, union_ms
from overlap.store.annindex import AnnIndex
from overlap.store.catalog import Catalog
from overlap.store.ingest_manifest import manifest_as_corpus
from overlap.store.manifest import Manifest, build_manifest, read_manifest

REPORT_SCHEMA = "report/1"


def self_dedupe(
    index_dir: Path,
    *,
    min_run_s: float = 10.0,
    include_weak: bool = False,
    nprobe: int = 64,
    threads: int = 0,
    probe_stride: int = DEFAULT_PROBE_STRIDE,
    progress: Any = None,
) -> dict[str, Any]:
    """Compare a corpus against itself (vendors deduping their own inventory).

    The trivial self-diagonal (every stream matches itself at slope 1) is
    excluded; anything that remains is duplicated footage inside the corpus.
    Exact-duplicate accounting by sha256 is skipped for the file's own entry.
    """
    with Catalog.open(index_dir) as catalog:
        manifest, stream_ids = build_manifest(catalog)
        exclude = dict(enumerate(stream_ids))
        report = compare_manifest(
            manifest,
            catalog,
            min_run_s=min_run_s,
            include_weak=include_weak,
            nprobe=nprobe,
            threads=threads,
            probe_stride=probe_stride,
            exclude_same_stream=exclude,
            self_mode=True,
            progress=progress,
        )
    report["mode"] = "self-dedupe"
    return report


def compare_two_manifests(
    offer_path: Path,
    against_path: Path,
    *,
    min_run_s: float = 10.0,
    include_weak: bool = False,
    nprobe: int = 64,
    max_manifest_bytes: int | None = None,
    threads: int = 0,
    probe_stride: int = DEFAULT_PROBE_STRIDE,
    progress: Any = None,
) -> dict[str, Any]:
    """Compare two manifests directly, owning the pixels of neither.

    The case this exists for: two aggregators offer a lab the same footage, the
    lab has bought neither, so there is nothing local to compare against. One
    manifest is loaded into a throwaway index and the other is compared against
    it.

    Coverage is narrower than a comparison against indexed footage, and the
    report says so: neither side has pixels, so no crop geometries exist and a
    cropped copy of one offer will not be found in the other. Everything the
    plain fingerprint covers - re-encoding, container swaps, trims, splices,
    speed changes, mirroring, cross-format laundering - still applies.
    """
    emit = progress or (lambda _e: None)
    offer = read_manifest(offer_path, max_bytes=max_manifest_bytes)
    against = read_manifest(against_path, max_bytes=max_manifest_bytes)
    with tempfile.TemporaryDirectory(prefix="overlap-pair-") as tmp:
        catalog = manifest_as_corpus(against, Path(tmp) / "against.ovl")
        try:
            emit(
                {
                    "event": "stage",
                    "stage": "loaded",
                    "against_files": len(against.files),
                    "against_hours": round(against.total_hours, 2),
                }
            )
            report = compare_manifest(
                offer,
                catalog,
                min_run_s=min_run_s,
                include_weak=include_weak,
                nprobe=nprobe,
                threads=threads,
                probe_stride=probe_stride,
                progress=progress,
            )
        finally:
            catalog.close()
    report["mode"] = "manifest-vs-manifest"
    report["against"] = {
        "path": str(against_path),
        "label": against.label,
        "files": len(against.files),
        "hours": round(against.total_hours, 2),
        "sample_fps": against.sample_fps,
    }
    # Density is the exporter's choice and it bounds what can be found, so it
    # belongs in the report rather than in a footnote.
    report["coverage_note"] = (
        "Neither side supplied pixels, so no crop geometries were available: a "
        "cropped copy would not be found. Both manifests were compared at their "
        "exported density "
        f"({offer.sample_fps:g} fps offered, {against.sample_fps:g} fps compared against); "
        "manifests strided below 4 fps lose recall against re-cut footage."
    )
    return report


def compare_manifest_file(
    manifest_path: Path,
    index_dir: Path,
    *,
    min_run_s: float = 10.0,
    include_weak: bool = False,
    nprobe: int = 64,
    max_manifest_bytes: int | None = None,
    threads: int = 0,
    probe_stride: int = DEFAULT_PROBE_STRIDE,
    progress: Any = None,
) -> dict[str, Any]:
    manifest = read_manifest(manifest_path, max_bytes=max_manifest_bytes)
    with Catalog.open(index_dir) as catalog:
        return compare_manifest(
            manifest,
            catalog,
            min_run_s=min_run_s,
            include_weak=include_weak,
            nprobe=nprobe,
            threads=threads,
            probe_stride=probe_stride,
            progress=progress,
        )


def compare_manifest(
    manifest: Manifest,
    catalog: Catalog,
    *,
    min_run_s: float = 10.0,
    include_weak: bool = False,
    nprobe: int = 64,
    exclude_same_stream: dict[int, int] | None = None,
    self_mode: bool = False,
    threads: int = 0,
    probe_stride: int = DEFAULT_PROBE_STRIDE,
    progress: Any = None,
) -> dict[str, Any]:
    """Compare an in-memory manifest against a catalog. Returns report/1 dict."""
    _check_compatibility(manifest, catalog)
    if threads > 0:
        # Applies to every FAISS search below, including the exact per-pair pass.
        faiss.omp_set_num_threads(threads)
    emit = progress or (lambda _e: None)
    score_params = ScoreParams(min_run_s=min_run_s)

    # Stage 0: exact duplicates via sha256 join. Meaningless in self mode,
    # where every file trivially matches itself.
    exact_file_idxs: set[int] = set()
    if not self_mode:
        corpus_by_sha: dict[bytes, dict[str, Any]] = {}
        for row in catalog.file_rows():
            if row["status"] == "done" and row["sha256"]:
                corpus_by_sha[bytes(row["sha256"])] = row
        exact_file_idxs = {i for i, f in enumerate(manifest.files) if f.sha256 in corpus_by_sha}
    emit({"event": "stage", "stage": "exact", "matches": len(exact_file_idxs)})

    # Stage 1 + 2 for everything else.
    ann = AnnIndex.build_or_load(catalog, progress=progress)
    emit({"event": "stage", "stage": "ann", "codes": ann.n_codes})
    candidates = generate_candidates(
        manifest,
        ann,
        catalog,
        nprobe=nprobe,
        probe_stride=probe_stride,
        exclude_same_stream=exclude_same_stream,
    )
    emit({"event": "stage", "stage": "candidates", "stream_pairs": len(candidates.hits)})

    corpus_files = {r["file_id"]: r for r in catalog.file_rows()}
    corpus_streams = {row.stream_id: row for row in catalog.iter_streams()}

    # Chain runs per stream pair.
    matches_by_file: dict[int, list[dict[str, Any]]] = {}
    # Weak-tier runs are never counted in the headline, but discarding them
    # silently turns real evidence into a bare 0%. They are carried through so
    # the report can say "weak evidence exists, look closer".
    weak_by_file: dict[int, list[dict[str, Any]]] = {}
    for (qs_idx, corpus_sid), hits in candidates.hits.items():
        stream = manifest.streams[qs_idx]
        if manifest.files and stream.file_idx in exact_file_idxs:
            continue  # already fully accounted as exact
        srow_for_params = corpus_streams.get(corpus_sid)
        matcher = HoughChainMatcher(
            ChainParams(
                sample_fps=stream.sample_fps,
                corpus_fps=srow_for_params.sample_fps if srow_for_params else None,
                min_inliers=8,
            )
        )
        for run in matcher.find_runs(
            hits.q_ms, hits.c_ms, hits.dist, hits.mirrored, hits.crop_variant
        ):
            tier = assign_tier(run, score_params)
            if tier is None:
                continue
            srow = corpus_streams.get(corpus_sid)
            frow = corpus_files.get(srow.file_id) if srow else None
            target = matches_by_file if tier != "weak" or include_weak else weak_by_file
            target.setdefault(stream.file_idx, []).append(
                {
                    "query_stream": stream.stream_key,
                    "corpus_file": str(frow["relpath"]) if frow else f"stream:{corpus_sid}",
                    "corpus_stream": srow.stream_key if srow else str(corpus_sid),
                    "q": [run.q_start_ms / 1000.0, run.q_end_ms / 1000.0],
                    "c": [run.c_start_ms / 1000.0, run.c_end_ms / 1000.0],
                    "covered": [[a / 1000.0, b / 1000.0] for a, b in run.covered_ms],
                    "speed_ratio": round(run.speed_ratio, 3),
                    "mirrored": run.mirrored,
                    "crop_pct": _variant_pct(ann, run.crop_variant),
                    "crop_geometry": _variant_label(ann, run.crop_variant),
                    "n_inliers": run.n_inliers,
                    "density": round(run.density, 3),
                    "mean_dist": round(run.mean_dist, 1),
                    "tier": tier,
                    "confidence": confidence(run, score_params),
                }
            )
    emit({"event": "stage", "stage": "chained", "files_with_matches": len(matches_by_file)})

    return _build_report(
        manifest,
        matches_by_file,
        weak_by_file,
        exact_file_idxs,
        candidates.indeterminate,
        score_params,
    )


def _check_compatibility(manifest: Manifest, catalog: Catalog) -> None:
    for key, manifest_value in (
        ("algo_id", manifest.algo_id),
        ("prep_id", manifest.prep_id),
    ):
        index_value = catalog.get_meta(key)
        if index_value is not None and index_value != manifest_value:
            raise IndexError_(
                f"manifest was fingerprinted with {key}={manifest_value!r} but this index "
                f"uses {key}={index_value!r}; both sides must run compatible overlap versions"
            )


def _variant_pct(ann: AnnIndex, variant_idx: int) -> float:
    v = ann.describe_variant(variant_idx)
    return v.pct if v is not None else 0.0


def _variant_label(ann: AnnIndex, variant_idx: int) -> str:
    v = ann.describe_variant(variant_idx)
    return v.label() if v is not None else "uncropped"


def _matched_segment_count(runs: list[dict[str, Any]], tol_s: float = 2.0) -> int:
    """How many separate stretches of owned footage the offer was built from.

    Both timelines matter. A file reassembled from two cuts of one master is
    *continuous in the offer* - the pieces sit end to end - while the jump
    shows up in the corpus timeline. Counting gaps on the offer's timeline
    alone therefore reports one segment for exactly the manipulation this is
    meant to expose, so a run only continues the previous segment when it is
    adjacent on *both* axes and comes from the same source.
    """
    if not runs:
        return 0
    ordered = sorted(runs, key=lambda m: m["q"][0])
    count = 1
    prev = ordered[0]
    for run in ordered[1:]:
        continues = (
            run["corpus_file"] == prev["corpus_file"]
            and abs(run["q"][0] - prev["q"][1]) <= tol_s
            and abs(run["c"][0] - prev["c"][1]) <= tol_s
        )
        if not continues:
            count += 1
        prev = run
    return count


def _file_duration_ms(manifest: Manifest, file_idx: int) -> int:
    durations = [s.duration_ms for s in manifest.streams if s.file_idx == file_idx]
    return max(durations, default=0)


def _build_report(
    manifest: Manifest,
    matches_by_file: dict[int, list[dict[str, Any]]],
    weak_by_file: dict[int, list[dict[str, Any]]],
    exact_file_idxs: set[int],
    indeterminate: dict[int, list[tuple[int, int]]],
    params: ScoreParams,
) -> dict[str, Any]:
    files_out: list[dict[str, Any]] = []
    total_offered_ms = 0
    matched_ms_by_tier: dict[str, float] = {
        "exact": 0.0,
        "strong": 0.0,
        "probable": 0.0,
        "weak": 0.0,
    }
    speed_adjusted_ms = 0.0
    slowdown_files: set[int] = set()
    spliced_files: set[int] = set()
    flipped_files: set[int] = set()
    cropped_files: set[int] = set()
    partial_files: set[int] = set()
    weak_only_files: set[int] = set()
    corpus_view: dict[str, set[str]] = {}

    for idx, mfile in enumerate(manifest.files):
        duration_ms = _file_duration_ms(manifest, idx)
        total_offered_ms += duration_ms
        entry: dict[str, Any] = {
            "relpath": mfile.relpath,
            "duration_s": round(duration_ms / 1000.0, 1),
            "sha256_exact": idx in exact_file_idxs,
            "overlap_pct": 0.0,
            "matches": [],
        }
        if idx in exact_file_idxs:
            entry["overlap_pct"] = 100.0
            matched_ms_by_tier["exact"] += duration_ms
            speed_adjusted_ms += duration_ms
        elif idx in matches_by_file:
            runs = sorted(matches_by_file[idx], key=lambda m: m["q"][0])
            entry["matches"] = runs
            # Covered stretches, not the outer span: a run with a hole in it
            # matched less time than it spans.
            intervals = [
                (int(a * 1000), int(b * 1000))
                for m in runs
                for a, b in (m.get("covered") or [m["q"]])
            ]
            matched = min(union_ms(intervals), duration_ms) if duration_ms else union_ms(intervals)
            entry["overlap_pct"] = round(100.0 * matched / duration_ms, 1) if duration_ms else 0.0
            best_tier = min(
                (m["tier"] for m in runs), key=lambda t: {"strong": 0, "probable": 1, "weak": 2}[t]
            )
            matched_ms_by_tier[best_tier] += matched
            for m in runs:
                q_span = (m["q"][1] - m["q"][0]) * 1000
                speed_adjusted_ms += q_span / max(m["speed_ratio"], 1e-6)
                if m["speed_ratio"] >= params.slowdown_ratio:
                    slowdown_files.add(idx)
                if m["mirrored"]:
                    flipped_files.add(idx)
                if m.get("crop_pct", 0.0) > 0.0:
                    cropped_files.add(idx)
                corpus_view.setdefault(m["corpus_file"], set()).add(mfile.relpath)

            # Concatenation accounting. Two disjoint matched runs mean the
            # offered file was assembled from separate pieces of owned
            # footage - the "split a master and sell it back as a new
            # session" trick - whether those pieces came from one corpus
            # file or several. Reporting only the multi-source case would
            # have missed single-master reassembly entirely.
            distinct_sources = sorted({m["corpus_file"] for m in runs})
            segments = _matched_segment_count(runs)
            entry["matched_sources"] = distinct_sources
            entry["matched_segments"] = segments
            if segments > 1 or len(distinct_sources) > 1:
                spliced_files.add(idx)
            if matched < duration_ms * 0.95:
                # Part of the offer did not match anything owned: either new
                # footage was appended, or the copy was trimmed.
                partial_files.add(idx)
        if idx in weak_by_file and idx not in matches_by_file and idx not in exact_file_idxs:
            weak_runs = sorted(weak_by_file[idx], key=lambda m: m["q"][0])
            weak_seconds = round(
                union_ms([(int(m["q"][0] * 1000), int(m["q"][1] * 1000)) for m in weak_runs])
                / 1000.0,
                1,
            )
            entry["weak_evidence"] = {
                "seconds": weak_seconds,
                "runs": weak_runs,
                "note": (
                    "below the reporting threshold and NOT counted in the overlap "
                    "percentage; re-run with --tier weak to include it"
                ),
            }
            weak_only_files.add(idx)
        q_streams = [i for i, s in enumerate(manifest.streams) if s.file_idx == idx]
        spans = [span for qs in q_streams for span in indeterminate.get(qs, [])]
        if spans:
            entry["indeterminate_spans"] = [
                [round(a / 1000.0, 1), round(b / 1000.0, 1)] for a, b in spans
            ]
        files_out.append(entry)

    matched_total_ms = sum(matched_ms_by_tier[t] for t in ("exact", "strong", "probable"))
    return {
        "schema": REPORT_SCHEMA,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "params": {"min_run_s": params.min_run_s},
        "manifest": {
            "label": manifest.label,
            "algo_id": manifest.algo_id,
            "merkle_root": manifest.merkle_root.hex(),
            "n_files": len(manifest.files),
        },
        "summary": {
            "offered_hours": round(total_offered_ms / 3.6e6, 3),
            "matched_hours": round(matched_total_ms / 3.6e6, 3),
            "overlap_pct": round(100.0 * matched_total_ms / total_offered_ms, 1)
            if total_offered_ms
            else 0.0,
            "speed_adjusted_matched_hours": round(speed_adjusted_ms / 3.6e6, 3),
            "by_tier_hours": {
                tier: round(ms / 3.6e6, 3) for tier, ms in matched_ms_by_tier.items()
            },
            "flags": {
                "slowdown_files": len(slowdown_files),
                "spliced_files": len(spliced_files),
                "flipped_files": len(flipped_files),
                "cropped_files": len(cropped_files),
                "partially_matched_files": len(partial_files),
                "weak_only_files": len(weak_only_files),
            },
            "files_offered": len(manifest.files),
            "files_with_overlap": len(exact_file_idxs) + len(matches_by_file),
        },
        "files": files_out,
        "corpus_view": [
            {"corpus_file": cf, "matched_by": sorted(qs)} for cf, qs in sorted(corpus_view.items())
        ],
    }
