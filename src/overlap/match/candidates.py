"""Find which corpus streams an offer touches, then compare those exactly.

Runs as a cascade, because the accurate stage and the expensive stage need not
be the same one:

1. Sweep every ``probe_stride``-th query frame against the index. This only
   nominates pairs of streams, so it can be sparse and approximate.
2. Compare each nominated pair exactly, every query frame against that one
   corpus stream (:mod:`overlap.match.pairwise`).

Only the sweep grows with corpus size, and a pair needs just one matching frame
out of the ten the shortest acceptable run contains, so nothing is lost by
making it sparse: density, inliers and geometry are recovered in stage 2.

Both stages search the whole manifest at once, since the index is a set of
on-disk shards and searching stream by stream would re-read all of it per
stream.

Hub suppression: a corpus frame matched by an outsized share of one stream's
frames is static-scene noise, and is dropped rather than allowed to fabricate
runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from overlap.hashing.base import FLAG_LOW_QUALITY, HASH_BYTES
from overlap.match.pairwise import compare_stream_pair

if TYPE_CHECKING:
    from overlap.store.annindex import AnnIndex
    from overlap.store.catalog import Catalog
    from overlap.store.manifest import Manifest

# Permissive candidate radius; acceptance happens at run level. 56 covers the
# measured hash drift of a <=125 ms temporal phase shift on realistic footage
# (the worst case with a 4 fps corpus index), while random 256-bit pairs
# concentrate at 128 +/- 8 - still ~9 sigma away from false positives.
DEFAULT_RADIUS = 56
DEFAULT_K = 16
QUALITY_FLOOR = 20

# Every Nth query frame is swept against the index. The shortest run the scorer
# will accept is 10 s, which at the ~1 fps manifest grid is 10 frames, so a
# stride of 4 still puts 2-3 chances inside the shortest detectable match - and
# a pair needs only one. Set to 1 to sweep densely.
DEFAULT_PROBE_STRIDE = 4

# Query rows per corpus pass. 32M rows is ~1 GB of codes; with the stride above
# that is far more offered footage than any single delivery.
MAX_QUERY_BLOCK = 32_000_000


@dataclass
class HitSet:
    """All candidate hits between one query stream and one corpus stream."""

    q_ms: np.ndarray  # int64
    c_ms: np.ndarray  # int64
    dist: np.ndarray  # int32
    mirrored: np.ndarray  # bool
    crop_variant: np.ndarray  # int16; 0 = uncropped corpus variant matched


@dataclass
class CandidateResult:
    # (query_stream_index, corpus_stream_id) -> hits
    hits: dict[tuple[int, int], HitSet] = field(default_factory=dict)
    # per query stream: [start_ms, end_ms] spans that cannot be judged
    indeterminate: dict[int, list[tuple[int, int]]] = field(default_factory=dict)


@dataclass
class _QueryStream:
    """One manifest stream's usable frames, ready to search."""

    qs_idx: int
    codes: np.ndarray  # (n_usable, HASH_BYTES)
    times_ms: np.ndarray  # (n_usable,) int64
    sample_fps: float


def _grid_ms(n: int, fps: float) -> np.ndarray:
    return ((np.arange(n, dtype=np.float64) + 0.5) * 1000.0 / fps).astype(np.int64)


def _flag_spans(mask: np.ndarray, times_ms: np.ndarray, fps: float) -> list[tuple[int, int]]:
    """Contiguous True regions of ``mask`` as [start_ms, end_ms] spans."""
    spans: list[tuple[int, int]] = []
    half = int(500 / fps)
    start = None
    for i, flagged in enumerate(mask):
        if flagged and start is None:
            start = int(times_ms[i]) - half
        elif not flagged and start is not None:
            spans.append((start, int(times_ms[i - 1]) + half))
            start = None
    if start is not None:
        spans.append((start, int(times_ms[-1]) + half))
    return spans


def generate_candidates(
    manifest: Manifest,
    ann: AnnIndex,
    catalog: Catalog,
    *,
    k: int = DEFAULT_K,
    radius: int = DEFAULT_RADIUS,
    nprobe: int = 64,
    quality_floor: int = QUALITY_FLOOR,
    exclude_same_stream: dict[int, int] | None = None,
    probe_stride: int = DEFAULT_PROBE_STRIDE,
    max_query_block: int = MAX_QUERY_BLOCK,
) -> CandidateResult:
    """Nominate stream pairs by sweeping the index, then compare each exactly.

    ``exclude_same_stream`` maps query stream index -> corpus stream_id for
    self-comparison (dedupe of a corpus against itself): hits landing on the
    same stream within 3 s of the query time are the trivial self-diagonal and
    are excluded.
    """
    result = CandidateResult()
    streams = _collect_queries(manifest, result, quality_floor)
    if not streams:
        return result

    pairs = _sweep(
        streams, ann, k=k, radius=radius, nprobe=nprobe,
        probe_stride=max(1, probe_stride), max_query_block=max_query_block,
    )
    if not pairs:
        return result

    corpus_rows = {row.stream_id: row for row in catalog.iter_streams()}
    n_rungs = len(ann.crop_variants)
    by_idx = {s.qs_idx: s for s in streams}

    for qs_idx, corpus_sids in sorted(pairs.items()):
        qstream = by_idx[qs_idx]
        own = (exclude_same_stream or {}).get(qs_idx)
        for corpus_sid in sorted(corpus_sids):
            corpus_row = corpus_rows.get(corpus_sid)
            if corpus_row is None:
                continue
            rows, frame_idx, mirrored, variant, dist = compare_stream_pair(
                catalog, corpus_row, qstream.codes, n_rungs=n_rungs, k=k, radius=radius
            )
            if rows.size == 0:
                continue
            q_ms = qstream.times_ms[rows]
            c_ms = ((frame_idx + 0.5) * 1000.0 / corpus_row.sample_fps).astype(np.int64)

            keep = _suppress_hubs(frame_idx, qstream.codes.shape[0])
            if own == corpus_sid:
                # The trivial self-diagonal: a stream always matches itself.
                keep &= np.abs(c_ms - q_ms) >= 3000
            if not keep.any():
                continue
            result.hits[(qs_idx, corpus_sid)] = HitSet(
                q_ms=q_ms[keep],
                c_ms=c_ms[keep],
                dist=dist[keep],
                mirrored=mirrored[keep],
                crop_variant=variant[keep],
            )
    return result


def _collect_queries(
    manifest: Manifest, result: CandidateResult, quality_floor: int
) -> list[_QueryStream]:
    """Each manifest stream's usable frames, and its unjudgeable spans.

    Indeterminate spans are a property of the query side alone, so they are
    known before any search happens.
    """
    streams: list[_QueryStream] = []
    for qs_idx, stream in enumerate(manifest.streams):
        n = stream.n_frames
        if n == 0:
            continue
        qualities = np.frombuffer(stream.qualities, dtype=np.uint8)
        flags = np.frombuffer(stream.flags, dtype=np.uint8)
        times = _grid_ms(n, stream.sample_fps)
        unusable = (qualities < quality_floor) | ((flags & FLAG_LOW_QUALITY) != 0)
        spans = _flag_spans(unusable, times, stream.sample_fps)
        if spans:
            result.indeterminate[qs_idx] = spans
        keep = np.nonzero(~unusable)[0]
        if keep.size == 0:
            continue
        all_codes = np.frombuffer(stream.hashes, dtype=np.uint8).reshape(n, HASH_BYTES)
        streams.append(
            _QueryStream(
                qs_idx=qs_idx,
                codes=np.ascontiguousarray(all_codes[keep]),
                times_ms=times[keep],
                sample_fps=stream.sample_fps,
            )
        )
    return streams


def _sweep(
    streams: list[_QueryStream],
    ann: AnnIndex,
    *,
    k: int,
    radius: int,
    nprobe: int,
    probe_stride: int,
    max_query_block: int,
) -> dict[int, set[int]]:
    """Which corpus streams each query stream might overlap.

    Sparse and approximate on purpose: the exact comparison that follows only
    needs the pair, and a pair is nominated by any single matching frame.
    """
    probe_codes = [s.codes[::probe_stride] for s in streams]
    owner = np.concatenate(
        [np.full(c.shape[0], s.qs_idx, np.int32) for c, s in zip(probe_codes, streams, strict=True)]
    )
    codes = np.ascontiguousarray(np.vstack(probe_codes))

    pairs: dict[int, set[int]] = {}
    for lo in range(0, codes.shape[0], max_query_block):
        block = codes[lo : lo + max_query_block]
        hits = ann.search_resolved(
            block, k=k, radius=radius, nprobe=nprobe, max_hits_per_query=k
        )
        if len(hits) == 0:
            continue
        qs_of_hit = owner[hits.query_row + lo]
        for qs_idx, corpus_sid in np.unique(
            np.stack([qs_of_hit, hits.stream_id.astype(np.int32)], axis=1), axis=0
        ):
            pairs.setdefault(int(qs_idx), set()).add(int(corpus_sid))
    return pairs


def _suppress_hubs(frame_idx: np.ndarray, n_queries: int) -> np.ndarray:
    """Drop hits on corpus frames that matched an outsized share of the query.

    A genuine run maps query frames to corpus frames roughly one-to-one, so a
    corpus frame drawing far more than its share is static content.
    """
    threshold = max(32, int(0.005 * max(n_queries, 1)) + 1)
    uniq, counts = np.unique(frame_idx, return_counts=True)
    hubs = uniq[counts > threshold]
    if hubs.size == 0:
        return np.ones(frame_idx.size, dtype=bool)
    return ~np.isin(frame_idx, hubs)
