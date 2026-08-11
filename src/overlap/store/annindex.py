"""Sharded FAISS binary (Hamming) index over the catalog's frame hashes.

Each corpus frame contributes several codes, so a manipulated copy is found by
its plain fingerprints without the querying side guessing the manipulation:
identity plus horizontal mirror, and one pair per crop geometry. A code's
position encodes its variant, which is how a hit reports "cropped ~12%,
mirrored" without storing anything per hit.

The index is a set of independent on-disk shards. Only one is resident at a
time, so peak memory is a shard plus the query set rather than the whole
corpus; shard membership lives in the catalog, so adding footage builds shards
only for the new streams. Everything under ``ann/`` is derived: deleting it is
safe and costs only rebuild time.

Frames flagged FLAG_LOW_QUALITY are not indexed in any variant, since near
featureless frames collide across unrelated footage.

Rationale and measurements: docs/architecture.md.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import faiss
import numpy as np

from overlap.errors import IndexError_
from overlap.hashing.base import FLAG_LOW_QUALITY, HASH_BYTES
from overlap.hashing.prep import CropVariant, crop_variants_spec, parse_crop_variants_spec

if TYPE_CHECKING:
    from overlap.store.catalog import Catalog, StreamRow

HASH_BITS = HASH_BYTES * 8

# Codes per shard. 32M codes is ~1 GB of payload: small enough that one shard
# loads quickly and build memory stays near 2 GB, large enough that IVF
# clustering pays for itself and a 10 TB index stays around 300 shards.
DEFAULT_SHARD_CODES = 32_000_000

# Below this many codes a flat (exact) index beats the cost of IVF training.
# Set from where IVF becomes *trainable*, not from where it becomes worthwhile:
# nlist is 4*sqrt(n) centroids and FAISS wants ~39 training points each, so
# training is sound above ~25k codes. Anything above that is left to IVF because
# a flat shard is scanned exhaustively - measured at 13x slower once a corpus is
# split into shards small enough to hit it, which is exactly the case a user
# lowering index.shard_codes to fit a small machine lands in.
_IVF_THRESHOLD = 100_000

# Queries per FAISS call. Caps the size of the (n, k) result buffers without
# reloading the shard, which is why it does not cost an extra corpus pass.
_QUERY_BATCH = 500_000

# Streams with nothing indexable (no frames, or every frame low quality) are
# assigned here so that re-opening the index does not retry them forever.
_EMPTY_SHARD = "-none-"

Progress = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class ShardMeta:
    """What is needed to decide a shard is still valid, without reading it."""

    name: str
    n_codes: int
    n_frames: int
    n_streams: int
    codes_per_frame: int
    crop_variants: str

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_codes": self.n_codes,
            "n_frames": self.n_frames,
            "n_streams": self.n_streams,
            "codes_per_frame": self.codes_per_frame,
            "crop_variants": self.crop_variants,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> ShardMeta:
        return cls(
            name=str(d["name"]),
            n_codes=int(d["n_codes"]),
            n_frames=int(d["n_frames"]),
            n_streams=int(d["n_streams"]),
            codes_per_frame=int(d["codes_per_frame"]),
            crop_variants=str(d.get("crop_variants", "")),
        )


@dataclass
class ResolvedHits:
    """Flat, already-resolved hits. Which shard produced one is not visible."""

    query_row: np.ndarray  # int64 row in the query array
    stream_id: np.ndarray  # int64
    frame_idx: np.ndarray  # int32
    mirrored: np.ndarray  # bool
    crop_variant: np.ndarray  # int16, 0 = uncropped
    dist: np.ndarray  # int32

    def __len__(self) -> int:
        return int(self.query_row.size)

    @classmethod
    def empty(cls) -> ResolvedHits:
        return cls(
            np.empty(0, np.int64),
            np.empty(0, np.int64),
            np.empty(0, np.int32),
            np.empty(0, bool),
            np.empty(0, np.int16),
            np.empty(0, np.int32),
        )

    @classmethod
    def concat(cls, parts: list[ResolvedHits]) -> ResolvedHits:
        parts = [p for p in parts if len(p)]
        if not parts:
            return cls.empty()
        columns = zip(*(tuple(vars(p).values()) for p in parts), strict=True)
        return cls(*(np.concatenate(a) for a in columns))

    def take(self, sel: np.ndarray) -> ResolvedHits:
        return ResolvedHits(*(arr[sel] for arr in vars(self).values()))


class AnnIndex:
    """Searchable Hamming index over on-disk shards, with variant resolution."""

    def __init__(
        self,
        ann_dir: Path,
        shards: list[ShardMeta],
        crop_variants: tuple[CropVariant, ...],
    ) -> None:
        self.dir = ann_dir
        self.shards = shards
        self.crop_variants = crop_variants
        # Per shard, local variant index -> union variant index. Lets shards
        # built under different ladders report comparable geometry.
        self._remap = {s.name: _variant_remap(s, crop_variants) for s in shards}

    # -- properties ---------------------------------------------------------

    @property
    def n_codes(self) -> int:
        return sum(s.n_codes for s in self.shards)

    @property
    def n_shards(self) -> int:
        return len(self.shards)

    @property
    def bytes_on_disk(self) -> int:
        return sum(p.stat().st_size for p in self.dir.glob("shard-*") if p.is_file())

    def describe_variant(self, variant_idx: int) -> CropVariant | None:
        """The crop geometry behind a variant index (None = uncropped)."""
        if variant_idx <= 0 or variant_idx > len(self.crop_variants):
            return None
        return self.crop_variants[variant_idx - 1]

    # -- build / load -------------------------------------------------------

    @classmethod
    def build_or_load(
        cls,
        catalog: Catalog,
        *,
        progress: Progress | None = None,
        shard_codes: int | None = None,
        rebuild: bool = False,
    ) -> AnnIndex:
        """Bring the shard set in line with the catalog, building only what changed.

        Reconciliation is per stream: a shard survives when every stream
        assigned to it is still present with the same frame and rung counts.
        Anything else is rebuilt, so the cost of re-opening an index is
        proportional to what moved, not to the size of the archive.
        """
        emit = progress or (lambda _e: None)
        if shard_codes is None:
            shard_codes = _meta_int(catalog, "shard_codes", DEFAULT_SHARD_CODES)
        ann_dir = catalog.index_dir / "ann"
        ann_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = ann_dir / "shards.json"

        known: dict[str, ShardMeta] = {}
        if not rebuild:
            known = _read_manifest(manifest_path)
        assignments = {} if rebuild else catalog.shard_assignments()

        # Which shards can be kept, and which streams still need a home.
        covered: dict[str, int] = {}
        pending: list[StreamRow] = []
        doomed: set[str] = set()
        for row in catalog.iter_streams():
            entry = assignments.get(row.stream_id)
            if entry is None:
                pending.append(row)
                continue
            shard_name, n_frames, n_rungs = entry
            if (n_frames, n_rungs) != (row.n_frames, row.n_crop_rungs):
                # Re-indexed since it was sharded: its shard now holds codes
                # that no longer describe this stream.
                if shard_name != _EMPTY_SHARD:
                    doomed.add(shard_name)
                else:
                    pending.append(row)
                continue
            if shard_name == _EMPTY_SHARD:
                continue  # unchanged and still has nothing to index
            if shard_name not in known or not _shard_files_present(ann_dir, shard_name):
                pending.append(row)
                continue
            covered[shard_name] = covered.get(shard_name, 0) + 1

        # A shard that lost streams (their files were removed from the index)
        # still holds their codes, so it has to be rebuilt too.
        for name, meta in known.items():
            if covered.get(name, 0) != meta.n_streams:
                doomed.add(name)

        if doomed:
            for row in catalog.iter_streams():
                entry = assignments.get(row.stream_id)
                if entry is not None and entry[0] in doomed:
                    pending.append(row)
            catalog.drop_shard_assignments(doomed)
            for name in doomed:
                _delete_shard(ann_dir, name)
            emit({"event": "ann", "status": "stale", "shards_dropped": len(doomed)})

        kept = [m for name, m in known.items() if name not in doomed]
        if not pending:
            variants = _union_variants(catalog, kept)
            _write_manifest(manifest_path, kept, variants)
            return cls(ann_dir, kept, variants)

        emit(
            {
                "event": "ann",
                "status": "building",
                "streams_pending": len(pending),
                "shards_kept": len(kept),
            }
        )
        spec = crop_variants_spec(parse_crop_variants_spec(catalog.get_meta("crop_variants") or ""))
        built = _build_shards(
            catalog, ann_dir, pending, spec, shard_codes, _next_seq(kept), emit
        )
        shards = kept + built
        variants = _union_variants(catalog, shards)
        _write_manifest(manifest_path, shards, variants)
        emit(
            {
                "event": "ann",
                "status": "ready",
                "shards": len(shards),
                "codes": sum(s.n_codes for s in shards),
            }
        )
        return cls(ann_dir, shards, variants)

    # -- search -------------------------------------------------------------

    def search_resolved(
        self,
        queries: np.ndarray,
        *,
        k: int = 16,
        radius: int = 56,
        nprobe: int = 64,
        max_hits_per_query: int = 16,
        progress: Progress | None = None,
    ) -> ResolvedHits:
        """Search the whole corpus with the whole query set; resolve every hit.

        One pass over the shards: each is loaded once, searched against every
        query, and released. Peak memory is a single shard plus the queries,
        independent of corpus size.
        """
        if queries.ndim != 2 or queries.shape[1] != HASH_BYTES or queries.dtype != np.uint8:
            raise IndexError_(f"queries must be (n, {HASH_BYTES}) uint8")
        if queries.shape[0] == 0 or not self.shards:
            return ResolvedHits.empty()

        queries = np.ascontiguousarray(queries)
        parts: list[ResolvedHits] = []
        for shard in self.shards:
            parts.extend(self._search_one_shard(shard, queries, k, radius, nprobe))
            if progress:
                progress({"event": "shard_searched", "shard": shard.name})
        return _cap_per_query(ResolvedHits.concat(parts), max_hits_per_query)

    def _search_one_shard(
        self, shard: ShardMeta, queries: np.ndarray, k: int, radius: int, nprobe: int
    ) -> list[ResolvedHits]:
        index, stream_ids, frame_idx = _load_shard(self.dir, shard)
        if isinstance(index, faiss.IndexBinaryIVF):
            index.nprobe = max(1, nprobe)
        remap = self._remap[shard.name]
        out: list[ResolvedHits] = []
        try:
            for lo in range(0, queries.shape[0], _QUERY_BATCH):
                batch = queries[lo : lo + _QUERY_BATCH]
                dists, positions = index.search(batch, k)
                keep = (positions >= 0) & (dists <= radius)
                if not keep.any():
                    continue
                rows, _cols = np.nonzero(keep)
                pos = positions[keep].astype(np.int64)
                slot = pos // shard.codes_per_frame
                code = pos % shard.codes_per_frame
                out.append(
                    ResolvedHits(
                        query_row=rows.astype(np.int64) + lo,
                        stream_id=stream_ids[slot],
                        frame_idx=frame_idx[slot],
                        mirrored=(code % 2 == 1),
                        crop_variant=remap[code // 2],
                        dist=dists[keep].astype(np.int32),
                    )
                )
        finally:
            del index  # release before the next shard loads
        return out


# -- build helpers ----------------------------------------------------------


def _build_shards(
    catalog: Catalog,
    ann_dir: Path,
    pending: list[StreamRow],
    spec: str,
    shard_codes: int,
    first_seq: int,
    emit: Progress,
) -> list[ShardMeta]:
    """Turn pending streams into shards, holding at most one shard in memory."""
    variants = parse_crop_variants_spec(spec)
    codes_per_frame = (1 + len(variants)) * 2
    built: list[ShardMeta] = []
    # (codes, frame_idx, stream_id, catalog n_frames, catalog n_crop_rungs)
    buf: list[tuple[np.ndarray, np.ndarray, int, int, int]] = []
    buf_codes = 0
    seq = first_seq

    def flush() -> None:
        nonlocal buf, buf_codes, seq
        if not buf:
            return
        meta = _write_shard(
            ann_dir,
            seq,
            np.vstack([b[0] for b in buf]),
            np.concatenate([np.full(b[1].size, b[2], np.int64) for b in buf]),
            np.concatenate([b[1] for b in buf]),
            codes_per_frame=codes_per_frame,
            spec=spec,
        )
        catalog.assign_shard(meta.name, [(b[2], b[3], b[4]) for b in buf])
        built.append(meta)
        emit({"event": "shard", "shard": meta.name, "codes": meta.n_codes})
        buf, buf_codes, seq = [], 0, seq + 1

    seen: set[int] = set()
    for row in pending:
        if row.stream_id in seen:  # a doomed shard can re-offer a stream
            continue
        seen.add(row.stream_id)
        block = stream_codes(catalog, row, codes_per_frame, len(variants))
        if block is None:
            # Nothing indexable (no frames, or all low quality). Still record
            # the assignment, or every re-open would retry this stream forever.
            catalog.assign_shard(_EMPTY_SHARD, [(row.stream_id, row.n_frames, row.n_crop_rungs)])
            continue
        codes, frames = block
        buf.append((codes, frames, row.stream_id, row.n_frames, row.n_crop_rungs))
        buf_codes += codes.shape[0]
        if buf_codes >= shard_codes:
            flush()
    flush()
    return built


def stream_codes(
    catalog: Catalog, row: StreamRow, codes_per_frame: int, n_rungs: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Every indexable code for one stream, laid out variant-major per frame.

    Returns ``(codes, frame_idx)`` where ``codes`` has ``codes_per_frame`` rows
    per kept frame, or None when the stream has nothing worth indexing.
    """
    n = row.n_frames
    if n == 0:
        return None
    hashes, mirrors, _q, flags = catalog.stream_hashes(row.stream_id)
    flag_arr = np.frombuffer(flags, dtype=np.uint8)
    keep = np.nonzero((flag_arr & FLAG_LOW_QUALITY) == 0)[0]
    if keep.size == 0:
        return None
    ids_arr = np.frombuffer(hashes, dtype=np.uint8).reshape(n, HASH_BYTES)
    mx_arr = np.frombuffer(mirrors, dtype=np.uint8).reshape(n, HASH_BYTES)

    # Per frame: [rung0 identity, rung0 mirror, rung1 identity, rung1 mirror, ...]
    block = np.empty((keep.size, codes_per_frame, HASH_BYTES), dtype=np.uint8)
    block[:, 0] = ids_arr[keep]
    block[:, 1] = mx_arr[keep]

    ch = cm = None
    if row.n_crop_rungs:
        crop_hashes, crop_mirrors = catalog.stream_crop_hashes(row.stream_id)
        expected = n * row.n_crop_rungs * HASH_BYTES
        if len(crop_hashes) == expected and len(crop_mirrors) == expected:
            ch = np.frombuffer(crop_hashes, dtype=np.uint8).reshape(
                n, row.n_crop_rungs, HASH_BYTES
            )
            cm = np.frombuffer(crop_mirrors, dtype=np.uint8).reshape(
                n, row.n_crop_rungs, HASH_BYTES
            )
    usable = min(row.n_crop_rungs, n_rungs) if ch is not None else 0
    for r in range(usable):
        block[:, 2 + 2 * r] = ch[keep, r]  # type: ignore[index]
        block[:, 3 + 2 * r] = cm[keep, r]  # type: ignore[index]
    # A stream indexed with fewer rungs than the current ladder repeats its
    # uncropped code rather than leaving the slot undefined.
    for r in range(usable, n_rungs):
        block[:, 2 + 2 * r] = ids_arr[keep]
        block[:, 3 + 2 * r] = mx_arr[keep]

    return block.reshape(-1, HASH_BYTES), keep.astype(np.int32)


def _write_shard(
    ann_dir: Path,
    seq: int,
    codes: np.ndarray,
    stream_ids: np.ndarray,
    frame_idx: np.ndarray,
    *,
    codes_per_frame: int,
    spec: str,
) -> ShardMeta:
    name = f"shard-{seq:06d}"
    codes = np.ascontiguousarray(codes)
    n = codes.shape[0]
    if n >= _IVF_THRESHOLD:
        nlist = int(max(1, min(4 * np.sqrt(n), n // 64)))
        index: faiss.IndexBinary = faiss.IndexBinaryIVF(
            faiss.IndexBinaryFlat(HASH_BITS), HASH_BITS, nlist
        )
        rng = np.random.default_rng(0)
        sample = codes[rng.choice(n, size=min(n, max(64 * nlist, 100_000)), replace=False)]
        index.train(sample)
        index.add(codes)
    else:
        index = faiss.IndexBinaryFlat(HASH_BITS)
        index.add(codes)
    # Mapping first: a shard is only usable with both files, and the manifest
    # naming it is written last of all.
    np.savez(ann_dir / f"{name}.npz", stream_ids=stream_ids, frame_idx=frame_idx)
    faiss.write_index_binary(index, str(ann_dir / f"{name}.faiss"))
    return ShardMeta(
        name=name,
        n_codes=n,
        n_frames=int(frame_idx.size),
        n_streams=int(np.unique(stream_ids).size),
        codes_per_frame=codes_per_frame,
        crop_variants=spec,
    )


def _load_shard(
    ann_dir: Path, shard: ShardMeta
) -> tuple[faiss.IndexBinary, np.ndarray, np.ndarray]:
    index = faiss.read_index_binary(str(ann_dir / f"{shard.name}.faiss"))
    maps = np.load(ann_dir / f"{shard.name}.npz")
    return index, maps["stream_ids"], maps["frame_idx"]


def _shard_files_present(ann_dir: Path, name: str) -> bool:
    return (ann_dir / f"{name}.faiss").is_file() and (ann_dir / f"{name}.npz").is_file()


def _delete_shard(ann_dir: Path, name: str) -> None:
    for suffix in (".faiss", ".npz"):
        (ann_dir / f"{name}{suffix}").unlink(missing_ok=True)


def _next_seq(shards: list[ShardMeta]) -> int:
    return 1 + max((int(s.name.rsplit("-", 1)[1]) for s in shards), default=-1)


# -- variant reconciliation -------------------------------------------------


def _union_variants(catalog: Catalog, shards: list[ShardMeta]) -> tuple[CropVariant, ...]:
    """Current ladder first, then any geometry only older shards carry."""
    union: list[CropVariant] = list(
        parse_crop_variants_spec(catalog.get_meta("crop_variants") or "")
    )
    known = {(v.side, round(v.frac, 4)) for v in union}
    for shard in shards:
        for v in parse_crop_variants_spec(shard.crop_variants):
            key = (v.side, round(v.frac, 4))
            if key not in known:
                known.add(key)
                union.append(v)
    return tuple(union)


def _variant_remap(shard: ShardMeta, union: tuple[CropVariant, ...]) -> np.ndarray:
    """Map a shard's local variant indices onto the union ladder."""
    pos = {(v.side, round(v.frac, 4)): i + 1 for i, v in enumerate(union)}
    local = parse_crop_variants_spec(shard.crop_variants)
    remap = np.zeros(1 + max(len(local), shard.codes_per_frame // 2), dtype=np.int16)
    for i, v in enumerate(local):
        remap[i + 1] = pos.get((v.side, round(v.frac, 4)), 0)
    return remap


# -- misc -------------------------------------------------------------------


def _cap_per_query(hits: ResolvedHits, cap: int) -> ResolvedHits:
    """Keep the closest ``cap`` hits per query so a shard sweep cannot blow up.

    Without this, hit memory grows with shard count: a static scene present in
    every shard would return k hits per shard per query.
    """
    if len(hits) == 0 or cap <= 0:
        return hits
    order = np.lexsort((hits.dist, hits.query_row))
    q = hits.query_row[order]
    idx = np.arange(q.size)
    starts = np.where(np.r_[True, q[1:] != q[:-1]], idx, 0)
    rank = idx - np.maximum.accumulate(starts)
    sel = np.sort(order[rank < cap])
    return hits.take(sel)


def _read_manifest(path: Path) -> dict[str, ShardMeta]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return {s["name"]: ShardMeta.from_json(s) for s in doc["shards"]}
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return {}


def _write_manifest(
    path: Path, shards: list[ShardMeta], variants: tuple[CropVariant, ...]
) -> None:
    payload = {
        "version": 1,
        "crop_variants": crop_variants_spec(variants),
        "shards": [s.to_json() for s in sorted(shards, key=lambda s: s.name)],
    }
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _meta_int(catalog: Catalog, key: str, default: int) -> int:
    raw = catalog.get_meta(key)
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        return default


def describe_index(catalog: Catalog) -> dict[str, Any]:
    """Report the search index's state without building or loading anything.

    ``streams_unsharded`` is the honest answer to "is this index ready to
    search": those streams are fingerprinted and safe on disk, but the next
    comparison has to shard them first.
    """
    ann_dir = catalog.index_dir / "ann"
    shards = _read_manifest(ann_dir / "shards.json")
    assigned = catalog.shard_assignments()
    total = sum(1 for _ in catalog.iter_streams())
    return {
        "shards": len(shards),
        "codes": sum(s.n_codes for s in shards.values()),
        "bytes_on_disk": sum(
            p.stat().st_size for p in ann_dir.glob("shard-*") if p.is_file()
        ),
        "streams_sharded": len(assigned),
        "streams_unsharded": max(0, total - len(assigned)),
    }


def iter_query_blocks(codes: np.ndarray, block_rows: int) -> Iterator[tuple[int, np.ndarray]]:
    """Split a query array into blocks, each costing one pass over the corpus."""
    for lo in range(0, codes.shape[0], block_rows):
        yield lo, codes[lo : lo + block_rows]
