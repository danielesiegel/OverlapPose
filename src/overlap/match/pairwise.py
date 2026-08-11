"""Exact comparison of one query stream against one corpus stream.

Once the sweep has named a pair, a single stream's codes are small enough to
compare exhaustively: every usable query frame against every code of that
stream, with no quantizer in the path.

So recall does not depend on search tuning, and cost is bounded by the two
streams rather than by the archive around them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import faiss
import numpy as np

from overlap.hashing.base import HASH_BYTES

if TYPE_CHECKING:
    from overlap.store.catalog import Catalog, StreamRow

HASH_BITS = HASH_BYTES * 8


def compare_stream_pair(
    catalog: Catalog,
    corpus_row: StreamRow,
    queries: np.ndarray,
    *,
    n_rungs: int,
    k: int,
    radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exactly search one corpus stream with every query frame.

    Returns ``(query_row, frame_idx, mirrored, crop_variant, dist)``, flat and
    already resolved. Variant indices refer to the catalog's current ladder,
    which is the prefix of the index's union ladder.
    """
    from overlap.store.annindex import stream_codes

    codes_per_frame = (1 + n_rungs) * 2
    block = stream_codes(catalog, corpus_row, codes_per_frame, n_rungs)
    if block is None or queries.shape[0] == 0:
        return _empty()
    codes, frame_idx = block

    index = faiss.IndexBinaryFlat(HASH_BITS)
    index.add(codes)
    dists, positions = index.search(np.ascontiguousarray(queries), k)
    keep = (positions >= 0) & (dists <= radius)
    if not keep.any():
        return _empty()

    rows, _cols = np.nonzero(keep)
    pos = positions[keep].astype(np.int64)
    slot = pos // codes_per_frame
    code = pos % codes_per_frame
    return (
        rows.astype(np.int64),
        frame_idx[slot],
        code % 2 == 1,
        np.clip((code // 2).astype(np.int16), 0, n_rungs),
        dists[keep].astype(np.int32),
    )


def _empty() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty(0, np.int64),
        np.empty(0, np.int32),
        np.empty(0, bool),
        np.empty(0, np.int16),
        np.empty(0, np.int32),
    )
