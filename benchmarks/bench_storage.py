"""What an index costs on disk, per frame per indexed code.

Not 32 bytes. A digest is 32 bytes, but the catalog stores identity and mirror
digests plus the same pair per crop rung, and the shards then store those codes
again alongside IVF centroids and the frame-to-stream mapping. Pricing a run on
the digest payload alone understates it by more than half.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from _common import clips, parse_args

from overlap.ingest import index_paths
from overlap.store.annindex import AnnIndex
from overlap.store.catalog import Catalog

PRESETS = {
    "fast (2 codes)": ("", ""),
    "mild (6 codes)": ("0.94", "bottom:0.06"),
    "balanced (12 codes)": ("0.94,0.88,0.82,0.76,0.70", ""),
}


def main() -> None:
    args = parse_args(__doc__)
    paths = clips(args.clips, args.limit)
    print(f"{len(paths)} clips\n")
    header = f"{'preset':22s} {'codes/f':>7s} {'catalog':>9s} {'shards':>9s}"
    print(f"{header} {'B/frame':>8s} {'B/frame/code':>13s}")
    for label, (ladder, edges) in PRESETS.items():
        tmp = Path(tempfile.mkdtemp(prefix="ovbench-"))
        try:
            idx = tmp / "corpus.ovl"
            stats = index_paths(paths, idx, crop_ladder=ladder, crop_edges=edges, workers=0)
            with Catalog.open(idx) as cat:
                AnnIndex.build_or_load(cat)
                rungs = max({r.n_crop_rungs for r in cat.iter_streams()}, default=0)
            cat_b = sum(p.stat().st_size for p in idx.glob("catalog.sqlite*"))
            ann_b = sum(p.stat().st_size for p in (idx / "ann").glob("shard-*"))
            codes = (1 + rungs) * 2
            per_frame = (cat_b + ann_b) / max(stats.frames, 1)
            print(
                f"{label:22s} {codes:>7d} {cat_b / 1e6:>8.1f}M {ann_b / 1e6:>8.1f}M "
                f"{per_frame:>8.0f} {per_frame / codes:>13.1f}"
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
