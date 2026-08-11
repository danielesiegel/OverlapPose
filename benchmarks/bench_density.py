"""How much detection is lost by sampling the corpus less densely?

This sets index.fps, the largest single cost driver in the tool, so it is the
benchmark worth running before planning anything. The "offer" is built the way an
aggregator repackages: concatenate the episodes, re-cut on boundaries that align
with nothing, transcode. No cropping and no adversarial games, just repackaging of
footage the corpus owns in full, so anything below 100% is loss.

Needs the ffmpeg CLI on PATH.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from _common import clips, parse_args

from overlap.ingest import index_paths
from overlap.match.compare import compare_manifest
from overlap.store.catalog import Catalog
from overlap.store.manifest import build_manifest

SEGMENT_S = 137  # coprime with everything


def build_offer(paths: list[Path], out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    listing = out.parent / "concat.txt"
    listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in paths), encoding="utf-8")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-an",
         "-f", "segment", "-segment_time", str(SEGMENT_S), "-reset_timestamps", "1",
         str(out / "offer_%04d.mp4")],
        check=True,
    )
    return len(list(out.glob("*.mp4")))


def main() -> None:
    args = parse_args(__doc__)
    paths = clips(args.clips, args.limit)
    root = Path(tempfile.mkdtemp(prefix="ovdensity-"))
    try:
        n_offer = build_offer(paths, root / "offer-src")
        print(f"corpus: {len(paths)} episodes;  offer: {n_offer} files, same footage "
              f"concatenated, re-cut at {SEGMENT_S}s, transcoded\n")
        print(f"{'corpus fps':>10s} {'overlap':>9s} {'files hit':>11s}")
        for fps in (4.0, 2.0, 1.0, 0.5):
            cdir, odir = root / f"c{fps}.ovl", root / f"o{fps}.ovl"
            index_paths(paths, cdir, sample_fps=fps, crop_ladder="", crop_edges="", workers=0)
            index_paths(list((root / "offer-src").glob("*.mp4")), odir,
                        sample_fps=max(fps, 1.0), crop_ladder="", crop_edges="", workers=0)
            with Catalog.open(odir) as ocat:
                manifest, _ = build_manifest(ocat)
            with Catalog.open(cdir) as ccat:
                rep = compare_manifest(manifest, ccat)
            s = rep["summary"]
            print(f"{fps:>10.1f} {s['overlap_pct']:>8.1f}% {s['files_with_overlap']:>6d}/{n_offer}")
        print("\nThe offer is entirely owned footage, so anything under 100% is loss.")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
