"""How deep a crop survives the plain hash, and what each ladder rung buys.

Sets the default crop geometries. The breaking points are asymmetric: removing a
strip from one side shifts content as well as narrowing the field of view, so it
fails at roughly half the depth of a centred crop.
"""

from __future__ import annotations

from _common import centre_crop, clips, edge_crop, gray_frames, parse_args

from overlap.hashing import PdqKernel
from overlap.hashing.pdq_numpy import hamming

RADIUS = 56
LADDERS = {
    "none (identity only)": (),
    "1 rung 0.94": (0.94,),
    "2 rungs 0.94,0.88": (0.94, 0.88),
    "5 rungs 0.94..0.70": (0.94, 0.88, 0.82, 0.76, 0.70),
}


def main() -> None:
    args = parse_args(__doc__)
    kernel = PdqKernel()
    frames = gray_frames(clips(args.clips, args.limit), args.per_clip)
    base = [kernel.hash_frame(f).hash for f in frames]
    print(f"{len(frames)} frames, accept at Hamming <= {RADIUS} of 256\n")

    for label, make in (
        ("centre crop", lambda f, p: centre_crop(f, 1 - p / 100)),
        ("bottom strip removed", lambda f, p: edge_crop(f, p / 100)),
    ):
        print(f"{label}, against the uncropped hash:")
        print(f"  {'crop':>5s} {'median':>7s} {'worst':>7s}  caught")
        for pct in (1, 2, 3, 4, 5, 6, 8, 10, 15):
            d = sorted(
                hamming(kernel.hash_frame(make(f, pct)).hash, b)
                for f, b in zip(frames, base, strict=True)
            )
            caught = sum(1 for x in d if x <= RADIUS)
            print(f"  {pct:4d}% {d[len(d) // 2]:7d} {d[-1]:7d}  {caught}/{len(d)}")
        print()

    print("deepest centre crop where every frame is still caught:")
    for label, rungs in LADDERS.items():
        rung_hashes = [[kernel.hash_frame(centre_crop(f, r)).hash for r in rungs] for f in frames]
        deepest = 0
        for pct in range(1, 40):
            ok = 0
            for idx, f in enumerate(frames):
                q = kernel.hash_frame(centre_crop(f, 1 - pct / 100)).hash
                best = min([hamming(q, base[idx])] + [hamming(q, h) for h in rung_hashes[idx]])
                ok += best <= RADIUS
            if ok < len(frames):
                break
            deepest = pct
        print(f"  {label:22s} {(1 + len(rungs)) * 2:2d} codes/frame  {deepest:2d}%")


if __name__ == "__main__":
    main()
