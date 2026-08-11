"""Where does indexing time go, and would hardware decode help?

Once the crop ladder is shallow, decode is the larger half of the work, so this
splits the two. It also measures the aggregate decode ceiling of CPU against GPU
at several concurrencies, which is the comparison that decides the question: a
consumer GPU has one NVDEC engine and a fixed ceiling, while CPU decode scales per
core, and indexing already runs a worker per core.

--cuda needs a working `ffmpeg -hwaccel cuda`. The published conclusion (GPU
loses) is specific to small frames and many cores; on 1080p or few cores it
reverses.
"""

from __future__ import annotations

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import av
from _common import clips, gray_frames, parse_args

from overlap.hashing import PdqKernel


def footage_seconds(paths: list) -> float:  # type: ignore[type-arg]
    total = 0.0
    for p in paths:
        with av.open(str(p)) as c:
            st = c.streams.video[0]
            total += float((st.duration or 0) * st.time_base)
    return total


def _run(path: object, *, cuda: bool) -> None:
    decode_only(str(path), cuda)


def decode_only(path: str, cuda: bool) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    if cuda:
        cmd += ["-hwaccel", "cuda"]
    subprocess.run([*cmd, "-i", path, "-f", "null", "-"], check=False, capture_output=True)


def main() -> None:
    args = parse_args(__doc__, cuda={"action": "store_true", "help": "also measure NVDEC"})
    paths = clips(args.clips, args.limit)
    total_s = footage_seconds(paths)

    t0 = time.perf_counter()
    frames = gray_frames(paths, args.per_clip)
    decode_s = time.perf_counter() - t0

    kernel = PdqKernel()
    kernel.hash_frame(frames[0])
    t0 = time.perf_counter()
    for f in frames:
        kernel.hash_frame(f)
    hash_s = time.perf_counter() - t0

    print(f"{len(frames)} sampled frames from {total_s:.0f}s of footage")
    print(f"  decode + convert {decode_s:7.2f}s")
    print(f"  hash             {hash_s:7.2f}s")
    print(f"  decode is {decode_s / (decode_s + hash_s) * 100:.0f}% of the work at 2 codes/frame\n")

    print(f"{'concurrency':>11s} {'CPU fps':>10s}" + (f"{'GPU fps':>10s}" if args.cuda else ""))
    frames_per = {p: 0 for p in paths}
    for p in paths:
        with av.open(str(p)) as c:
            st = c.streams.video[0]
            rate = float(st.average_rate or 0)
            frames_per[p] = int(rate * float((st.duration or 0) * st.time_base))
    for k in (1, 2, 4, 8):
        chosen = (paths * 4)[:k]
        want = sum(frames_per[p] for p in chosen)
        row = f"{k:>11d}"
        for cuda in ([False, True] if args.cuda else [False]):
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=k) as ex:
                list(ex.map(partial(_run, cuda=cuda), chosen))
            row += f"{want / (time.perf_counter() - t0):>10,.0f}"
        print(row)


if __name__ == "__main__":
    main()
