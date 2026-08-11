# Benchmarks

Every performance and calibration number in `docs/architecture.md` comes from one
of these scripts. They are here so the numbers can be checked and disputed rather
than taken on trust.

## Provenance of the published figures

All published figures were measured on **one machine** (Windows 11, 20 logical
cores, GTX 1080, NVMe) against **one dataset** (a 260-clip sample of
`builddotai/Egocentric-100K`: 456x256 H.264, 30 fps, 3-minute clips, ~0.6 Mbps).
That is a narrow basis. Treat them as the right order of magnitude and the correct
*relative* comparisons, not as constants. Re-run on your own hardware and footage
before planning capacity from them.

Where a figure is quoted to three significant figures in the docs, it is the
arithmetic mean of one such run, not a converged estimate.

## Running them

```
uv sync --group dev
python benchmarks/bench_hash.py                      # no footage needed
python benchmarks/bench_crop.py    --clips DIR
python benchmarks/bench_density.py --clips DIR        # needs the ffmpeg CLI
python benchmarks/bench_storage.py --clips DIR
python benchmarks/bench_metrics.py --clips DIR
python benchmarks/bench_decode.py  --clips DIR        # optional: --cuda
```

`DIR` is any directory of video files. A few dozen clips of a minute or more is
enough for stable numbers; the published runs used 24-260 clips depending on the
question.

## What each answers

| script | question | published figure |
|---|---|---|
| `bench_hash.py` | How fast is one fingerprint, and does the kernel still satisfy its invariants? | 1.82 ms/hash |
| `bench_crop.py` | How deep a crop survives, and what does each ladder rung buy? | 5% centred, 3% one-sided; 1 rung reaches 11% |
| `bench_density.py` | How much detection is lost by sampling the corpus less densely? | 99.5% at 4 fps, 40.1% at 1 fps |
| `bench_storage.py` | What does an index actually cost per frame? | ~72 bytes per frame per code |
| `bench_metrics.py` | Would cosine similarity, or keeping DCT magnitudes, work better? | no: 2-3x worse separation |
| `bench_decode.py` | Where does indexing time go, and would GPU decode help? | decode is 77%; NVDEC is ~2.4x slower here |

## Caveats worth reading before quoting any of this

- **`bench_density.py` is the one that matters most.** It sets `index.fps`, which
  is the single largest cost driver in the tool. It builds its own "repackaged"
  offer with ffmpeg (concatenate, re-cut on boundaries that align with nothing,
  transcode) so the result reflects what an aggregator actually does.
- **`bench_decode.py --cuda` needs a working `ffmpeg -hwaccel cuda`.** The
  conclusion that GPU decode loses is specific to small frames and many cores. On
  1080p or 4K footage, or on a 2-4 core VM, it reverses.
- Numbers from these scripts are wall-clock on a shared desktop. Pin cores and
  drop priority if you want them repeatable.
