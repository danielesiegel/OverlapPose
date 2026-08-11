# .ovlm manifest format (version 1)

The manifest is the interchange artifact between vendor and lab. It contains
perceptual fingerprints and file metadata - never frames or pixels - and is
designed to be compact (~120 KB per hour of footage at the default export
density), integrity-checked, and safe to parse as untrusted input.

## Binary layout

```
offset  size  field
0       4     magic  "OVLM"
4       2     format_version, u16 little-endian  (currently 1)
6       4     header_len, u32 little-endian
10      n     header, UTF-8 JSON (see below)
10+n    ...   section bytes, concatenated, each zstd-compressed independently
```

## Header JSON

```json
{
  "schema": 1,
  "algo_id": "pdq2",
  "prep_id": "p1",
  "sample_fps": 1.0,
  "label": "Q3 kitchen teleop",
  "tool": "overlap 0.1.0",
  "dataset": {
    "merkle_root": "<hex sha256>",
    "n_files": 331,
    "n_streams": 340,
    "total_hours": 512.4,
    "total_frames": 1844640
  },
  "sections": [
    {"name": "files.msgpack",   "offset": 0,  "len": 1, "raw_len": 1,
     "codec": "zstd", "sha256": "<hex of compressed bytes>"},
    {"name": "streams.msgpack", "...": "..."},
    {"name": "frames.bin",      "...": "..."}
  ]
}
```

- `algo_id` + `prep_id` identify the exact fingerprint implementation.
  Comparison refuses mismatched identifiers: hashes from different
  implementations must never be silently mixed.
- Section offsets are relative to the end of the header. Readers must verify
  each section's `sha256` before decompressing and must honor `raw_len` as a
  decompression output cap.

## Sections

**`files.msgpack`** - msgpack array, sorted by relpath:

```
[[relpath: str, size: int, sha256: bytes(32), container: str], ...]
```

With `--anonymize-paths`, relpaths are content-derived
(`f<sha256-prefix>.<ext>`); the Merkle root binds whichever names the
manifest promises.

**`streams.msgpack`** - msgpack array, one entry per visual stream:

```
[[file_idx: int, stream_key: str, codec: str, width: int, height: int,
  duration_ms: int, sample_fps: float, n_frames: int], ...]
```

`stream_key` is `"v0"`, `"v1"`, … for video tracks and the topic name for
robotics containers (e.g. `"/cam_wrist/image_raw/compressed"`).

**`frames.bin`** - per stream, in `streams.msgpack` order, concatenated:

```
identity hashes:  n_frames × 32 bytes
qualities:        n_frames × u8   (0..100)
flags:            n_frames × u8   (bit 0: low-quality, excluded from matching)
```

Frame timestamps are implicit: frame *k* of a stream samples the presented
timeline at `t = (k + 0.5) / sample_fps` seconds.

## Fingerprints

Crop and orientation variants live only in the local index, never in a
manifest: the querying side sends plain uncropped fingerprints and the corpus
side does the variant work. A lab can therefore deepen its own detection
(deeper crop ladders, edge crops) without asking vendors to re-export
anything, and manifests stay comparable across configurations.

`algo_id: "pdq2"` is overlap's mirror-symmetric implementation of the PDQ
perceptual hash (256-bit DCT-sign over a smoothed 64×64 luminance
downsample). Only identity-orientation digests are stored in manifests; the
local index additionally stores each frame's horizontal-mirror digest, which
is how flipped copies are caught without doubling manifest size. Bit packing:
bit `k = i*16 + j` of the 16×16 DCT sign matrix lands in byte `k >> 3`,
MSB-first.

## Merkle root

```
leaf   = SHA-256( 0x00 || relpath_utf8 || 0x00 || file_sha256 )
node   = SHA-256( 0x01 || left || right )
```

Leaves are sorted by relpath; levels pair left-to-right and an unpaired node
is promoted unchanged; the empty dataset has a root of 32 zero bytes. The
domain prefixes (0x00/0x01) prevent leaf/node confusion. `overlap verify`
recomputes this over delivered files to confirm a delivery matches the quote.

## Privacy properties

The manifest contains no frames and no pixels - but perceptual hashes are
not encryption. Given a manifest, an adversary can confirm whether footage
they *already possess* matches it, and coarse visual structure (brightness
layout) leaks by design. Treat manifests as confidential business documents.
Keyed/PSI exchange is on the roadmap.

## What must match between two sides

Only `algo_id` and `prep_id`. Comparison verifies them and refuses a mismatch,
because hashes from a different function or normalization are not comparable.

`sample_fps` is *informational*: each stream carries its own rate and the
matcher works in real time, so a manifest exported at 1 fps compares correctly
against a corpus indexed at 2, 4 or 8 fps. Crop and orientation variants are
corpus-side only and never appear in a manifest, so the two sides may run
entirely different variant configurations.

## Compatibility policy

Pre-1.0: the format may change between minor versions; `format_version`
mismatches are refused with a clear error. Post-1.0: readers will accept at
least one prior format version.
