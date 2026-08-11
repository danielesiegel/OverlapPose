# overlap

**Perceptual fingerprinting and overlap detection for robotics datasets. Local-only.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

## The problem

Robotics labs buy egocentric and teleoperation datasets largely sight-unseen:
the footage is the product, so vendors can't share it before the sale, and
buyers can't check what they're getting before paying. That gap gets
exploited. Off-the-shelf datasets are resold to multiple buyers with the
files re-encoded, trimmed, spliced, mirrored, or - a known trick for
inflating billable hours - **slowed down**, so that plain checksum
comparison sees "new" data. A lab can end up paying twice for footage it
already owns and only find out after delivery, if at all.

overlap closes that gap without anyone sharing raw data:

1. **The vendor** fingerprints their dataset locally and exports a compact
   manifest (`.ovlm`) - perceptual fingerprints and file metadata, never
   frames or pixels. Roughly 120 KB per hour of footage.
2. **The lab** keeps a local fingerprint index of everything it already owns
   and compares the incoming manifest against it - entirely offline.
3. The report answers the purchasing question directly: *"62% of these hours
   match footage we already own - here are the files, the matched segments,
   and three files that were slowed to double their runtime."* Buy the other
   38%.

Both sides run the same open tool, so neither has to trust the other's
tooling - and honest vendors benefit most: `overlap self-dedupe` proves a
quote contains no internal duplicates, and `overlap verify` lets the buyer
confirm the delivery matches the quoted manifest bit-for-bit.

## How it works

```
videos / MCAP / ROS bags ──► decoded frames ──► perceptual hashes (pdq2)
                                                      │
                              local index (SQLite + shards)  ◄── lab corpus
                                                      │
vendor manifest (.ovlm, no pixels) ──► compare ──► report (JSON / HTML)
```

Fingerprints are computed on **decoded pixels, never on file bytes** - so
re-encoding, container swaps, metadata stripping, and re-wrapping an MCAP
camera topic as an `.mp4` all collapse to the same fingerprints.

Matching happens at the *segment* level: shared footage forms straight lines
in the (query-time × corpus-time) plane, and the line's slope directly
measures speed changes - a file slowed to 0.5x shows up as a slope-2 line,
flagged as billable-hours inflation with the underlying real footage hours
quantified. Splices appear as multiple line segments from different sources;
trims as partial segments.

Files assembled out of pieces are reported as such: each matched stretch is
its own segment, so an offer built from two cuts of one master reads as *"2
segments from 1 owned file"*, and an offer that pads owned footage with new
material reports only the owned share.

Manipulations that change the *frame itself* are handled by indexing the
corpus in several variants, so the buyer never has to guess what was done:
each frame is stored mirrored as well as upright (flips), re-hashed at a
ladder of centered crops (zoom-crops to ~30%), and - when enabled - at
one-sided edge crops (`--crop-edges bottom,top`) for copies that trim an
overlay strip away. A match reports which variant it landed on - *"cropped
bottom 18%, mirrored"* - so the report describes the manipulation, not just
the overlap.

Evidence that falls below the reporting threshold is listed separately rather
than dropped: a 0% headline never hides a partial match.

## Quickstart

Install (Python ≥ 3.10; [ffmpeg not required](#install) for normal use):

```
pipx install overlap-cli        # command is `overlap`
```

**You're a vendor** - fingerprint inventory, prove freshness, export a manifest:

```
overlap index D:\datasets\teleop-2026q3
overlap self-dedupe                       # find internal duplicates before quoting
overlap export -o q3-offer.ovlm --label "Q3 kitchen teleop" --anonymize-paths
# send q3-offer.ovlm to the buyer (it contains fingerprints, not footage)
```

**You're a lab** - index what you own once, then screen every offer:

```
overlap index /data/corpus                # resumable; run it after every purchase
overlap compare q3-offer.ovlm -o report.json --html report.html
# report.html: overlap %, matched segments, slowdown/splice/flip flags
```

Gate a procurement pipeline on it (exit code 3 = overlap at/above threshold):

```
overlap compare q3-offer.ovlm --fail-over 20 && echo "clean enough to buy"
```

After delivery, verify the bytes match what was quoted:

```
overlap verify q3-offer.ovlm --data /data/incoming/delivery
```

## The buying methodology

The commands map onto a purchasing decision: keep fingerprints of everything
you have seen, ask for a manifest covering the whole offer, check it against
the sample you were shown, ask how much you already own, and ask whether two
sellers are selling the same thing. Step by step, with the commands and what
each step can and cannot establish: [docs/buying-workflow.md](docs/buying-workflow.md).

## What it detects - and what it does not

overlap's detection claims are governed by a matrix that is **literally
executable**: the table below is generated from the same file the test suite
runs, and a test fails if they drift apart.

<!-- detection-matrix:start (generated by scripts/gen_detection_matrix_doc.py) -->
| manipulation | example tested | status | test row |
|---|---|---|---|
| metadata strip / rename | Byte-identical file with a different name and stripped metadata | **designed to detect** | `identical-rename` |
| re-encode | Transcode to H.265 at visibly lower quality | **designed to detect** | `reencode-h265-crf30` |
| container swap | Repackaged mp4 -> mkv without re-encoding | **designed to detect** | `container-swap-mkv` |
| trimming | A 12-second cut from the middle of a corpus file | **designed to detect** | `trim-12s` |
| trimming | A cut at an arbitrary sub-second point (sampling grids misaligned) | **designed to detect** | `trim-arbitrary-phase` |
| splicing | 15 s of corpus footage spliced with unrelated footage | **designed to detect** | `splice-15s-plus-other` |
| reassembly (one master) | Two non-adjacent pieces of one master concatenated and sold as a new session | **designed to detect** | `merge-two-same` |
| concatenation (two owned files) | Pieces of two different owned masters concatenated into one offer | **designed to detect** | `merge-two-masters` |
| concatenation (owned + new) | Owned footage concatenated with genuinely new footage; only the owned half should count | **designed to detect** | `merge-owned-plus-new` |
| speed change (slowdown) | Slowed to half speed - the billable-hours inflation trick | **designed to detect** | `speed-0.5x` |
| speed change (speedup) | Sped up to double speed | **designed to detect** | `speed-2x` |
| horizontal flip | Mirrored left-right | **designed to detect** | `hflip` |
| crop (slight) | Center crop keeping 95% of each dimension, upscaled back | **designed to detect** | `crop-5pct` |
| crop (moderate) | Center crop keeping 85%; needs --preset balanced | **designed to detect** | `crop-15pct` |
| crop (heavy) | Center crop keeping 70%; needs --preset balanced | **designed to detect** | `crop-30pct` |
| crop (beyond the ladder) | Center crop keeping 60% - past the deepest rung of the deepest ladder | not detected | `crop-40pct` |
| edge crop (bottom strip) | Bottom 15% removed (overlay strip trimmed); needs --crop-edges bottom | **designed to detect** | `crop-bottom-15` |
| edge crop (top strip) | Top 10% removed; lands between the 6% and 12% rungs | **designed to detect** | `crop-top-10` |
| edge crop (thin bottom bar) | Bottom 8% removed - a trimmed HUD strip, caught by the default bottom rung | **designed to detect** | `crop-bottom-8-default` |
| crop (slight zoom) | Center crop keeping 92% - caught by the default centred rung | **designed to detect** | `crop-centre-8-default` |
| edge crop, variants disabled | Bottom 15% removed - deeper than the default bottom rung reaches | not detected | `crop-bottom-default-config` |
| letterbox / aspect change | 15% black bars top and bottom | **designed to detect** | `letterbox-15pct` |
| watermark / overlay | Corner text overlay at 60% opacity | **designed to detect** | `watermark-corner` |
| color grading | Saturation +40%, gamma 1.15, brightness and contrast shifts | **designed to detect** | `colorgrade` |
| frame-rate resample | Resampled from 24 fps to 15 fps | **designed to detect** | `fps-resample-15` |
| cross-format laundering | Footage re-wrapped as an MCAP camera topic (or extracted from one) | **designed to detect** | `launder-mcap` |
| false-positive control | Entirely different footage must not match | not detected | `unrelated-footage` |
| false-positive control (same room) | Same environment and camera rig, a different part of the scene - must not match | not detected | `same-scene-different-view` |
| short clips (documented floor) | Clips shorter than the 10 s evidence floor are not detected | not detected | `clip-below-floor` |

*Every row of this table is an executable test:* [`tests/integration/test_detection_matrix.py`](tests/integration/test_detection_matrix.py) *generates each manipulation with ffmpeg and asserts the stated status - rows marked "not detected" are asserted too, so the table can neither over-claim nor under-claim.*
<!-- detection-matrix:end -->

Honest limits, stated plainly:

- **Different recordings of the same scene do not match.** overlap detects
  copied *footage*, not repeated *content*. A vendor who re-records
  the same kitchen produces new data.
- **Crops deeper than the indexed ladder are not detected.** The default
  ladder reaches ~30%; a copy cropped to 60% of frame or tighter falls
  outside it (matrix row `crop-40pct`). Extend `index.crop_ladder` if your
  counterparties crop harder - the cost is index size, not accuracy.
- **Clips shorter than the evidence floor (10 s by default) are not
  reported** - below that, perceptual evidence is too thin to accuse anyone.
- **Edge crops need `--crop-edges`** (off by default because each side adds
  10 codes per frame to the index). With default settings a bottom-strip crop
  is *not* detected; the matrix asserts both states.
- **Indexing below 4 fps degrades detection** of arbitrary trims,
  concatenations and speed changes. `overlap index` warns when you do it.
- **Long static scenes** (a robot idling) are excluded from matching as
  near-featureless and reported as *indeterminate*, never silently counted
  either way.
- A determined adversary who knows this tool can craft transforms that evade
  it. overlap raises the cost of casual fraud and documents exactly where its
  boundaries are. It is a detection tool, not a proof of authenticity.

## The manifest

A `.ovlm` manifest contains perceptual fingerprints (32 bytes per sampled
frame), file names (optionally anonymized), sizes, sha256 digests, and a
Merkle root binding it all together - about **120 KB per hour** of footage.
It contains no frames and no pixels. Note that perceptual fingerprints are
not encryption: they leak coarse visual structure by design. Treat manifests
as confidential business documents. Format spec: [docs/manifest-spec.md](docs/manifest-spec.md).

## The report

![comparison report](docs/assets/report.png)

The HTML report is a single self-contained file (no external assets) that a
lab can attach to an email to procurement - or back to the vendor.

## Scale

Corpus size is bounded by disk rather than memory: the search index is a set
of shards, only one resident at a time. Fingerprints cost about 6 MB per hour
of footage at the default settings, indexing runs at roughly 9x realtime per
core, and `overlap merge` lets several machines share the work. Capacity
planning, measured throughput, and a 500 TB projection:
[docs/scale.md](docs/scale.md).

## Install

```
pipx install overlap-cli          # or: uv tool install overlap-cli
pip install overlap-cli           # library use: `import overlap`
pip install 'overlap-cli[ros]'    # + ROS1 .bag / rosbag2 support
```

- **Supported inputs:** `.mp4 .mkv .avi .mov .webm` video, `.mcap` (image
  topics), ROS1 `.bag`, rosbag2 (with the `ros` extra).
- **ffmpeg is not required** - video decoding uses bundled FFmpeg libraries
  (PyAV). The ffmpeg CLI is only needed to run the test suite's fixture
  generation.
- **Windows is fully supported** (and is a primary development platform).
  See [docs/windows.md](docs/windows.md).
- Docker: `docker run -v /data:/data ghcr.io/world-archive/overlap index /data`
- Check your environment anytime: `overlap doctor`

## Configuration

Precedence: CLI flag > `OVERLAP_*` environment variable > `./overlap.toml` >
user config > default. `overlap config` prints every effective value **with
its source**. The index location defaults to your platform data directory;
point `--index` (or `OVERLAP_INDEX`) at a fast volume for large corpora.

Two defaults are worth knowing, both set from measurements documented in
[docs/architecture.md](docs/architecture.md):

- The local index samples at **4 fps** - the density where speed-changed and
  arbitrarily-trimmed copies stay inside the hash radius. Exported manifests
  stride back down to ~1 fps, so manifest size doesn't pay for it.
- **`index.crop_ladder = 0.94,0.88,0.82,0.76,0.70`** - each frame is also
  hashed at these centered crops so zoom-cropped copies are detectable.
  This is what the index spends its size on: disabling it
  (`--crop-ladder ""`) shrinks the index ~6x and makes crops invisible.

Neither affects manifests or the vendor side: a vendor's export is the same
either way, so a lab can deepen its own detection without asking anyone.

**Both sides can configure independently.** The only settings that must match
are the hash function and its normalization (`algo_id`/`prep_id`), and a
mismatch is refused rather than guessed at. Sampling rates need not agree - the matcher works in real time, so a manifest exported at 1 fps compares
correctly against a corpus indexed at 2, 4 or 8 fps. Lowering your own
sampling rate only weakens your own detection; it cannot corrupt a comparison
or make anyone else's manifest unreadable.

## Privacy & trust posture

- **Everything runs locally.** No telemetry, no update checks, no network
  I/O anywhere in the package - enforced by a test that scans every
  source file for network clients, stated in [SECURITY.md](SECURITY.md).
  Pull requests adding network calls are rejected.
- The web UI binds to loopback with token auth on by default.
- Manifests are untrusted input: parsing is strict, size-capped, and fails
  closed.

## Roadmap

- Off-center crop variants (centered and edge-aligned crops are covered; a
  crop anchored somewhere in between is not)
- A crop-invariant descriptor, so coverage stops costing index size
- HDF5 / LeRobot dataset and image-sequence readers (the reader interface is
  a public plugin point - `overlap.readers` entry-point group)
- Embedding-based second-tier verification for heavy grading/overlay cases
- Keyed / private-set-intersection manifest exchange
- Foxglove `CompressedVideo` (H.264-in-MCAP) topics

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) - including the two house rules:
claim hygiene (every detection claim cites a matrix row) and no network I/O.
The most valuable contribution is a **detection-gap report**: a manipulation
that slipped through, filed with the issue template, becomes a new matrix row.

## License

Apache-2.0 - see [LICENSE](LICENSE).
