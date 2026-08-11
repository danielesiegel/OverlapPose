# Architecture

```
           readers (plugin point)          hashing                store
  mp4/mkv ─┐                        ┌──────────────────┐   ┌────────────────┐
  MCAP    ─┼─► sampled frames ────► │ prep-v1 + pdq2   │ ► │ SQLite catalog │
  ROS bags─┘   (t_ms, pixels)       │ identity+mirror  │   │ + search shards│
                                    └──────────────────┘   └───────┬────────┘
                                                                   │
  vendor .ovlm manifest ──► candidates (ANN) ──► Hough chaining ──► scoring ──► report
```

## Package layout

| module | responsibility |
|---|---|
| `overlap.readers` | container -> sampled decoded frames; plugin point (`overlap.readers` entry-point group) |
| `overlap.hashing` | prep-v1 normalization + the pdq2 kernel (identity + mirror digests) |
| `overlap.ingest` | file discovery, multiprocess pipeline, sha256/Merkle |
| `overlap.store` | SQLite catalog (source of truth), sharded search index (derived), .ovlm manifests |
| `overlap.match` | candidate generation, diagonal-run chaining, scoring, compare/verify |
| `overlap.render` | report JSON -> self-contained HTML / Markdown |
| `overlap.cli`, `overlap.server` | the two front-ends over the same core calls |

Two seams are narrow so hot loops can move to native code
without touching callers: `overlap.hashing.base.HashKernel` and the matcher
in `overlap.match.chain`.

## The fingerprint (pdq2)

pdq2 follows the shape of Meta's published PDQ algorithm (256-bit DCT-sign
hash over a smoothed 64×64 luminance downsample) with one deliberate
deviation: the downsample stage is **bit-exactly mirror-symmetric** (fixed
256×256 pre-resize in float32, odd box windows, 4×4 block-mean decimation).
The reference pipeline is not; without this, flip detection is fragile on
low-asymmetry content.

**pdq2 is not interchangeable with reference PDQ, and no cross-implementation
distance is claimed here.** `algo_id` names this implementation rather than the
abstract algorithm, and a mismatch between two sides is refused rather than
approximated. What is verified, by tests that fail on drift:

| property | how it is pinned |
|---|---|
| exact bit behaviour | golden digests in `tests/unit/test_pdq.py` - a change here requires a new `algo_id`, not a fixed test |
| mirror symmetry | `hash(fliplr(x)) == mirror(x)` bit-for-bit, on two frame profiles |
| brightness shift (+24) | ≤ 4 of 256 bits |
| downscale/upscale re-encode proxy | ≤ 12 of 256 bits |
| unrelated frames | > 80 bits apart (random pairs concentrate at 128) |
| speed | a per-hash budget, so the 13x speedup cannot silently regress |

The name carries the version for exactly this reason: `pdq1` hashed at 512×512
in float64 and is not comparable, so the rename is what stops two incompatible
indexes from being silently mixed.

Every frame yields two digests (identity + horizontal mirror; the mirror
costs almost nothing - it's a column-sign flip in DCT space, re-thresholded).
The local index stores both; manifests carry identity only. A flipped copy's
identity hashes then match the original's mirror entries, and the hit parity
tells the matcher the match is mirrored. Content-based orientation
canonicalization was tried and rejected: frames with weak left-right
asymmetry select sides unreliably.

prep-v1 normalization before hashing: BT.601 luma, then letterbox/pillarbox
strip - border geometry is detected on a 30-frame probe (per-side median so
transient dark scenes don't trigger it) and applied as a *fixed* crop for the
whole stream, because per-frame detection would jitter hashes. Aspect-ratio
changes need no handling: the hash squashes everything to a square.

## What two sides must agree on

Vendors and labs configure their own indexes independently, so it matters
exactly which settings are part of the interchange contract:

| setting | shared? | why |
|---|---|---|
| `algo_id` (hash function) | **must match** | different hashes are not comparable; a mismatch is refused |
| `prep_id` (normalization) | **must match** | same reason - refused |
| sampling rate (`index.fps`) | free per side | a frame's hash depends on the frame, not on how often you sampled |
| crop / orientation variants | free per side | corpus-side only; manifests never carry them |
| export stride | free per side | each stream declares its own rate |

Sampling rates do not have to line up because the matcher works in real
milliseconds, not sample indices: both sides' timestamps are reconstructed
from their own declared rate, and shared footage still forms a straight line.
What a *denser corpus* buys is phase coverage - whatever instants the vendor
happened to sample, some corpus sample sits within half a corpus interval of
them. That is the entire reason the corpus default is 4 fps while manifests
export at ~1 fps: the side that can afford density should carry it, and the
side sending fingerprints should stay small.

The practical consequence is worth stating plainly: **lowering your own
`index.fps` only weakens your own detection.** It cannot corrupt a comparison
or make another party's manifest unreadable. Likewise a lab can enable deeper
crop variants unilaterally and immediately benefit, without asking any vendor
to re-export anything.

Settings are recorded per stream, so one index may legitimately hold streams
built at different rates or with different variant sets - `overlap index`
reports when it notices a change instead of refusing, because forcing a full
re-index of an archive to enable one variant would make the feature unusable.
`tests/integration/test_interop.py` pins all of this.

## Why the index samples at 4 fps

Frame hashes are computed on a fixed time grid. A speed-changed or
arbitrarily-trimmed copy samples the *same footage at different phases*, and
DCT-sign hashes drift under temporal phase shift when the camera moves.
Measured on realistic footage profiles (slow pan, fast pan, static scene
with local motion):

| phase shift | slow pan | static + local motion | fast pan |
|---|---|---|---|
| 42 ms  | 0 - 8 bits | 14 - 24 | 18 - 22 |
| 125 ms | 16 - 22    | 40 - 50 | 50 - 56 |
| 250 ms | 26 - 34    | 60 - 68 | 94 - 102 |

A 4 fps corpus grid bounds the worst-case misalignment at 125 ms, keeping
drift within the candidate radius (56 of 256 bits - random pairs concentrate
at 128 ± 8, so acceptance stays ~9σ from noise).

That is the mechanism. The consequence was then measured end to end, because
density is the largest cost driver in the tool and the temptation to lower it is
constant. 24 real egocentric episodes were indexed at four densities, and the
"offer" was those same episodes concatenated, re-cut at 137-second boundaries
that align with nothing, and transcoded - pure repackaging, no adversarial
tricks, of footage the corpus owns entirely:

| corpus fps | overlap reported | offer files matched | index MB/h |
|---|---|---|---|
| **4.0** | **99.5%** | 32 of 32 | 0.92 |
| 2.0 | 86.2% | 31 of 32 | 0.46 |
| 1.0 | 40.1% | 19 of 32 | 0.23 |
| 0.5 | 4.8% | 4 of 32 | 0.12 |

Everything below 100% is loss on footage that is entirely owned. So 4 fps is
load-bearing rather than cautious, and lowering it does not degrade detection
gently - it removes it. Two things follow:

- **Do not lower `index.fps` to save time.** It barely does: seek overhead
  replaces decode work, so the measured cost per hour of footage is nearly flat
  across these rates. The lever that actually saves time and space is the number
  of geometries per frame (`--preset`), which cuts both proportionally.
- **Manifests exported at ~1 fps are sized for sending, not for matching.** That
  is fine in the normal direction, where the manifest is the query against a
  dense corpus. It is not fine when a manifest becomes the *corpus* - importing
  one, or comparing two - so ask for `--stride 1` when that is the intent.

## Why Hamming, and why the magnitudes are discarded

Cosine similarity, dot product and Euclidean distance come up as alternatives.
On binary codes they are not alternatives: for +/-1 vectors `dot = n - 2*hamming`,
so all four are monotone transforms of one another and rank every pair
identically. Measured on real frames, the Spearman correlation between Hamming
and cosine over the same codes is 1.000000. Choosing between them is a
relabeling.

The real question is whether the *representation* should keep the DCT magnitudes
and be compared by cosine, instead of keeping only the signs. It should not, and
the reason is the opposite of what it looks like. Measured on 41 real frames, as
separation from the unrelated-pair baseline of each representation:

| manipulation | sign bits (Hamming) | float DCT (cosine) |
|---|---|---|
| re-encode, jpeg q40 | **7.7σ** | 2.8σ |
| brightness +24 | **7.4σ** | 2.8σ |
| centre crop 5% | **5.1σ** | 2.6σ |
| centre crop 15% | 1.3σ | 1.4σ |
| centre crop 20% | 0.6σ | 0.9σ |
| centre crop 30% | -0.3σ | 0.1σ |

Unrelated real frames sit at 121 +/- 16 of 256 bits - a tight noise floor. The
same frames under cosine on raw coefficients spread from roughly 0 to 0.8 (mean
0.284, sd 0.257), because egocentric footage shares a lot of low-frequency
structure. **Median-thresholding is what makes the representation
discriminative**; discarding magnitude is not a lossy shortcut taken to save
space. Keeping the floats costs 8-32x the storage, gives up bit-reproducibility
across platforms and Python versions, and at the crop depths where sign bits fail
cosine is also at the noise floor - 0.9σ is not a detector.

Deep embeddings are the same trade further along: crop-robust because they are
trained with crop augmentation, at 1.5-4 KB per frame against 32 bytes, on a GPU,
and not reproducible enough for two mutually suspicious parties to agree
bit-for-bit. `algo_id` exists so such a backend *could* be added; nothing about
the current one is in its way.

## Why the index stores a crop ladder

Cropping re-frames every pixel, so it moves a frame hash far more than any
codec or grading change does. Measured distance between hashes of the same
frame at two different crop scales:

| scale mismatch | 3% | 6% | 9% | 15% | 30% |
|---|---|---|---|---|---|
| median distance (of 256) | 22-38 | 52-60 | 74-90 | 100-114 | 124+ |

Walked a percent at a time on 61 real frames, against the plain uncropped hash,
the breaking points are sharp and asymmetric:

| crop | centred: frames caught | one-sided (bottom strip) |
|---|---|---|
| 3% | 61 of 61 | 61 of 61 (worst case 56, at the limit) |
| 4% | 61 of 61 | 55 of 61 |
| 5% | 61 of 61 (worst 56) | 14 of 61 |
| 6% | 45 of 61 | 1 of 61 |
| 8-9% | none | none |

**One-sided crops break at roughly half the depth of centred ones.** Removing a
bottom strip shifts content as well as narrowing the field of view, so the centred
rungs cannot recover it at any depth - it needs its own geometry. And what each
ladder depth buys, as the deepest crop where *every* frame is still caught:

| ladder | codes/frame | reaches |
|---|---|---|
| none | 2 | 5% centred, 3% one-sided |
| **1 centred + 1 bottom (default)** | **6** | **11% centred, ~9% one-sided** |
| 2 centred rungs | 6 | 15% centred, 3% one-sided |
| 5 centred rungs (`balanced`) | 12 | 33% centred, 3% one-sided |

That comparison decided the default. End to end on real footage the five-rung
ladder caught a 14% centred crop and **missed an 8% bottom strip entirely**, while
one rung each way caught both at half the codes. For egocentric hand data the
deeper coverage is defending against a crop that destroys the product - crop hard
and the hands leave the frame or the pixels are mush, so the footage stops being
sellable as manipulation data. The shallow one-sided crop is the one a reseller can
actually apply, so that is what the default pays for. `--preset balanced` and
`thorough` remain for adversarial depth on a small, high-value corpus.

A copy cropped by 15% is ~112 bits from the uncropped corpus frame - pure
noise. But it is only ~30 bits from a corpus frame hashed *at the same crop
scale*, which is well inside the candidate radius. So the corpus stores each
frame re-hashed at crop geometries. Rungs sit 6% apart because that is the
widest spacing where the worst-case mismatch (3%, mid-rung) stays under the
radius.

The variant a hit lands on is recovered from its position in the index, so
the matcher reports *which* crop depth matched - the report says "cropped
~15%" rather than merely "matched". Cost: index size scales with
`(1 + geometries) x 2 orientations` = 6 codes per frame by default; manifests
are unchanged, because the querying side sends plain uncropped fingerprints and
the corpus side does the work. Beyond the deepest rung detection stops - that boundary is asserted by matrix row `crop-40pct`.

Known limitation: the ladder is *centered*. A copy cropped off-center (say,
the right two-thirds of frame) shifts content relative to every rung and is
not reliably caught. Off-center variants are on the roadmap; adding them
multiplies index size, so they are off by default.

**Why the ladder sits on the corpus side.** Cropping composes - cropping a
cropped frame just crops it further - so in principle the ladder could live on
the offer side instead, which is one or two orders of magnitude smaller. It
cannot, and the reason is the threat model rather than the mathematics: the
offer's manifest is produced by the party being checked, and a seller cropping to
evade detection would simply not ship the crop variants. The ladder has to be
computed by whoever holds both the pixels and the incentive to detect cropping,
which is the buyer's own corpus.

A crop also cannot be handled by a cleverer invariant. Rotation and scale
invariance are achievable with a log-polar or Fourier-Mellin representation, but
a crop removes field of view: the content differs, so no global
descriptor of the frame can be invariant to it. Re-hashing at candidate
geometries is the cheapest way to align two fields of view without falling back
to keypoints, which cost 10-100x the storage.

## The matcher

Candidate hits between one query stream and one corpus stream are points in
the (query-time, corpus-time) plane. Shared footage lies on a line
`c = a·q + b`:

- slope `a` estimates the time mapping - `speed_ratio = 1/a`, so a file
  slowed to 0.5x yields a slope-0.5 line, reported as ~2x billable-hours
  inflation with the underlying footage quantified;
- several disjoint lines = a spliced file; a line covering part of the query
  = a trim.

Lines are found with a Hough transform over (log-slope, offset) - 37 slope
bins covering 0.125x - 8x - then refined by least squares. Frame-level
thresholds stay permissive; acceptance happens at the *run* level where
evidence accumulates: minimum 10 s span, ≥ 8 distinct matched samples,
density ≥ 0.5, mean best-per-slot distance ≤ 42. Tiers: `exact` (sha256),
`strong`, `probable` (headline), `weak` (never in the headline, but always
*reported* - see below).

Three rules exist because of a specific failure. A file assembled from two
non-adjacent pieces of one master draws two parallel diagonals against the
same corpus stream, and a compromise slope through both clusters can outvote
either true line:

1. **A peak may not consume its hits until it looks like a real diagonal**
   (density ≥ 0.3). Otherwise the compromise run swallows the evidence both
   genuine runs needed, and neither is ever found.
2. **Inlier sets are cut at gaps in query time.** One long sparse smear
   across two segments becomes two dense runs, which is the real structure.
3. **Long, dense runs earn a higher distance ceiling** (mean distance up to
   64 instead of 42 when a run spans ≥ 15 s at ≥ 60% density). Unrelated
   footage does not produce 15-second diagonals at 60% density, whereas
   phase-misaligned copies of *identical* footage produce exactly that: every
   frame is a near-miss because the corpus never sampled the instants the copy
   did. Judging such a run on per-frame crispness alone discarded an 18 s
   match at 75% density.

**Concatenation is counted on both timelines.** A file reassembled from two
cuts of one master is *continuous in the offer* - the pieces sit end to end - and the jump appears only in the corpus timeline. Counting gaps on the
offer's timeline alone reports one segment for exactly the manipulation this
is meant to expose, so a matched run only continues the previous segment when
it is adjacent on both axes and comes from the same source.

**Weak evidence is never silent.** A run that clears the structural bar but
not the reporting threshold is excluded from the overlap percentage - and
listed separately, with its own section in the report and a note on the CLI.
A bare "0%" that quietly hid a 20-second weak match would be the most
damaging thing this tool could print.

Static-scene degeneracy is handled twice: near-featureless frames are
flagged at hash time and excluded from the index, and corpus frames hit by
an outsized share of query frames are suppressed as hubs - their spans are
reported as *indeterminate* rather than silently matched or unmatched.

## Storage

An index directory (`*.ovl/`) holds `catalog.sqlite` (WAL; the source of
truth: files, streams, hash chunks) and `ann/` (search shards). Indexing
commits one transaction per file, which is the resumability unit: interrupt
anytime, re-run to continue.

### Shards, and why the index grows instead of rebuilding

The search structure is not one file but a set of independent shards, each
holding up to `index.shard_codes` codes (default 32M, about 1 GB) with its own
IVF clustering and its own mapping back to `(stream, frame, variant)`. Two
consequences, both of which are the point rather than a side effect:

**Only one shard is resident at a time.** Peak memory during a comparison is
one shard plus the query set, whatever the corpus size, so a corpus is bounded
by disk rather than by RAM. Search iterates shards on the outside and query
batches on the inside, so each shard is read exactly once per comparison.

**Adding footage does not rebuild the index.** Which shard holds a stream is
recorded in the catalog (`ann_shards`, one row per stream, with the frame and
rung counts it was built from). Re-opening an index compares that against the
current catalog: shards whose streams are all unchanged are kept, new streams
are built into new shards, and a stream that was re-indexed invalidates only
the one shard that held it. Adding a terabyte to a ten-terabyte index costs a
terabyte of work, not eleven.

Shards also record their own `codes_per_frame` and crop ladder, so extending
`index.crop_ladder` does not invalidate existing shards - older ones simply
carry fewer variants, the same way per-stream `n_crop_rungs` already works.
Variant indices from every shard are remapped into one union ladder when the
index is opened, so a hit reports the same geometry regardless of when it was
built.

Everything under `ann/` is derived: deleting it is always safe and costs only
rebuild time. Shards below 1M codes use exact Hamming search; larger ones use
IVF.

## Scale notes

Indexing cost is dominated by two things: decoding the video, and hashing
every sampled frame in every indexed variant. With the default 4 fps grid and
the centered crop ladder that is 12 hashes per sampled frame, or 48 PDQ
hashes per second of footage - so hashing is *not* a rounding error on top of
decode, and adding crop variants scales it linearly.

Measured end to end on real egocentric footage (456x256, 3-minute clips) at the
default 4 fps, on four cores of one desktop. **One machine, one dataset**: treat
these as the right order of magnitude and the correct relative comparisons, not
as constants. `benchmarks/` has the scripts to re-run them on your own hardware.

| quantity | measured |
|---|---|
| indexing | **0.106 core-seconds per second of footage** at 12 codes/frame (9.4x realtime per core) |
| storage | **~72 bytes per frame per indexed code**, about 2.2x the digest payload |
| sweep | 2.3e8 code comparisons per second on 4 cores |
| exact comparison | 4.4 s for a 21-stream offer against 494 nominated pairs |

Work is parallel across files, so more cores scale close to linearly, and
`overlap merge` lets that span machines. Manifest export is ~1 s.

Disk, not RAM, is what a large corpus consumes - and it costs more than the
digests do. A digest is 32 bytes, but a frame occupies **about 2.2x its payload** once
everything real is counted: the catalog holds an identity and a mirror digest and
the same pair per crop rung, then the shards hold those codes again alongside IVF
centroids and the frame-to-stream mapping. Measured on 260 real clips, 12.97 h of
footage came to 68.5 MB of catalog plus 92.1 MB of shards - 860 bytes per frame at
12 codes per frame. Pricing a run on the payload alone under-reports it by roughly half, which at archive scale is the difference between a corpus that fits and
one that fills the volume partway through a three-week job.

At 4 fps that is 6.2 MB per hour for the default (6 codes), 12.4 for `balanced`
and 53.7 for `thorough`. That is the main tuning dial: lower `index.fps`, shorten
`index.crop_ladder`, or drop edge variants, depending on which manipulations you
actually need to catch. `overlap index` prints the coverage and cost of whatever
configuration you chose.

### Why there is no GPU decode

Decode, not hashing, is the larger half of indexing once the crop ladder is
shallow: measured at 77% of the work at `--preset fast`. The footage forces it.
These clips are 30 fps with keyframes 8.33 s apart, so sampling at 4 fps still
means decoding every frame and discarding 87% of them - inter-frame coding gives
no way to reach frame N without its references, and decoding keyframes only
yields 0.12 fps, far too sparse to match on.

That makes hardware decode the obvious lever, and it was measured rather than
assumed. Two findings, on a GTX 1080 against 456x256 H.264:

**Agreement is fine.** Piping frames from the system ffmpeg instead of PyAV is
bit-identical (2,698 frames, zero difference). NVDEC then differs from software
decode on 22% of frames, by a median of 0 and a worst case of 4 bits of 256 -
the same order as a jpeg re-encode, against a candidate radius of 56. The cause
is the colour conversion, not the decode: H.264 decoding is normatively
bit-exact, but NVDEC hands back NV12 where software gives YUV420P, and libswscale
takes different chroma paths for the two.

**Speed is not.** Decode-only aggregate throughput, frames per second:

| concurrent streams | CPU (software) | GPU (NVDEC) |
|---|---|---|
| 1 | 1,961 | **2,821** |
| 2 | 3,509 | **5,378** |
| 4 | 6,514 | 6,214 |
| 8 | **10,451** | 6,077 |

A consumer GPU has one NVDEC engine, so it saturates near 6,200 fps and then
degrades. CPU decode keeps scaling, because indexing already runs a worker per
core. For 10.43 billion frames that is 19.5 days GPU-bound against about 8 days
on 16 cores - so on this workload the GPU is ~2.4x *slower*, and it still leaves
the hashing to the CPU.

The reason is frame size. CPU decode cost scales with pixels while NVDEC is fixed
throughput, so hardware decode pays off on 1080p and 4K, or wherever cores are
scarce - one or two streams show a 1.5x gain. It does not pay off on small frames
with many cores. Buying cores is the lever here, which is why `overlap merge`
exists and `index.hwaccel` does not.

### A 500 TB delivery, projected from those constants

A corpus that size cannot be staged here, so this is the cost model with the
measured constants substituted in - not a measurement. The model was checked
against sweeps of a real 2.24M-code index at 1, 2, 4 and 8 shards and explains
them within 1.7x; the slowest fitted rate is used, so these err long. Comparison
figures are for a 5,000-hour offer, the size a large delivery runs to.

| 500 TB of | hours | per hour | fingerprints | fingerprinting (one-time) | compare 5,000 h |
|---|---|---|---|---|---|
| 1080p vendor footage | 171,000 | 6.2 MB/h | 1.06 TB | 447 core-days (22 d on 20 cores) | 0.6 d |
| 720p vendor footage | 370,000 | 6.2 MB/h | 2.29 TB | 966 core-days (48 d on 20) | 1.3 d |
| low-res training footage | 1,790,000 | 6.2 MB/h | 11.09 TB | 4,675 core-days (234 d on 20) | 6.4 d |
| low-res, `--preset fast` | 1,790,000 | 2.1 MB/h | 3.70 TB | 2,503 core-days (125 d on 20) | 2.1 d |

Comparison figures assume 16 GB shards; 64 GB shards halve them.

Three things this table is saying:

- **Fingerprinting dominates, and it is one-time.** It is also the part that
  parallelises perfectly across machines, which is what `overlap merge` exists
  for. At the extreme end - half a petabyte of low-bitrate training footage with
  every crop geometry indexed - one machine is not a sensible plan; ten is.
- **Coverage is the biggest single dial.** The last row is the same footage as
  the third with the crop ladder off: 3x less storage and nearly half the time, at
  the cost of not detecting cropped copies at all. `overlap index` prints that
  trade before it starts so it is chosen rather than inherited.
- **Shard size buys comparison speed.** The same 1080p corpus sweeps in 4.9 days
  at 1 GB shards, 1.2 days at 16 GB, and 0.6 days at 64 GB. This is the one place
  where RAM buys speed rather than capacity.

Memory during a comparison is set by `index.shard_codes`, not by corpus size - one shard plus the query set. Raise it on a large machine to search fewer,
larger shards (which is also faster, see below); lower it to work through a
multi-terabyte index on a small one.

### Comparison cost, and the cascade

A comparison runs in two stages, because the accurate stage and the expensive
stage do not have to be the same stage:

1. **Sweep** - every `compare.probe_stride`-th offered frame (default every
   4th) is searched against the whole index. This nominates *pairs of streams*
   and nothing more.
2. **Compare** - each nominated pair is then searched exactly, every offered
   frame against every code of that one corpus stream, straight from the
   catalog with no quantizer in the path.

Only stage 1 grows with the size of the archive, and it is the sparse,
approximate one. Stage 2 decides what the report says, and its cost is bounded
by the two streams being compared rather than by the archive around them - two
three-minute clips are about a million code comparisons.

This is why recall does not depend on search tuning. A pair needs one matching
frame out of the ten that the shortest acceptable run contains, and once
nominated, the run's density, inlier count and geometry are recovered in full.
An `nprobe` set too low for a 10 TB corpus, or a trim that lands badly against
the coarse grid, costs nothing.

Sweep cost grows as `sqrt(N x S)` for `N` total codes in `S` shards, since each
shard's quantizer is probed independently. Fewer, larger shards are therefore
cheaper to search as well as being the reason to have more RAM - which is the
one place where memory buys speed rather than capacity.

### Storage backends: what is and is not pluggable

Stated plainly, because it bounds who can use this at what scale:

| layer | implementation | pluggable |
|---|---|---|
| readers (containers) | PyAV, mcap, rosbags | **yes** - `overlap.readers` entry points |
| similarity search | sharded FAISS binary | **no** - but the shard set is the seam |
| catalog / metadata | SQLite | **no** - hardcoded |

Neither storage layer is abstracted behind a protocol, and that is a decision
rather than an omission. A second catalog implementation would be cost without a
user: SQLite handles the metadata volume of any single-machine corpus, and
`overlap.store.catalog` is the only module that touches the database if that ever
changes.

Similarity search once needed a backend seam because it had a hard ceiling - the
index was held in memory, so corpus size was bounded by RAM. Sharding removed
the ceiling, which removed the reason for the seam. What replaced it is a
different and more useful boundary: a shard is a self-contained file plus a
catalog row, so shard *production* is separable from shard *search*. That is the
seam to use for work an external database would otherwise be asked to do.

A note on why a general-purpose vector database is not the answer here: matches
are accepted out to a Hamming radius of 56 of 256 bits, which is 22% of the
code. Multi-index hashing and LSH schemes degenerate at that radius - the
partitions needed to guarantee a match become small enough that lookups touch
most of the table - so any backend ends up doing the same IVF-or-scan work over
the same bytes, with a network hop added.

### Choosing the sampling rate

4 fps is the default because it is the point where phase-sensitive
manipulations stay detectable (see above). 2 fps halves index size and
indexing time, and is a reasonable choice for a first pass over a very large
archive, but expect arbitrary trims, concatenations and speed changes to slip
into the weak tier or below - `overlap index` prints a note when you go below
4 fps for exactly this reason.
