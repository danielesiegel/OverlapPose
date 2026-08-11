# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Pre-1.0 and not yet stable: minor versions may change CLI flags, the manifest
schema, or the index database schema. Such changes are always listed here, and
patch versions never make them. Pin an exact version if you depend on this.

## [Unreleased]

### Changed - breaking

- **Fingerprint is now `pdq2`** (256x256 working resolution in float32). `pdq1`
  hashed at 512x512 in float64 and is not comparable, so indexes and manifests
  built with it must be rebuilt. 13x faster per hash: 92% of the old cost was a
  smoothing filter that upsampled typical robotics footage before smoothing it.
  All 29 detection matrix rows unchanged.
- **Manifest extension is `.ovlm`**, not `.olm`, which is Outlook for Mac's
  archive format. Nothing else about the format changed.
- **Default crop geometries are now one centred rung plus one bottom rung**
  (6 codes/frame, was 12). Measured on real frames: the plain hash covers 5%
  centred and only 3% one-sided, one rung each way reaches 11% and ~9%, and end
  to end the old five-rung default caught a 14% centred crop while *missing an 8%
  bottom strip entirely*. `--preset balanced|thorough` restores deeper coverage.
- `index.hwaccel` removed. It was read by nothing. Measured on a GTX 1080, GPU
  decode is ~2.4x *slower* for this workload than CPU decode across many cores,
  because one NVDEC engine saturates while CPU decode scales per core.

### Added

- **`overlap import`** - load a manifest's fingerprints into an index, so offers
  can be screened against footage nobody owns: a published dataset, or an offer
  that was declined and may return under another name.
- **`overlap compare A.ovlm --against B.ovlm`** - compare two manifests directly,
  for when two aggregators offer the same data and the buyer owns neither.
- **`overlap audit-sample`** - check that a manifest describes the same footage as
  the sample a seller shared, which is the only footage a buyer holds before
  paying.
- **`overlap merge`** - combine indexes built on separate machines, moving hash
  blobs rather than recomputing them, so a corpus can exceed what one machine can
  fingerprint.
- **`export --split-gb N`** - write a manifest as a directory of parts, each a
  complete manifest, since 96,000 hours at full density is ~44 GB.
- **Sharded search index.** The index is a set of bounded on-disk shards, so peak
  memory is one shard plus the query set rather than the whole corpus, and growth
  is incremental: adding footage builds shards for the new footage only.
- **`--preset fast|mild|balanced|thorough`** and a coverage summary printed before
  indexing starts, so detection depth and its cost are chosen rather than
  inherited.
- `compare --threads` to cap search parallelism on shared machines.
- Mirror digests are derivable from identity digests, so a manifest carrying one
  digest per frame still catches mirrored resale.

### Fixed

- Run density was measured against the query's own sampling rate, so a query
  denser than the corpus could not reach the acceptance threshold at all: a 4 fps
  side auditing a 1 fps manifest reported 0% overlap on footage identical apart
  from a transcode.
- A Hough peak fitted *between* two parallel diagonals could outvote both and
  consume their hits, so a file reassembled from two cuts of one master reported
  0% overlap. Such a peak is now split on gaps in either timeline, and discarded
  without consuming hits if no split survives.
- Manifest headers advertised the density the *index* was built at while the
  streams carried the strided density.
- Unreadable media aborted a run with `AttributeError` instead of being skipped
  and reported: the video reader caught `av.AVError`, an alias PyAV removed in
  version 12.
- Shards below the IVF threshold were scanned exhaustively, making a comparison
  13x slower for anyone who lowered `index.shard_codes` to fit a small machine.

### Added (initial release)

-  perceptual fingerprint indexing (`index`), manifest export
  (`export`), corpus comparison (`compare`), post-delivery verification
  (`verify`), inventory self-deduplication (`self-dedupe`), file/manifest
  inspection (`inspect`), report rendering (`report`), and a local web UI
  (`ui`).
- Mirror-symmetric PDQ fingerprint (identity + mirror digests; flipped copies
  are caught without extra work).
- Temporal matcher: Hough diagonal-run detection; speed changes are measured
  from run slope, so slowed-down copies (billable-hours inflation) are
  detected and quantified; splices and trims localized to segments.
- Crop-geometry indexing: each corpus frame is additionally hashed at crop
  geometries, so cropped copies are detected and the report states the depth.
  Manifests are unaffected.
- Optional edge-crop variants (`--crop-edges bottom,top`) for one-sided
  strip crops, which centered rungs cannot recover because removing an edge
  changes the aspect ratio.
- Concatenation reporting: matched segments and distinct owned sources are
  counted per file, so a master split and resold as a "new session" is
  flagged, not just multi-source splices.
- Weak-tier evidence is reported separately instead of being dropped, so a
  0% headline can never hide a partial match.
- Matched runs report the time they actually cover, so a run spanning a gap
  no longer credits the gap as matched footage.
- Interop is explicit: only `algo_id`/`prep_id` must match between two sides;
  sampling rate and crop variants are private choices, verified by tests. An
  index may be extended with different settings instead of requiring a full
  re-index, and `overlap index` reports the change.
- Readers for video containers (mp4, mkv, avi, mov, webm), MCAP image topics,
  ROS1 bags, and rosbag2 (`[ros]` extra); reader plugin entry-point group
  `overlap.readers`.
- Manifest format v1: compact (~120 KB/hour), integrity-checked, Merkle-rooted,
  parsed fail-closed as untrusted input.
- Executable detection matrix: every detection claim in the documentation is
  backed by a generated-fixture test, in both directions (over- and
  under-claiming both fail the suite).
