# CLI reference

Global options (before the command): `--index DIR` (or `OVERLAP_INDEX`),
`--config FILE`, `--json`, `--quiet/-q`, `-v`. Conventions: **stdout carries
data, stderr carries progress and logs**; `--json` switches long commands to
NDJSON event streams and short commands to a single JSON document.

## Exit codes

| code | meaning |
|---|---|
| 0 | success (for `compare`: ran cleanly; below `--fail-over` if given) |
| 1 | runtime error (unreadable index, bad manifest, failed verification) |
| 2 | usage error |
| 3 | `compare --fail-over`: overlap at or above the threshold |
| 4 | completed with some inputs failed (details on stderr / in events) |
| 130 | interrupted; progress saved, re-run to resume |

## Commands

### `overlap index PATHS...`

Fingerprint videos and robotics containers into the local index. Resumable:
unchanged files (path + size + mtime) are skipped; `--reindex` forces.
Options: `--fps` (corpus sampling density, default 4.0), `--crop-ladder`
(centered-crop keep fractions indexed per frame so zoom-crops are
detectable; default `0.94,0.88,0.82,0.76,0.70`, pass `""` to disable and
shrink the index ~6x), `--crop-edges` (also index one-sided strip crops,
e.g. `bottom,top`; off by default, each side adds 10 codes per frame),
`--preset fast|balanced|thorough` (coverage vs cost in one flag; an explicit
`--crop-ladder` still wins), `--workers N` (0 = auto), `--include GLOB` /
`--exclude GLOB` (repeatable), `--follow-symlinks`, `--dry-run`.

Before starting it prints what the chosen configuration detects, what it is
blind to, and what it costs per hour of footage. Under `--json` that arrives as
a `coverage` event, ahead of the `start` event, so a pipeline can record what a
given index was built to catch.

Indexing below 4 fps prints a warning: a copy cut at an arbitrary time
samples instants the corpus never sampled, and below 4 fps that phase gap
costs enough hash bits to lose trims, concatenations and speed changes.

### `overlap export`

Write a `.ovlm` manifest of the indexed corpus. Options: `-o PATH`,
`--label TEXT`, `--anonymize-paths` (content-derived file names for pre-sale
manifests), `--stride N` (0 = auto: stride to ~1 fps; pass 1 for full density,
which is what a manifest meant to be *imported* needs - see below).

`--only PREFIX` (repeatable) exports only files whose path starts with that
prefix; `--only-from FILE` reads prefixes one per line. Without either, the whole
index is exported. Use it to quote one dataset out of a mixed inventory without
keeping a separate index per sellable unit, and to carve a subset out of a large
corpus for a buyer. A prefix that matches nothing is an error rather than an
empty manifest. The subset is a normal manifest with its own Merkle root - it
describes the subset, and cannot be passed off as the parent corpus.

```
overlap export -o kitchen-only.ovlm --only teleop-2026q3/kitchen/ --stride 1
```

`--split-gb N` writes a directory of parts of about N GB each instead of one
file, splitting on file boundaries so every part is a complete readable manifest
and can be fetched or re-fetched on its own. Each entry in `parts.json` records
`first_relpath`/`last_relpath` and the directory `prefixes` it spans, so a
consumer can tell which part holds a given file instead of guessing. Needed past a few GB: at 4 fps,
96,000 hours of fingerprints is ~44 GB, more than most hosts will serve as a
single object. A `parts.json` records each part with its digest, and every part
carries the whole set's Merkle root so a subset cannot be passed off as the
complete manifest. `compare`, `import`, `audit-sample` and `inspect` all accept
either form.

### `overlap compare MANIFEST`

Compare an incoming manifest against the local index. Writes report JSON
(`-o`, default `report.json`); `--html PATH` renders the self-contained HTML
report; `--open` opens it. Tuning: `--min-run SECONDS` (evidence floor,
default 10), `--tier strong|probable|weak` (lowest tier to include),
`--nprobe N` (sweep recall knob), `--threads N` (cap search parallelism; 0 =
every core, which is wrong on a machine someone else is using). CI gating:
`--fail-over PCT` exits 3 when overlap ≥ PCT.

`--against OTHER.ovlm` compares two manifests instead of manifest-against-index - for when two sellers offer the same footage and you own neither yet. Coverage is
narrower there: neither side supplies pixels, so cropped copies cannot be found,
and the report carries a `coverage_note` saying so.

Crop depth in a report is the nearest *indexed* rung, not a measurement of
the copy - a 15% crop is reported as `~18%` because that is the variant it
matched. Labels carry a tilde for this reason.

The report separates *counted* overlap from **weak evidence**: runs with real
matching structure that fall below the reporting threshold are listed in
their own section and summarised as `weak_only_files`, never folded into the
headline and never silently dropped. `--tier weak` promotes them into the
numbers.

### `overlap verify MANIFEST --data DIR`

Post-delivery check: every file promised in the manifest must be present
byte-identically under DIR (matched by content, so renames are fine).
Reports missing/extra files and validates the manifest's own Merkle root.
Exit 1 on failure.

### `overlap import MANIFESTS...`

Load fingerprints from `.ovlm` manifests into the local index, without the footage
behind them. Use it to screen offers against data you have not bought: a
published dataset's fingerprint set, or an offer you declined that may return
under another name. Idempotent by content, so re-running is free.

Two limits, both reported rather than assumed away. A manifest carries no pixels,
so no crop geometries can be built for imported footage and a cropped copy of it
will not be found. And a manifest carries whatever density the exporter chose:
below 4 fps, recall against re-cut footage drops sharply (see
docs/architecture.md), so prefer manifests exported with `--stride 1`.

### `overlap audit-sample MANIFEST --sample DIR`

Check that a manifest describes the same footage as the sample the seller shared.
The sample is the only footage a buyer holds pixels for before paying, so it can
be fingerprinted locally and looked up in the manifest: if it was drawn from the
offered data, nearly all of it must appear. Reports the share found, which sample
files are absent, and how little of the offer the sample represents.

Fingerprints the sample at the manifest's own density by default, so both
sampling grids land on the same instants. It cannot prove the *rest* of the
manifest corresponds to footage the seller holds - `overlap verify` does that
once bytes arrive.

### `overlap merge SOURCES...`

Merge other index directories into this one, so a corpus can be fingerprinted
by several machines in parallel and combined afterwards. Hash chunks are moved
as stored blobs - never recomputed - so the expensive step happens once per
file however many machines take part. Files already present by content are
skipped, making a merge idempotent and safe to re-run after an interruption;
slices whose `algo_id`/`prep_id` do not match are refused rather than mixed.

Merged streams are sharded incrementally, so the search build covers the new
footage only. `--no-shard` defers that to the next comparison.

### `overlap self-dedupe`

Compare the corpus against itself, excluding each stream's trivial
self-match. Vendors run this before quoting; labs use it to find internal
duplication. Same report format as `compare`.

### `overlap inspect TARGET`

Describe a media file (streams/topics, codecs, durations; `--deep` decodes a
probe frame), a `.ovlm` manifest (counts, hours, merkle root), or a report
JSON (summary).

### `overlap report REPORT.json`

Re-render a saved report: `--format html|md`, `-o PATH`, `--open`.

### `overlap ui`

Serve the local web UI (default `127.0.0.1:8377`, token auth on - the URL
printed contains the token). `--no-browser` for SSH sessions,
`--no-token` for single-user machines. Over SSH:
`ssh -L 8377:127.0.0.1:8377 user@server`.

### `overlap status` / `overlap doctor` / `overlap config`

Corpus statistics · environment checks (attach `doctor --json` to bug
reports) · effective configuration with each value's source.

## Configuration file

`./overlap.toml` (project) or the user config directory (see
`overlap config` for the resolved path):

```toml
[index]
fps = 4.0
crop_ladder = "0.94,0.88,0.82,0.76,0.70"
crop_edges = ""        # e.g. "bottom,top" - catches strip crops, ~3x index
workers = 0            # 0 = auto

[compare]
min_run_s = 10.0
nprobe = 64

[ui]
port = 8377
token_auth = true
max_upload_mb = 512

[paths]
# index = "D:/fast-volume/corpus.ovl"
```

Environment variables: `OVERLAP_INDEX` plus `OVERLAP_<SECTION>_<KEY>` for any
key above (e.g. `OVERLAP_COMPARE_NPROBE=128`).
