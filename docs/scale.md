# Scale

The corpora this is for are measured in hundreds of terabytes, so the limits
are stated plainly.

**Corpus size is bounded by disk, not memory.** The search index is a set of
independent on-disk shards; one is resident at a time, so peak memory during a
comparison is a single shard plus the offer being compared, whatever the corpus
size. `index.shard_codes` sets that ceiling directly (default ~1 GB per shard).
Fingerprints cost a measured **6.2 MB per hour** of footage at the default 4 fps
and crop geometries - about 2.2x the digest payload, because the catalog and the search
shards each hold the codes and the shards add their own structure. 500 TB lands
between 1.1 TB and 11 TB depending on bitrate and the coverage you choose;
`overlap index` prints the figure for your configuration before it starts.

**Re-running never loses work, and never redoes it.** Indexing commits one
transaction per file, so interrupting is safe and a re-run skips what is
already done. Shard membership is tracked per stream, so adding footage builds
shards for the new footage only, and re-indexing one file rebuilds only the one
shard that held it. Adding a terabyte to a ten-terabyte index costs a terabyte
of work.

**A corpus can be larger than one machine can fingerprint.** Indexing runs at a
measured 9.4x realtime per core, which makes 500 TB of 1080p footage about 22 days
on 20 cores - or under 3 days on ten machines. So machines fingerprint slices in
parallel and `overlap merge` combines the results, moving hash blobs rather than
recomputing them:

```
# on each machine, over its own slice
overlap --index slice-07.ovl index /mnt/delivery/shard-07

# then, once
overlap --index corpus.ovl merge slice-*.ovl
```

Merging is idempotent, so it is safe to re-run, and refuses slices whose
fingerprints are not comparable rather than mixing them.

**Comparison cost does not scale with the archive the way you would expect.** A
sparse sweep finds which corpus streams an offer touches; each nominated pair is
then compared exactly against the catalog. Only the sweep grows with corpus size,
and it is the cheap approximate stage - so detection quality does not depend on
search tuning. `compare --threads` caps parallelism on shared machines.

Comparing a 5,000-hour offer against a 500 TB 1080p corpus projects to about
0.6 days on 20 cores. Full measurements, the cost model, and where it was
validated: [docs/architecture.md](architecture.md).

