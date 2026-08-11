## What

<!-- One or two sentences: what does this PR change and why. -->

## Checklist

- [ ] Tests added or updated for behavior changes
- [ ] `CHANGELOG.md` `Unreleased` section updated (user-visible changes)
- [ ] **Claim hygiene**: any new or changed statement about what overlap
      detects cites a row in `tests/detection_matrix.toml`, and new claims add
      a matrix row + fixture test in this same PR (see `docs/claim-hygiene.md`)
- [ ] `uv run pytest` passes locally (includes the project guards:
      no network I/O in `src/`, docs in sync with the detection matrix)
