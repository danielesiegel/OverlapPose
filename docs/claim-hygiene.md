# Claim hygiene

overlap's entire value is that both sides of a data transaction can trust
what it says. A single detection claim that turns out to be inflated costs
that trust permanently. These rules keep every claim tied to executed
evidence, and they are enforced in review (the PR template has a checkbox)
and by the test suite (`tests/unit/test_project_guards.py`, which checks
both the docs-matrix sync and the banned phrasing below).

## The rules

1. **Every statement about what overlap detects cites a row of
   [`tests/detection_matrix.toml`](../tests/detection_matrix.toml)** - the
   single source of truth. README and docs tables are generated from it by
   `scripts/gen_detection_matrix_doc.py`; a test fails when they drift.

2. **A new claim requires a new matrix row and a new fixture test in the
   same pull request.** No claim ships before its test.

3. **Under-claiming is also a violation.** Rows tiered `none` are asserted
   in tests; when a capability improves, the test fails with instructions to
   promote the row. The matrix must describe reality in both directions.

4. **Banned phrasing** (anywhere in docs, code comments, release notes):
   - "detects all …", "catches any …"
   - "guarantees", "guaranteed"
   - "tamper-proof", "fraud-proof", "impossible to evade"

5. **Approved verbs**, in decreasing strength:
   - *designed to detect* - only for `detect`-tier rows (hard-failing tests)
   - *robust to* - only for measured invariances (cite the measurement)
   - *best-effort against* - for `best-effort` rows
   - *does not detect* - for `none` rows and anything outside the matrix

6. **Numbers come from measurements**, not intuition. Thresholds, radii, and
   sampling rates cited in docs must reference where they were measured
   (tests or docs/architecture.md).

7. **Reports never assert novelty.** The strongest permitted claim for
   unmatched footage is: "absence of a match here is evidence, not proof, of
   novelty."

## Why this exists

A lab that catches one false "no-overlap" claim stops trusting the tool - and
tells other labs. A vendor falsely accused by an inflated detection claim has
a legitimate grievance. The detection matrix being executable is the
project's signature feature; this document is the editorial half of that
feature.
