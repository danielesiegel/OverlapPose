# Contributing to overlap

Thanks for your interest in improving overlap. This document covers setup,
workflow, and the two house rules that are specific to this project.

## Development setup

overlap uses [uv](https://docs.astral.sh/uv/) for environment management.
Any platform works; Windows is a first-class development environment and the
walkthrough below shows PowerShell first.

```powershell
# Windows (PowerShell)
git clone https://github.com/World-Archive/overlap.git
cd overlap
uv sync                       # creates .venv and installs everything incl. dev tools
uv run overlap --version
uv run pytest -m "not integration"
```

```bash
# Linux / macOS
git clone https://github.com/World-Archive/overlap.git
cd overlap
uv sync
uv run overlap --version
uv run pytest -m "not integration"
```

The integration suite additionally needs **ffmpeg** on PATH (used only to
generate tiny synthetic test clips - no media files are committed to the
repository):

- Windows: `winget install Gyan.FFmpeg`
- macOS: `brew install ffmpeg`
- Debian/Ubuntu: `apt install ffmpeg`

Then run the full suite with `uv run pytest`.

Lint and type-check before pushing:

```
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Or install the pre-commit hooks once: `uv run pre-commit install`.

## UI contributions need only Python

The web UI is server-rendered Jinja2 + htmx. There is no Node toolchain, no
bundler, and no build step: edit a template or `src/overlap/server/static/*`,
restart `overlap ui`, refresh the browser.

## House rule 1: claim hygiene

Every statement about what overlap detects is governed by
[docs/claim-hygiene.md](docs/claim-hygiene.md). The short version:

- Detection claims live in `tests/detection_matrix.toml` and nowhere else.
  README and docs tables are **generated** from it.
- A new claim requires a new matrix row and a new fixture test **in the same
  pull request**.
- Banned phrasing: "detects all", "guarantees", "tamper-proof". Approved
  verbs: *designed to detect*, *robust to*, *best-effort against*, *does not
  detect*.

## House rule 2: no network I/O

overlap runs entirely offline by design. Pull requests that add any network
call - telemetry, update checks, CDN assets, remote fonts - will be declined.
`tests/unit/test_project_guards.py` enforces it by scanning every source
file for network-client imports and raw socket use.

## Pull requests

- Keep PRs focused; one logical change per PR.
- Add or update tests for behavior changes.
- Update the `Unreleased` section of `CHANGELOG.md`.
- `uv run pytest` must pass locally before you open the PR (maintainers
  run the suite on Linux, macOS, and Windows before releasing).

## Releases

There is no hosted CI on this project - checks run in the normal test suite
(`uv run pytest`), which includes project guards for offline behavior and
documentation/matrix sync. Maintainers run the full suite on Linux, macOS,
and Windows before tagging a release, then build and publish with
`uv build`.

## Reporting a manipulation that slipped through

If a manipulated file matched (or failed to match) contrary to the detection
matrix, please use the **Detection gap** issue template. These reports feed
directly into new matrix rows and are the most valuable issues we receive.
