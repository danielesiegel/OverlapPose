# Task runner. Install: https://github.com/casey/just (winget install Casey.Just)

default:
    @just --list

# Create/refresh the dev environment
sync:
    uv sync --group dev --extra ros

# Run the fast test suite (no ffmpeg needed)
test:
    uv run pytest -m "not integration"

# Run everything, including ffmpeg-based integration tests
test-all:
    uv run pytest

lint:
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy

fix:
    uv run ruff check --fix .
    uv run ruff format .

# Regenerate the detection-matrix docs from tests/detection_matrix.toml
matrix-docs:
    uv run python scripts/gen_detection_matrix_doc.py

# Everything a release needs: lint, types, full suite (run on each OS)
release-check: lint test-all
    uv build

ui:
    uv run overlap ui
