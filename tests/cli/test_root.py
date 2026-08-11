from __future__ import annotations

import json

from typer.testing import CliRunner

import overlap
from overlap.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert overlap.__version__ in result.output


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output


def test_doctor_json_is_single_document(isolated_env) -> None:  # type: ignore[no-untyped-def]
    result = runner.invoke(app, ["--json", "doctor"])
    doc = json.loads(result.stdout)
    assert doc["overlap"] == overlap.__version__
    assert any(c["component"].startswith("numpy") for c in doc["checks"])


def test_config_json_reports_sources(isolated_env) -> None:  # type: ignore[no-untyped-def]
    result = runner.invoke(app, ["--json", "config"])
    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["values"]["index.fps"] == 4.0
    assert "paths" in doc
    assert doc["paths"]["index_dir"].endswith("corpus.ovl")
