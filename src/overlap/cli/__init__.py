"""CLI assembly.

Command modules live next to this file; each exposes one function that is
registered here. Global flags are handled in the root callback and stored in
an :class:`~overlap.cli._console.AppState` on the typer context.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

import overlap
from overlap.cli._console import AppState
from overlap.cli.audit_cmd import audit_sample_cmd
from overlap.cli.compare_cmd import compare_cmd
from overlap.cli.config_cmd import config_cmd
from overlap.cli.dedupe_cmd import self_dedupe_cmd
from overlap.cli.doctor import doctor
from overlap.cli.export_cmd import export_cmd
from overlap.cli.import_cmd import import_cmd
from overlap.cli.index_cmd import index_cmd
from overlap.cli.inspect_cmd import inspect_cmd
from overlap.cli.merge_cmd import merge_cmd
from overlap.cli.report_cmd import report_cmd
from overlap.cli.status import status
from overlap.cli.ui import ui
from overlap.cli.verify_cmd import verify_cmd
from overlap.config import load_config
from overlap.errors import OverlapError
from overlap.exit_codes import ExitCode

app = typer.Typer(
    name="overlap",
    help=(
        "Perceptual fingerprinting and overlap detection for robotics datasets. "
        "Runs entirely offline."
    ),
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"overlap {overlap.__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
    config_file: Path | None = typer.Option(
        None, "--config", envvar="OVERLAP_CONFIG", help="Explicit config file."
    ),
    index: Path | None = typer.Option(
        None,
        "--index",
        envvar="OVERLAP_INDEX",
        help="Index directory (created on first use).",
    ),
    json_mode: bool = typer.Option(
        False, "--json", help="Machine output: NDJSON events / JSON documents on stdout."
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress; errors only."),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Increase log detail."),
) -> None:
    cfg = load_config(explicit_file=config_file)
    if index is not None:
        cfg.set("paths.index", str(index), "--index")
    ctx.obj = AppState(config=cfg, json_mode=json_mode, quiet=quiet, verbosity=verbose)


app.command("index")(index_cmd)
app.command("import")(import_cmd)
app.command("export")(export_cmd)
app.command("compare")(compare_cmd)
app.command("audit-sample")(audit_sample_cmd)
app.command("verify")(verify_cmd)
app.command("merge")(merge_cmd)
app.command("self-dedupe")(self_dedupe_cmd)
app.command("inspect")(inspect_cmd)
app.command("report")(report_cmd)
app.command("ui")(ui)
app.command("status")(status)
app.command("config")(config_cmd)
app.command("doctor")(doctor)


def main() -> None:
    """Console entry point with overlap's documented exit-code mapping."""
    try:
        app()
    except OverlapError as exc:
        typer.secho(f"error: {exc}", err=True, fg=typer.colors.RED)
        sys.exit(ExitCode.RUNTIME_ERROR)
    except KeyboardInterrupt:
        typer.secho("Interrupted - progress saved. Re-run the same command to resume.", err=True)
        sys.exit(ExitCode.INTERRUPTED)
