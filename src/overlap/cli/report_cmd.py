"""`overlap report` - re-render a saved comparison report."""

from __future__ import annotations

import json
import webbrowser
from pathlib import Path

import typer

from overlap.cli._console import get_state
from overlap.render import render_html, render_markdown


def report_cmd(
    ctx: typer.Context,
    report_json: Path = typer.Argument(..., exists=True, readable=True),
    fmt: str = typer.Option("html", "--format", help="html | md"),
    output: Path | None = typer.Option(None, "-o", "--output"),
    open_out: bool = typer.Option(False, "--open", help="Open the rendered report."),
) -> None:
    """Render a report JSON (from `compare` or `self-dedupe`) to HTML or Markdown."""
    state = get_state(ctx)
    report = json.loads(report_json.read_text(encoding="utf-8"))
    if not str(report.get("schema", "")).startswith("report/"):
        raise typer.BadParameter(f"{report_json} is not an overlap report")

    if fmt == "html":
        out_path = output or report_json.with_suffix(".html")
        out_path.write_text(render_html(report), encoding="utf-8")
    elif fmt == "md":
        out_path = output or report_json.with_suffix(".md")
        out_path.write_text(render_markdown(report), encoding="utf-8")
    else:
        raise typer.BadParameter("--format must be html or md")

    state.err.print(f"Rendered {out_path}")
    if open_out and fmt == "html":
        webbrowser.open(out_path.resolve().as_uri())
