"""Standalone HTML report: one self-contained file, no external assets.

The output is the artifact a lab forwards to procurement or back to the
vendor, so it must open anywhere - all CSS is inline and the timeline
visuals are plain inline SVG.
"""

from __future__ import annotations

from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape

_env: Environment | None = None


def _environment() -> Environment:
    global _env
    if _env is None:
        _env = Environment(
            loader=PackageLoader("overlap.render", "templates"),
            autoescape=select_autoescape(["html"]),
        )
        _env.filters["tier_color"] = tier_color
    return _env


def tier_color(tier: str) -> str:
    return {
        "exact": "#7c3aed",
        "strong": "#dc2626",
        "probable": "#ea580c",
        "weak": "#9ca3af",
    }.get(tier, "#9ca3af")


def render_html(report: dict[str, Any]) -> str:
    template = _environment().get_template("report.html.j2")
    return template.render(report=report)
