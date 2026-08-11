"""Report renderers. All consume the report/1 dict from overlap.match.compare."""

from overlap.render.html import render_html
from overlap.render.markdown import render_markdown

__all__ = ["render_html", "render_markdown"]
