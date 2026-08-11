"""HTML pages (server-rendered Jinja2 + htmx)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from overlap.render import render_html
from overlap.server.jobs import list_reports, load_report

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))


def _stats(request: Request) -> dict[str, Any]:
    from overlap.server.routes.api import stats

    return stats(request)


@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> Any:
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "stats": _stats(request),
            "reports": list_reports()[:8],
            "jobs": [j.describe() for j in request.app.state.jobs.list()[:8]],
        },
    )


@router.get("/index", response_class=HTMLResponse)
def index_page(request: Request) -> Any:
    return templates.TemplateResponse(request, "index.html", {"stats": _stats(request)})


@router.get("/compare", response_class=HTMLResponse)
def compare_page(request: Request) -> Any:
    return templates.TemplateResponse(request, "compare.html", {"stats": _stats(request)})


@router.get("/export", response_class=HTMLResponse)
def export_page(request: Request) -> Any:
    return templates.TemplateResponse(request, "export.html", {"stats": _stats(request)})


@router.get("/reports", response_class=HTMLResponse)
def reports_page(request: Request) -> Any:
    return templates.TemplateResponse(request, "reports.html", {"reports": list_reports()})


@router.get("/reports/{report_id}", response_class=HTMLResponse)
def report_view(report_id: str) -> Any:
    """The full report is its own self-contained page (same HTML a lab
    would email to procurement)."""
    report = load_report(report_id)
    if report is None:
        raise HTTPException(404, "no such report")
    return HTMLResponse(render_html(report))
