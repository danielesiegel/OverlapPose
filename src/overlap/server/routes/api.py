"""/api/v1 - the JSON contract. The htmx pages, tests, and any future
frontend all consume exactly this surface."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

import overlap
from overlap.render import render_html, render_markdown
from overlap.server.jobs import delete_report, list_reports, load_report
from overlap.store.catalog import Catalog
from overlap.store.manifest import FORMAT_VERSION, SCHEMA_VERSION, SUFFIX

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/version")
def version() -> dict[str, Any]:
    return {
        "overlap": overlap.__version__,
        "manifest_format": FORMAT_VERSION,
        "manifest_schema": SCHEMA_VERSION,
    }


@router.get("/stats")
def stats(request: Request) -> dict[str, Any]:
    index_dir = request.app.state.config.index_dir
    if not (index_dir / "catalog.sqlite").exists():
        return {"index_dir": str(index_dir), "exists": False}
    with Catalog.open(index_dir) as catalog:
        s = catalog.stats()
        meta = {k: catalog.get_meta(k) for k in ("algo_id", "prep_id", "sample_fps")}
    return {
        "index_dir": str(index_dir),
        "exists": True,
        "meta": meta,
        "files_done": s.files_done,
        "files_error": s.files_error,
        "streams": s.streams,
        "frames": s.frames,
        "hours": round(s.total_duration_ms / 3.6e6, 2),
        "db_bytes": s.db_bytes,
    }


@router.get("/files")
def files(request: Request, limit: int = 200) -> dict[str, Any]:
    index_dir = request.app.state.config.index_dir
    if not (index_dir / "catalog.sqlite").exists():
        return {"files": []}
    with Catalog.open(index_dir) as catalog:
        rows = catalog.file_rows()
    for row in rows:
        row["sha256"] = row["sha256"].hex() if row["sha256"] else None
    return {"files": rows[: max(0, limit)], "total": len(rows)}


class IndexJobRequest(BaseModel):
    paths: list[str] = Field(min_length=1)
    reindex: bool = False
    workers: int = 0


@router.post("/jobs/index", status_code=202)
def start_index(request: Request, body: IndexJobRequest) -> dict[str, str]:
    cfg = request.app.state.config
    missing = [p for p in body.paths if not Path(p).exists()]
    if missing:
        raise HTTPException(400, f"paths do not exist on this machine: {missing}")
    job = request.app.state.jobs.start_index(
        body.paths,
        sample_fps=float(cfg.get("index.fps")),
        crop_ladder=str(cfg.get("index.crop_ladder")),
        crop_edges=str(cfg.get("index.crop_edges")),
        workers=body.workers,
        reindex=body.reindex,
    )
    return {"job_id": job.job_id}


@router.post("/compare", status_code=202)
async def start_compare(request: Request, manifest: UploadFile) -> dict[str, str]:
    cfg = request.app.state.config
    max_bytes = int(cfg.get("ui.max_upload_mb")) * 1_000_000
    payload = await manifest.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise HTTPException(413, f"manifest exceeds the {max_bytes // 1_000_000} MB upload cap")
    tmp = Path(tempfile.mkstemp(suffix=SUFFIX, prefix="upload-")[1])
    tmp.write_bytes(payload)
    job = request.app.state.jobs.start_compare(
        tmp,
        min_run_s=float(cfg.get("compare.min_run_s")),
        nprobe=int(cfg.get("compare.nprobe")),
        max_manifest_bytes=max_bytes,
        label=manifest.filename,
    )
    return {"job_id": job.job_id}


@router.post("/self-dedupe", status_code=202)
def start_self_dedupe(request: Request) -> dict[str, str]:
    cfg = request.app.state.config
    job = request.app.state.jobs.start_self_dedupe(min_run_s=float(cfg.get("compare.min_run_s")))
    return {"job_id": job.job_id}


@router.get("/jobs")
def jobs(request: Request) -> dict[str, Any]:
    return {"jobs": [j.describe() for j in request.app.state.jobs.list()]}


@router.get("/jobs/{job_id}")
def job_detail(request: Request, job_id: str) -> dict[str, Any]:
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    detail: dict[str, Any] = job.describe()
    return detail


@router.post("/jobs/{job_id}/cancel")
def job_cancel(request: Request, job_id: str) -> dict[str, Any]:
    ok = request.app.state.jobs.cancel(job_id)
    if not ok:
        raise HTTPException(409, "job is not running")
    return {"cancelling": True}


@router.get("/jobs/{job_id}/events")
async def job_events(request: Request, job_id: str) -> EventSourceResponse:
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")

    async def stream() -> Any:
        cursor = 0
        while True:
            if await request.is_disconnected():
                return
            while cursor < len(job.events):
                event = job.events[cursor]
                cursor += 1
                yield {"event": event.get("event", "message"), "data": json.dumps(event)}
                if event.get("event") == "job_end":
                    return
            await asyncio.sleep(0.25)

    return EventSourceResponse(stream())


class ExportRequest(BaseModel):
    label: str | None = None
    anonymize_paths: bool = False


@router.post("/export")
def export(request: Request, body: ExportRequest) -> dict[str, Any]:
    from time import strftime

    from overlap.paths import state_dir
    from overlap.store.manifest import export_manifest

    cfg = request.app.state.config
    index_dir = cfg.index_dir
    if not (index_dir / "catalog.sqlite").exists():
        raise HTTPException(409, "nothing indexed yet")
    out_dir = state_dir() / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"corpus-{strftime('%Y%m%d-%H%M%S')}{SUFFIX}"
    out = out_dir / name
    with Catalog.open(index_dir) as catalog:
        manifest = export_manifest(
            catalog,
            out,
            label=body.label,
            anonymize_paths=body.anonymize_paths,
            target_fps=1.0,
        )
    return {
        "name": name,
        "bytes": out.stat().st_size,
        "files": len(manifest.files),
        "hours": round(manifest.total_hours, 2),
        "download": f"/api/v1/manifests/{name}",
    }


@router.get("/manifests/{name}")
def download_manifest(name: str) -> Any:
    from fastapi.responses import FileResponse

    from overlap.paths import state_dir

    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "bad name")
    path = state_dir() / "manifests" / name
    if not path.is_file():
        raise HTTPException(404, "no such manifest")
    return FileResponse(path, filename=name, media_type="application/octet-stream")


@router.get("/reports")
def reports() -> dict[str, Any]:
    return {"reports": list_reports()}


@router.get("/reports/{report_id}")
def report_json(report_id: str) -> dict[str, Any]:
    report = load_report(report_id)
    if report is None:
        raise HTTPException(404, "no such report")
    return report


@router.delete("/reports/{report_id}")
def report_delete(report_id: str) -> dict[str, Any]:
    if not delete_report(report_id):
        raise HTTPException(404, "no such report")
    return {"deleted": report_id}


@router.get("/reports/{report_id}/render")
def report_render(report_id: str, format: str = "html") -> Any:
    report = load_report(report_id)
    if report is None:
        raise HTTPException(404, "no such report")
    if format == "html":
        return HTMLResponse(render_html(report))
    if format == "md":
        return PlainTextResponse(render_markdown(report), media_type="text/markdown")
    raise HTTPException(400, "format must be html or md")
