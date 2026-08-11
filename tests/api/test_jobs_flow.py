"""End-to-end web-UI job flow: index -> export -> compare -> report."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from overlap.config import load_config
from overlap.server import create_app

pytestmark = pytest.mark.integration

TOKEN = "flow-token"


@pytest.fixture()
def app(isolated_env, monkeypatch):  # type: ignore[no-untyped-def]
    import overlap.paths as paths_mod
    import overlap.server.jobs as jobs_mod

    state = isolated_env / "state"
    monkeypatch.setattr(paths_mod, "state_dir", lambda: state)
    monkeypatch.setattr(paths_mod, "reports_dir", lambda: state / "reports")
    monkeypatch.setattr(jobs_mod, "reports_dir", lambda: state / "reports")
    return create_app(load_config(), token=TOKEN)


def call(app: Any, method: str, url: str, **kw: Any) -> httpx.Response:
    async def go() -> httpx.Response:
        headers = kw.pop("headers", {})
        headers["Authorization"] = f"Bearer {TOKEN}"
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://ui", timeout=30
        ) as client:
            return await client.request(method, url, headers=headers, **kw)

    return asyncio.run(go())


def wait_for_job(app: Any, job_id: str, timeout: float = 120.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        doc = call(app, "GET", f"/api/v1/jobs/{job_id}").json()
        if doc["status"] != "running":
            return doc
        time.sleep(0.3)
    raise TimeoutError(f"job {job_id} did not finish")


def test_full_ui_flow(app: Any, base_clips, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    # 1. index the clips directory through the API
    r = call(app, "POST", "/api/v1/jobs/index", json={"paths": [str(base_clips["dir"])]})
    assert r.status_code == 202
    job = wait_for_job(app, r.json()["job_id"])
    assert job["status"] == "done", job
    assert job["result"]["indexed"] == 2

    stats = call(app, "GET", "/api/v1/stats").json()
    assert stats["files_done"] == 2

    # 2. export a manifest of the corpus
    r = call(app, "POST", "/api/v1/export", json={"label": "ui flow"})
    assert r.status_code == 200, r.text
    manifest_meta = r.json()
    assert manifest_meta["files"] == 2
    manifest_bytes = call(app, "GET", manifest_meta["download"]).content
    assert manifest_bytes[:4] == b"OVLM"

    # 3. compare that manifest against the same corpus (self-comparison via
    #    upload: everything should be an exact sha256 match)
    r = call(app, "POST", "/api/v1/compare", files={"manifest": ("corpus.olm", manifest_bytes)})
    assert r.status_code == 202
    job = wait_for_job(app, r.json()["job_id"])
    assert job["status"] == "done", job
    report_id = job["result"]["report_id"]
    assert job["result"]["summary"]["overlap_pct"] == 100.0

    # 4. report is persisted, listable, renderable
    listing = call(app, "GET", "/api/v1/reports").json()
    assert any(r["report_id"] == report_id for r in listing["reports"])
    html = call(app, "GET", f"/api/v1/reports/{report_id}/render?format=html").text
    assert "of offered footage matches" in html
    page = call(app, "GET", f"/reports/{report_id}").text
    assert "overlap" in page

    # 5. SSE stream replays events for the finished job
    events = call(app, "GET", f"/api/v1/jobs/{job['job_id']}/events").text
    assert "job_end" in events
