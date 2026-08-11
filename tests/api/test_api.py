"""Web API contract tests (no server process - ASGI transport)."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

import overlap
from overlap.config import load_config
from overlap.server import create_app

TOKEN = "test-token-123"


@pytest.fixture()
def app(isolated_env, monkeypatch):  # type: ignore[no-untyped-def]
    # reports/state must not touch the real user dirs
    import overlap.paths as paths_mod

    state = isolated_env / "state"
    monkeypatch.setattr(paths_mod, "state_dir", lambda: state)
    monkeypatch.setattr(paths_mod, "reports_dir", lambda: state / "reports")
    import overlap.server.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "reports_dir", lambda: state / "reports")
    return create_app(load_config(), token=TOKEN)


def request(
    app: Any, method: str, url: str, *, token: str | None = TOKEN, **kw: Any
) -> httpx.Response:
    async def go() -> httpx.Response:
        headers = kw.pop("headers", {})
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://ui") as client:
            return await client.request(method, url, headers=headers, **kw)

    return asyncio.run(go())


def test_health_is_open_without_token(app: Any) -> None:
    r = request(app, "GET", "/api/v1/health", token=None)
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_everything_else_requires_token(app: Any) -> None:
    assert request(app, "GET", "/api/v1/stats", token=None).status_code == 401
    assert request(app, "GET", "/", token=None).status_code == 401
    assert request(app, "GET", "/api/v1/stats", token="wrong").status_code == 401


def test_bearer_token_accepted(app: Any) -> None:
    r = request(app, "GET", "/api/v1/version")
    assert r.status_code == 200
    assert r.json()["overlap"] == overlap.__version__


def test_query_token_sets_cookie_and_redirects(app: Any) -> None:
    r = request(app, "GET", f"/?token={TOKEN}", token=None)
    assert r.status_code == 303
    assert "overlap_session" in r.headers.get("set-cookie", "")


def test_stats_on_empty_index(app: Any) -> None:
    r = request(app, "GET", "/api/v1/stats")
    assert r.status_code == 200
    doc = r.json()
    assert doc["exists"] is False


def test_pages_render(app: Any) -> None:
    for path in ("/", "/index", "/compare", "/reports", "/export"):
        r = request(app, "GET", path)
        assert r.status_code == 200, path
        assert "<nav>" in r.text
        assert "cdn." not in r.text  # offline: no external assets ever


def test_index_job_rejects_missing_paths(app: Any) -> None:
    r = request(app, "POST", "/api/v1/jobs/index", json={"paths": ["Z:/definitely/not/here"]})
    assert r.status_code == 400


def test_unknown_job_and_report_404(app: Any) -> None:
    assert request(app, "GET", "/api/v1/jobs/nope").status_code == 404
    assert request(app, "GET", "/api/v1/reports/nope").status_code == 404
    assert request(app, "GET", "/reports/nope").status_code == 404


def test_compare_upload_cap(app: Any, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    app.state.config.set("ui.max_upload_mb", 0, "test")
    r = request(app, "POST", "/api/v1/compare", files={"manifest": ("m.olm", b"x" * 2000)})
    assert r.status_code == 413


def test_export_on_empty_index_conflicts(app: Any) -> None:
    r = request(app, "POST", "/api/v1/export", json={})
    assert r.status_code == 409


def test_manifest_download_path_traversal_blocked(app: Any) -> None:
    r = request(app, "GET", "/api/v1/manifests/..%2Fsecrets.txt")
    assert r.status_code in (400, 404)
