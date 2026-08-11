"""The local web UI: a FastAPI app served on loopback by `overlap ui`.

Design rules:

- The CLI is the product; this UI is a viewport onto the same core calls and
  the same event stream. Nothing is UI-only.
- Fully offline: all assets (htmx included) are vendored into the wheel; the
  strict absence of external requests is part of the trust posture.
- Server-rendered Jinja2 + htmx, no Node toolchain: contributors need only
  Python to change the UI.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from overlap.config import Config
from overlap.server.auth import TokenAuthMiddleware
from overlap.server.jobs import JobManager

_HERE = Path(__file__).parent


def create_app(config: Config, *, token: str | None) -> FastAPI:
    from overlap.server.routes.api import router as api_router
    from overlap.server.routes.pages import router as pages_router

    app = FastAPI(title="overlap", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.config = config
    app.state.jobs = JobManager(config.index_dir)
    app.state.token = token

    app.add_middleware(TokenAuthMiddleware, token=token)
    app.mount("/static", StaticFiles(directory=_HERE / "static"), name="static")
    app.include_router(api_router, prefix="/api/v1")
    app.include_router(pages_router)
    return app
