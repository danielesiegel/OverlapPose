"""Token authentication for the local web UI.

Lab servers are often multi-user machines and the API can read any path the
operating user can, so authentication defaults to ON: a per-session token is
printed in the launch URL (Jupyter-style ``?token=...``), exchanged for a
cookie on first visit. ``overlap ui --no-token`` disables it for single-user
boxes.
"""

from __future__ import annotations

import hmac

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response

COOKIE_NAME = "overlap_session"
_EXEMPT = ("/api/v1/health",)


class TokenAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, token: str | None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(app)
        self.token = token

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        if self.token is None or request.url.path in _EXEMPT:
            return await call_next(request)

        query_token = request.query_params.get("token")
        if query_token is not None and hmac.compare_digest(query_token, self.token):
            # Exchange the URL token for a cookie so links stay clean.
            response: Response
            if request.method == "GET" and not request.url.path.startswith("/api/"):
                clean = request.url.remove_query_params("token")
                response = RedirectResponse(str(clean), status_code=303)
            else:
                response = await call_next(request)
            response.set_cookie(COOKIE_NAME, self.token, httponly=True, samesite="strict")
            return response

        cookie = request.cookies.get(COOKIE_NAME, "")
        if hmac.compare_digest(cookie, self.token):
            return await call_next(request)

        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer ") and hmac.compare_digest(auth_header[7:], self.token):
            return await call_next(request)

        return JSONResponse(
            {
                "detail": "authentication required: open the URL printed by "
                "`overlap ui` (it contains ?token=...) or send it as a Bearer token"
            },
            status_code=401,
        )
