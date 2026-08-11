"""`overlap ui` - serve the local web UI."""

from __future__ import annotations

import secrets
import threading
import webbrowser

import typer

from overlap.cli._console import get_state


def ui(
    ctx: typer.Context,
    host: str | None = typer.Option(None, "--host", help="Bind address (default 127.0.0.1)."),
    port: int | None = typer.Option(None, "--port", help="Port (default 8377)."),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="Don't open a browser (SSH / headless use)."
    ),
    no_token: bool = typer.Option(
        False, "--no-token", help="Disable token auth (single-user machines only)."
    ),
) -> None:
    """Start the web UI on localhost.

    Data never leaves this machine: the server binds to loopback by default
    and makes no outbound connections. On a remote data server, tunnel it:
    ssh -L 8377:127.0.0.1:8377 user@server
    """
    import uvicorn

    from overlap.server import create_app

    state = get_state(ctx)
    cfg = state.config
    bind_host = host or str(cfg.get("ui.host"))
    bind_port = port if port is not None else int(cfg.get("ui.port"))
    token_enabled = not no_token and bool(cfg.get("ui.token_auth"))
    token = secrets.token_hex(16) if token_enabled else None

    if bind_host not in ("127.0.0.1", "localhost", "::1"):
        state.err.print(
            "[yellow]Warning: binding to a non-loopback address exposes your "
            "filesystem index to the network. Prefer SSH port-forwarding.[/yellow]"
        )

    app = create_app(cfg, token=token)
    url = f"http://{bind_host}:{bind_port}/" + (f"?token={token}" if token else "")
    state.err.print(f"overlap ui running at:  [bold]{url}[/bold]")
    state.err.print(
        f"(data stays local; over SSH use: ssh -L {bind_port}:127.0.0.1:{bind_port} user@server)"
    )

    if not no_browser and bool(cfg.get("ui.open_browser")):
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()

    uvicorn.run(app, host=bind_host, port=bind_port, log_level="warning")
