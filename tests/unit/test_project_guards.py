"""Project-level guards that run with the normal test suite.

These enforce the two promises the project makes about itself - no network
I/O, and documentation that matches the executable detection matrix - so
they hold on any contributor's machine rather than depending on hosted CI.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).parent.parent.parent / "src" / "overlap"
REPO = Path(__file__).parent.parent.parent

# Outbound HTTP/socket clients. uvicorn/fastapi are allowed: they serve
# loopback, they don't make outbound connections.
BANNED_IMPORTS = {
    "requests",
    "httpx",
    "aiohttp",
    "urllib3",
    "urllib.request",
    "websockets",
    "socketio",
    "telemetry",
    "ftplib",
    "smtplib",
    "xmlrpc",
}


def iter_source_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def test_package_has_no_network_client_imports() -> None:
    """overlap must never phone home. This is a promise in SECURITY.md."""
    offenders: list[str] = []
    for path in iter_source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                root = name.split(".")[0]
                if name in BANNED_IMPORTS or root in BANNED_IMPORTS:
                    offenders.append(f"{path.relative_to(REPO)}:{node.lineno} imports {name}")
    assert not offenders, "overlap must not make outbound network calls; found:\n  " + "\n  ".join(
        offenders
    )


def test_package_never_opens_raw_sockets() -> None:
    offenders = [
        f"{path.relative_to(REPO)}"
        for path in iter_source_files()
        if "socket.socket(" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"raw socket use found in: {offenders}"


def test_detection_matrix_docs_are_in_sync() -> None:
    """README/docs tables are generated from tests/detection_matrix.toml."""
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "gen_detection_matrix_doc.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert result.returncode == 0, (
        "detection matrix documentation is stale - run:\n"
        "  python scripts/gen_detection_matrix_doc.py\n" + result.stderr
    )


def test_no_committed_media_fixtures() -> None:
    """Fixtures are generated, never committed (keeps the repo small)."""
    media = [
        p
        for ext in ("*.mp4", "*.mkv", "*.mov", "*.avi", "*.webm", "*.mcap", "*.bag")
        for p in (REPO / "tests").rglob(ext)
    ]
    tracked = []
    for path in media:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)],
            capture_output=True,
            cwd=REPO,
        )
        if result.returncode == 0:
            tracked.append(str(path.relative_to(REPO)))
    assert not tracked, f"media fixtures must not be committed: {tracked}"


@pytest.mark.parametrize(
    "banned",
    ["detects all", "guarantees", "tamper-proof", "fraud-proof", "impossible to evade"],
)
def test_docs_avoid_overclaiming_language(banned: str) -> None:
    """Claim hygiene (docs/claim-hygiene.md) enforced mechanically."""
    docs = [REPO / "README.md", *(REPO / "docs").glob("*.md")]
    offenders = []
    for path in docs:
        if path.name == "claim-hygiene.md":  # it quotes the banned list
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if banned in line.lower():
                offenders.append(f"{path.relative_to(REPO)}:{lineno}")
    assert not offenders, f"banned phrase {banned!r} found at: {offenders}"
