"""Shared test configuration.

Integration tests (marked ``integration``) need the ffmpeg CLI on PATH to
generate synthetic fixture clips. They are auto-skipped when it is missing,
with a clear message. CI always installs ffmpeg and separately asserts that
integration tests were collected, so CI can never silently skip them.
"""

from __future__ import annotations

import shutil

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if shutil.which("ffmpeg") is not None:
        return
    skip = pytest.mark.skip(reason="ffmpeg CLI not on PATH (needed to generate fixture media)")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture()
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path):  # type: ignore[no-untyped-def]
    """Point every default path and env override at a temp directory."""
    for var in list((__import__("os")).environ):
        if var.startswith("OVERLAP_"):
            monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OVERLAP_INDEX", str(tmp_path / "corpus.ovl"))
    return tmp_path


# Base clips are 40 s so that trims/splices leave runs comfortably above the
# matcher's evidence thresholds (min-run 10 s, >= 8 inliers at 1 fps).
CLIP_SECONDS = 40.0


@pytest.fixture(scope="session")
def base_clips(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Two deterministic clips, generated once per test session."""
    from tests.fixtures.ffmpeg_factory import make_teleop_clip

    d = tmp_path_factory.mktemp("clips")
    return {
        "a": make_teleop_clip(d / "clip_a.mp4", duration=CLIP_SECONDS, seed=11),
        "b": make_teleop_clip(d / "clip_b.mp4", duration=CLIP_SECONDS, seed=23),
        "dir": d,
        "seconds": CLIP_SECONDS,
    }
