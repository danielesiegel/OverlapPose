from __future__ import annotations

from pathlib import Path

import pytest

from overlap.config import DEFAULTS, load_config
from overlap.errors import ConfigError


def test_defaults_apply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # no project overlap.toml here
    cfg = load_config()
    assert cfg.get("index.fps") == 4.0
    assert cfg.sources["index.fps"] == "default"


def test_project_file_overrides_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "overlap.toml").write_text("[index]\nfps = 2.0\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg.get("index.fps") == 2.0
    assert cfg.sources["index.fps"].startswith("project config")


def test_env_overrides_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "overlap.toml").write_text("[index]\nfps = 2.0\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OVERLAP_INDEX_FPS", "0.5")
    cfg = load_config()
    assert cfg.get("index.fps") == 0.5
    assert cfg.sources["index.fps"] == "env (OVERLAP_INDEX_FPS)"


def test_explicit_config_file_beats_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "overlap.toml").write_text("[ui]\nport = 1111\n", encoding="utf-8")
    explicit = tmp_path / "special.toml"
    explicit.write_text("[ui]\nport = 2222\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    cfg = load_config(explicit_file=explicit)
    assert cfg.get("ui.port") == 2222


def test_unknown_key_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "overlap.toml").write_text("[index]\nbogus = 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError, match="unknown configuration key"):
        load_config()


def test_bool_coercion_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OVERLAP_UI_OPEN_BROWSER", "false")
    cfg = load_config()
    assert cfg.get("ui.open_browser") is False


def test_every_default_key_has_type() -> None:
    for key, (default, typ) in DEFAULTS.items():
        assert isinstance(typ, type), key
        if default is not None:
            assert isinstance(default, typ), key
