"""Default filesystem locations, resolved via platformdirs.

Single source of truth: every module that needs a default path imports from
here, so `overlap config` can report exactly where things live.
"""

from __future__ import annotations

from pathlib import Path

from platformdirs import user_config_dir, user_data_dir, user_state_dir

_APPNAME = "overlap"


def default_index_dir() -> Path:
    """Default index directory (an `.ovl` directory, created on first use)."""
    return Path(user_data_dir(_APPNAME, appauthor=False)) / "corpus.ovl"


def user_config_file() -> Path:
    return Path(user_config_dir(_APPNAME, appauthor=False)) / "config.toml"


def state_dir() -> Path:
    """Saved reports and job journal."""
    return Path(user_state_dir(_APPNAME, appauthor=False))


def reports_dir() -> Path:
    return state_dir() / "reports"
