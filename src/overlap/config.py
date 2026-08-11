"""Configuration loading with source-tracked precedence.

Precedence (highest wins):

1. CLI flag
2. ``OVERLAP_*`` environment variable
3. ``./overlap.toml`` in the working directory
4. user config file (see :func:`overlap.paths.user_config_file`)
5. built-in default

``overlap config`` prints every effective value together with the layer it
came from; that visibility is the debuggability contract of this module.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from overlap.errors import ConfigError
from overlap.paths import default_index_dir, user_config_file

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

# key -> (default, type). Keys are "section.name" and map to [section] name in TOML
# and to OVERLAP_SECTION_NAME environment variables.
DEFAULTS: dict[str, tuple[Any, type]] = {
    "paths.index": (None, str),  # None -> overlap.paths.default_index_dir()
    # Corpus-side sampling density. 4 fps keeps temporal phase misalignment
    # under 125 ms so speed-changed and arbitrarily-trimmed copies still land
    # inside the hash radius (measured; see docs/architecture.md). Manifests
    # are exported strided back down to ~1 fps, so only the local index pays.
    "index.fps": (4.0, float),
    # Corpus-side crop geometries, defaulting to the "mild" preset: one centred
    # rung and one bottom rung. Calibrated to what a reseller can crop and still
    # sell the footage, which for egocentric hand data is not much - crop deep
    # and the hands leave the frame or the pixels turn to mush.
    #
    # Measured on 61 real frames (docs/architecture.md). The plain hash alone
    # covers 5% centred and only 3% one-sided; one rung each way reaches 11%
    # centred and ~9% one-sided, and end to end that is the difference between
    # catching an 8% bottom strip and missing it. Going deeper costs
    # proportionally more for manipulations that destroy the product's value:
    # the five-rung ladder reaches 33% centred, and measured against a bottom
    # strip it catches nothing a single bottom rung does not.
    #
    # Empty string disables crops entirely (index shrinks 3x, and a 4% bottom
    # bar becomes invisible). Use --preset balanced or thorough for adversarial
    # depth on a small, high-value corpus.
    "index.crop_ladder": ("0.94", str),
    # One-sided crops need their own geometry: removing a bottom strip shifts
    # content as well as changing the field of view, so centred rungs cannot
    # recover it at any depth. "bottom,top" applies the 6%-to-30% ladder to both
    # sides; "bottom:0.08,0.16" names depths explicitly.
    "index.crop_edges": ("bottom:0.06", str),
    "index.workers": (0, int),  # 0 = auto (cpu_count - 2, min 1)
    "index.follow_symlinks": (False, bool),
    "compare.min_run_s": (10.0, float),
    "compare.tier": ("probable", str),
    "compare.nprobe": (64, int),
    # Search threads. 0 lets the math library use every core, which is wrong on
    # a shared storage server or a workstation someone is also using: a
    # comparison is memory-bandwidth heavy and starves everything else.
    "compare.threads": (0, int),
    # Every Nth offered frame is swept against the index to find which corpus
    # streams to examine; each nominated pair is then compared exactly, so this
    # trades sweep cost against the chance of overlooking a borderline-short
    # match. 1 sweeps densely.
    "compare.probe_stride": (4, int),
    # Codes per search shard, which is what bounds memory during a comparison:
    # peak use is roughly one shard (32 bytes/code) plus the query set, whatever
    # the corpus size. Raise it on a big machine to search fewer, larger shards;
    # lower it to index a multi-TB corpus on a small one.
    "index.shard_codes": (32_000_000, int),
    "ui.host": ("127.0.0.1", str),
    "ui.port": (8377, int),
    "ui.open_browser": (True, bool),
    "ui.token_auth": (True, bool),
    "ui.max_upload_mb": (512, int),
}


@dataclass
class Config:
    """Effective configuration with per-key source tracking."""

    values: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)

    def get(self, key: str) -> Any:
        return self.values[key]

    def set(self, key: str, value: Any, source: str) -> None:
        if key not in DEFAULTS:
            raise ConfigError(f"unknown configuration key: {key!r}")
        expected = DEFAULTS[key][1]
        if value is not None and not isinstance(value, expected):
            try:
                if expected is bool and isinstance(value, str):
                    value = value.strip().lower() in ("1", "true", "yes", "on")
                else:
                    value = expected(value)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"invalid value for {key}: {value!r}") from exc
        self.values[key] = value
        self.sources[key] = source

    @property
    def index_dir(self) -> Path:
        configured = self.values.get("paths.index")
        return Path(configured) if configured else default_index_dir()


def _apply_toml(cfg: Config, path: Path, source: str) -> None:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc
    for section, entries in data.items():
        if not isinstance(entries, dict):
            raise ConfigError(f"{path}: top-level keys must be [sections], got {section!r}")
        for name, value in entries.items():
            cfg.set(f"{section}.{name}", value, source)


def _env_var_name(key: str) -> str:
    return "OVERLAP_" + key.replace(".", "_").upper()


def load_config(explicit_file: Path | None = None, cwd: Path | None = None) -> Config:
    """Resolve configuration from defaults, config files, and environment."""
    cfg = Config()
    for key, (default, _) in DEFAULTS.items():
        cfg.set(key, default, "default")

    user_file = user_config_file()
    if user_file.is_file():
        _apply_toml(cfg, user_file, f"user config ({user_file})")

    project_file = (cwd or Path.cwd()) / "overlap.toml"
    if project_file.is_file():
        _apply_toml(cfg, project_file, f"project config ({project_file})")

    if explicit_file is not None:
        if not explicit_file.is_file():
            raise ConfigError(f"config file not found: {explicit_file}")
        _apply_toml(cfg, explicit_file, f"--config ({explicit_file})")

    for key in DEFAULTS:
        env_name = _env_var_name(key)
        if env_name in os.environ:
            cfg.set(key, os.environ[env_name], f"env ({env_name})")

    # OVERLAP_INDEX is the documented shorthand for paths.index and must work
    # for every entry point (CLI, server, library), not just the CLI option.
    if "OVERLAP_INDEX" in os.environ:
        cfg.set("paths.index", os.environ["OVERLAP_INDEX"], "env (OVERLAP_INDEX)")

    return cfg
