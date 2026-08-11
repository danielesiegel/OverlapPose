"""Shared helpers for the benchmark scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXT = (".mp4", ".mkv", ".avi", ".mov", ".webm")


def parse_args(description: str, **extra: dict) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--clips", type=Path, required=True, help="Directory of video files.")
    ap.add_argument("--limit", type=int, default=8, help="How many clips to use.")
    ap.add_argument("--per-clip", type=int, default=6, help="Frames sampled per clip.")
    for flag, kwargs in extra.items():
        ap.add_argument(f"--{flag}", **kwargs)
    return ap.parse_args()


def clips(directory: Path, limit: int) -> list[Path]:
    found = sorted(p for p in directory.rglob("*") if p.suffix.lower() in VIDEO_EXT)
    if not found:
        raise SystemExit(f"no video files under {directory}")
    return found[:limit]


def gray_frames(paths: list[Path], per_clip: int) -> list[np.ndarray]:
    """Evenly spaced grayscale frames, decoded the way the indexer decodes them."""
    import av

    out: list[np.ndarray] = []
    for path in paths:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            step = max(1, (stream.frames or 5400) // per_clip)
            for i, frame in enumerate(container.decode(stream)):
                if i % step == 0:
                    out.append(
                        cv2.cvtColor(frame.to_ndarray(format="bgr24"), cv2.COLOR_BGR2GRAY)
                    )
                if len(out) >= per_clip * len(paths):
                    break
    return out


def centre_crop(gray: np.ndarray, keep: float) -> np.ndarray:
    h, w = gray.shape
    dh, dw = int(round(h * (1 - keep) / 2)), int(round(w * (1 - keep) / 2))
    return gray[dh : h - dh or None, dw : w - dw or None]


def edge_crop(gray: np.ndarray, frac: float, side: str = "bottom") -> np.ndarray:
    h = gray.shape[0]
    cut = int(round(h * frac))
    return gray[:-cut] if side == "bottom" else gray[cut:]
