"""Synthetic fixture clips, generated with ffmpeg lavfi sources.

No media files are committed to the repository - every clip is generated
deterministically at test time (seconds of work, a few hundred KB each).
Clips need temporal structure (visibly different content each second) so the
temporal matcher has something to chain; ``testsrc2`` provides that, and the
"scene" variants add distinct phases so trims and splices land in
recognizable regions.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

SIZE = "320x180"
FPS = 24


def run_ffmpeg(args: list[str]) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{proc.stderr}")


def make_clip(dst: Path, *, duration: float = 8.0, variant: int = 0) -> Path:
    """A deterministic clip. ``variant`` shifts content so clips differ."""
    src = (
        f"testsrc2=size={SIZE}:rate={FPS}:duration={duration}"
        if variant == 0
        else f"mandelbrot=size={SIZE}:rate={FPS}:start_scale={2 + variant}"
    )
    args = ["-f", "lavfi", "-i", src]
    if variant != 0:
        args += ["-t", str(duration)]
    args += ["-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "20", str(dst)]
    run_ffmpeg(args)
    return dst


def make_teleop_clip(
    dst: Path, *, duration: float = 40.0, seed: int = 11, pan_start: int = 100
) -> Path:
    """A pseudo-teleop clip: slow camera pan over a textured scene plus a
    locally moving object - realistic motion statistics (unlike testsrc2,
    whose per-frame counters change faster than any real camera footage).

    Deterministic per seed: the background is seeded numpy noise rendered to
    a temporary PNG; the pan path and object trajectory are fixed functions.
    ``pan_start`` moves the camera path to a different region of the same
    scene - same room and rig, genuinely different footage, which is the
    false-positive control that matters most.
    """
    import cv2
    import numpy as np

    rng = np.random.default_rng(seed)
    small = rng.normal(128.0, 55.0, size=(500, 1200)).astype(np.float32)
    small = cv2.GaussianBlur(small, (0, 0), 4)
    small = cv2.normalize(small, None, 0, 255, cv2.NORM_MINMAX)
    background = cv2.resize(small, (2400, 1000), interpolation=cv2.INTER_CUBIC)
    bg_png = dst.with_suffix(".bg.png")
    cv2.imwrite(str(bg_png), background.astype(np.uint8))

    frames = int(duration * FPS)
    phase = seed % 7
    try:
        run_ffmpeg(
            [
                "-loop",
                "1",
                "-i",
                str(bg_png),
                "-f",
                "lavfi",
                "-i",
                f"color=white:s=40x40:r={FPS}",
                "-filter_complex",
                f"[0:v]crop=320:180:x='min(2080,{pan_start}+n*0.5)':y='min(820,50+n*0.2)'[bgv];"
                f"[bgv][1:v]overlay=x='mod(n*3+{phase * 31},280)'"
                f":y='40+20*sin(n/10+{phase})':shortest=0[out]",
                "-map",
                "[out]",
                "-frames:v",
                str(frames),
                "-r",
                str(FPS),
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "libx264",
                "-crf",
                "20",
                str(dst),
            ]
        )
    finally:
        bg_png.unlink(missing_ok=True)
    return dst
