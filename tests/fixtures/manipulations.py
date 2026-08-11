"""The manipulation library: every vendor trick as an executable transform.

Each function maps to a family in the detection matrix
(tests/detection_matrix.toml). They intentionally use the same tools a
manipulating vendor would (plain ffmpeg filters).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.fixtures.ffmpeg_factory import run_ffmpeg

if TYPE_CHECKING:
    from pathlib import Path


def reencode(src: Path, dst: Path, *, codec: str = "libx265", crf: int = 30) -> Path:
    run_ffmpeg(["-i", str(src), "-c:v", codec, "-crf", str(crf), "-pix_fmt", "yuv420p", str(dst)])
    return dst


def container_swap(src: Path, dst: Path) -> Path:
    """Repackage without re-encoding (mp4 -> mkv etc.); bytes differ, frames identical."""
    run_ffmpeg(["-i", str(src), "-c", "copy", str(dst)])
    return dst


def trim(src: Path, dst: Path, *, start: float, duration: float) -> Path:
    # -ss after -i: frame-accurate decode-then-cut (slow but exact for fixtures)
    run_ffmpeg(
        [
            "-i",
            str(src),
            "-ss",
            str(start),
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-crf",
            "20",
            str(dst),
        ]
    )
    return dst


def splice(src_a: Path, src_b: Path, dst: Path, *, a_seconds: float) -> Path:
    """First ``a_seconds`` of A followed by all of B, one continuous file."""
    run_ffmpeg(
        [
            "-i",
            str(src_a),
            "-i",
            str(src_b),
            "-filter_complex",
            f"[0:v]trim=0:{a_seconds},setpts=PTS-STARTPTS[a];"
            f"[1:v]setpts=PTS-STARTPTS[b];[a][b]concat=n=2:v=1[out]",
            "-map",
            "[out]",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            str(dst),
        ]
    )
    return dst


def merge_segments(src: Path, dst: Path, *, segments: list[tuple[float, float]]) -> Path:
    """Concatenate several (start, duration) pieces of ONE source into one file.

    This is the "split a master and sell it back as a new session" trick: the
    offer is entirely owned footage, just re-assembled out of order.
    """
    parts = []
    filters = []
    for i, (start, dur) in enumerate(segments):
        filters.append(f"[0:v]trim={start}:{start + dur},setpts=PTS-STARTPTS[p{i}]")
        parts.append(f"[p{i}]")
    graph = ";".join(filters) + ";" + "".join(parts) + f"concat=n={len(segments)}:v=1[out]"
    run_ffmpeg(
        [
            "-i",
            str(src),
            "-filter_complex",
            graph,
            "-map",
            "[out]",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            str(dst),
        ]
    )
    return dst


def speed(src: Path, dst: Path, *, factor: float) -> Path:
    """factor < 1 slows the clip down (the billable-hours inflation trick)."""
    run_ffmpeg(
        [
            "-i",
            str(src),
            "-vf",
            f"setpts={1.0 / factor}*PTS",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            str(dst),
        ]
    )
    return dst


def hflip(src: Path, dst: Path) -> Path:
    run_ffmpeg(["-i", str(src), "-vf", "hflip", "-c:v", "libx264", "-crf", "20", str(dst)])
    return dst


def crop(src: Path, dst: Path, *, keep: float) -> Path:
    """Center-crop to ``keep`` fraction of each dimension, scaled back up."""
    run_ffmpeg(
        [
            "-i",
            str(src),
            "-vf",
            f"crop=iw*{keep}:ih*{keep},scale=320:180",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            str(dst),
        ]
    )
    return dst


def crop_edge(src: Path, dst: Path, *, side: str, frac: float) -> Path:
    """Remove a strip from one edge only, then scale back to the original size.

    This is how an overlay/watermark strip gets trimmed away in practice.
    """
    if side in ("bottom", "top"):
        expr = f"crop=iw:ih*{1 - frac}:0:{'0' if side == 'bottom' else f'ih*{frac}'}"
    else:
        expr = f"crop=iw*{1 - frac}:ih:{'0' if side == 'right' else f'iw*{frac}'}:0"
    run_ffmpeg(
        [
            "-i",
            str(src),
            "-vf",
            f"{expr},scale=320:180",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            str(dst),
        ]
    )
    return dst


def letterbox(src: Path, dst: Path, *, bar_frac: float = 0.15) -> Path:
    run_ffmpeg(
        [
            "-i",
            str(src),
            "-vf",
            (f"scale=320:ih*(1-{2 * bar_frac}),pad=320:180:0:(oh-ih)/2:black"),
            "-c:v",
            "libx264",
            "-crf",
            "20",
            str(dst),
        ]
    )
    return dst


def watermark(src: Path, dst: Path, *, text: str = "SAMPLE") -> Path:
    run_ffmpeg(
        [
            "-i",
            str(src),
            "-vf",
            (
                f"drawtext=text={text}:fontcolor=white@0.6:fontsize=20:"
                f"x=10:y=10:box=1:boxcolor=black@0.4"
            ),
            "-c:v",
            "libx264",
            "-crf",
            "20",
            str(dst),
        ]
    )
    return dst


def colorgrade(src: Path, dst: Path) -> Path:
    run_ffmpeg(
        [
            "-i",
            str(src),
            "-vf",
            "eq=saturation=1.4:gamma=1.15:brightness=0.06:contrast=1.1",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            str(dst),
        ]
    )
    return dst


def fps_resample(src: Path, dst: Path, *, fps: int = 15) -> Path:
    run_ffmpeg(["-i", str(src), "-vf", f"fps={fps}", "-c:v", "libx264", "-crf", "20", str(dst)])
    return dst
