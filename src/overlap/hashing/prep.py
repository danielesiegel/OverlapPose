"""prep-v1 - frame normalization applied before hashing.

The prep pipeline version (``PREP_ID``) is part of fingerprint identity:
streams prepared with different versions never compare, because a silent prep
change would silently change every hash.

Steps:

1. grayscale (BT.601 luma - matches what OpenCV and video pipelines produce)
2. letterbox/pillarbox strip: detected once per stream on a probe set of
   frames, then applied as a *fixed* crop to every frame (per-frame detection
   would jitter and destabilize hashes)

Aspect-ratio changes need no handling here: PDQ squashes to a square, so
anamorphic stretch hashes identically by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PREP_ID = "p1"

# A row/col is "border" when its luma spread is below this (near-constant line).
_FLATNESS_SPREAD = 8
# Never crop more than this fraction per side - beyond it, "border" detection
# is more likely eating real (dark, static) content than a matte.
_MAX_CROP_FRAC = 0.40
# Number of probe frames the detector wants (callers sample up to this many).
BORDER_PROBE_FRAMES = 30


@dataclass(frozen=True)
class BorderCrop:
    """Fixed per-stream crop in pixels: top, bottom, left, right."""

    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0

    def is_noop(self) -> bool:
        return self.top == self.bottom == self.left == self.right == 0

    def as_str(self) -> str:
        return f"{self.top},{self.bottom},{self.left},{self.right}"

    @classmethod
    def from_str(cls, s: str) -> BorderCrop:
        t, b, left, r = (int(x) for x in s.split(","))
        return cls(t, b, left, r)


def to_gray(frame: np.ndarray) -> np.ndarray:
    """BT.601 luma as uint8. Accepts HxW gray or HxWx3 BGR."""
    if frame.ndim == 2:
        return frame if frame.dtype == np.uint8 else _to_u8(frame)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(_to_u8(frame), cv2.COLOR_BGR2GRAY)
    raise ValueError(f"unsupported frame shape {frame.shape}")


def _to_u8(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.uint8:
        return a
    if a.dtype == np.uint16:
        return (a >> 8).astype(np.uint8)
    clipped: np.ndarray = np.clip(a, 0, 255).astype(np.uint8)
    return clipped


def _flat_run(spreads: np.ndarray, limit: int) -> int:
    """Count consecutive near-constant lines from the start, up to limit."""
    n = 0
    for value in spreads[:limit]:
        if value >= _FLATNESS_SPREAD:
            break
        n += 1
    return n


def _frame_borders(gray: np.ndarray) -> tuple[int, int, int, int]:
    h, w = gray.shape
    row_spread = np.percentile(gray, 95, axis=1) - np.percentile(gray, 5, axis=1)
    col_spread = np.percentile(gray, 95, axis=0) - np.percentile(gray, 5, axis=0)
    max_v = int(h * _MAX_CROP_FRAC)
    max_h = int(w * _MAX_CROP_FRAC)
    return (
        _flat_run(row_spread, max_v),
        _flat_run(row_spread[::-1], max_v),
        _flat_run(col_spread, max_h),
        _flat_run(col_spread[::-1], max_h),
    )


def detect_border_crop(probe_frames: list[np.ndarray]) -> BorderCrop:
    """Detect a stable letterbox/pillarbox crop from a set of probe frames.

    Per-side median across frames: a matte is present in every frame, while a
    transiently dark scene only depresses a minority of probes.
    """
    if not probe_frames:
        return BorderCrop()
    sides = np.array([_frame_borders(to_gray(f)) for f in probe_frames])
    top, bottom, left, right = (int(np.median(sides[:, k])) for k in range(4))
    return BorderCrop(top, bottom, left, right)


def apply_crop(gray: np.ndarray, crop: BorderCrop) -> np.ndarray:
    if crop.is_noop():
        return gray
    h, w = gray.shape
    return gray[crop.top : h - crop.bottom or None, crop.left : w - crop.right or None]


def center_crop(gray: np.ndarray, keep: float) -> np.ndarray:
    """Centered zoom-crop keeping ``keep`` of each dimension (crop-ladder rungs).

    No rescale is needed before hashing: the hash pipeline squashes every
    input to a fixed square anyway, so cropping alone reproduces what a
    crop-and-upscale manipulation does to the signal.
    """
    if keep >= 1.0:
        return gray
    h, w = gray.shape
    dh, dw = int(round(h * (1.0 - keep) / 2)), int(round(w * (1.0 - keep) / 2))
    return gray[dh : h - dh or None, dw : w - dw or None]


def edge_strip(gray: np.ndarray, side: str, frac: float) -> np.ndarray:
    """Remove a strip of ``frac`` from one edge (a one-sided content crop).

    Distinct from :func:`center_crop`: removing one edge changes the aspect
    ratio, and because the hash squashes its input to a square that is a
    large, geometry-specific signal change. Measured, a bottom-strip copy is
    ~112 bits from the uncropped frame and ~125-139 from any *centered* crop
    variant, so only a matching strip geometry recovers it.
    """
    if frac <= 0:
        return gray
    h, w = gray.shape
    if side in ("bottom", "top"):
        d = int(round(h * frac))
        return gray[: h - d or None, :] if side == "bottom" else gray[d:, :]
    d = int(round(w * frac))
    return gray[:, : w - d or None] if side == "right" else gray[:, d:]


CROP_SIDES = ("center", "bottom", "top", "left", "right")
# Rungs 6% apart: measured worst-case mid-rung mismatch is 31-45 of 256 bits,
# inside the candidate radius, while an 8% step would leave gaps.
DEFAULT_EDGE_LADDER = (0.06, 0.12, 0.18, 0.24, 0.30)


@dataclass(frozen=True)
class CropVariant:
    """One indexed crop geometry. ``frac`` is the fraction removed."""

    side: str  # "center" (symmetric zoom) or an edge name
    frac: float

    @property
    def pct(self) -> float:
        return round(100.0 * self.frac, 1)

    def label(self) -> str:
        """Human label. Tilde because this is the nearest indexed rung, not a
        measurement of the copy: a 15% crop matches the 18% rung."""
        return "uncropped" if self.frac <= 0 else f"{self.side} ~{self.pct:g}%"

    def apply(self, gray: np.ndarray) -> np.ndarray:
        if self.frac <= 0:
            return gray
        if self.side == "center":
            return center_crop(gray, 1.0 - self.frac)
        return edge_strip(gray, self.side, self.frac)


def parse_crop_ladder(spec: str) -> tuple[float, ...]:
    """Parse the centered-ladder config string "0.94,0.88" -> keep fractions."""
    if not spec.strip():
        return ()
    rungs = []
    for part in spec.split(","):
        keep = float(part.strip())
        if not 0.5 <= keep < 1.0:
            raise ValueError(f"crop ladder rung {keep} outside the sane range [0.5, 1.0)")
        rungs.append(keep)
    return tuple(rungs)


def parse_crop_edges(spec: str) -> tuple[CropVariant, ...]:
    """Parse the edge-crop config string into variants.

    Accepts ``"bottom,top"`` (each side gets the default 6% ladder) or an
    explicit ``"bottom:0.08,0.16;top:0.08"``.
    """
    spec = spec.strip()
    if not spec:
        return ()
    variants: list[CropVariant] = []
    for group in spec.split(";"):
        group = group.strip()
        if not group:
            continue
        if ":" in group:
            side, fracs = group.split(":", 1)
            ladder = tuple(float(x) for x in fracs.split(",") if x.strip())
        else:
            # bare list of side names
            for side_name in group.split(","):
                side_name = side_name.strip()
                if side_name:
                    variants.extend(_edge_variants(side_name, DEFAULT_EDGE_LADDER))
            continue
        variants.extend(_edge_variants(side.strip(), ladder))
    return tuple(variants)


def _edge_variants(side: str, ladder: tuple[float, ...]) -> list[CropVariant]:
    if side not in CROP_SIDES or side == "center":
        raise ValueError(f"unknown crop edge {side!r}; expected one of bottom, top, left, right")
    out = []
    for frac in ladder:
        if not 0.0 < frac < 0.5:
            raise ValueError(f"edge crop fraction {frac} outside the sane range (0, 0.5)")
        out.append(CropVariant(side=side, frac=frac))
    return out


def build_crop_variants(center_spec: str, edges_spec: str) -> tuple[CropVariant, ...]:
    """Full ordered variant list (excluding the uncropped frame itself)."""
    centered = [
        CropVariant("center", round(1.0 - keep, 4)) for keep in parse_crop_ladder(center_spec)
    ]
    return tuple(centered) + parse_crop_edges(edges_spec)


def crop_variants_spec(variants: tuple[CropVariant, ...]) -> str:
    """Canonical string stored in index metadata so lookups can be labelled."""
    return ",".join(f"{v.side}:{v.frac:.4f}" for v in variants)


def parse_crop_variants_spec(spec: str) -> tuple[CropVariant, ...]:
    if not spec.strip():
        return ()
    out = []
    for token in spec.split(","):
        side, frac = token.split(":")
        out.append(CropVariant(side=side, frac=float(frac)))
    return tuple(out)
