"""Hash kernel protocol - the seam behind which the perceptual hash lives.

Everything downstream (ingestion, index, manifest, matcher) sees only
:class:`HashKernel` and :class:`FrameHash`. A future native (C/Rust)
implementation drops in behind this protocol without touching callers.

Flip handling: every frame yields two digests - identity and horizontal
mirror. The **local index stores both** (the mirror digest cannot be derived
from the identity digest later: each variant is re-thresholded at its own
median). **Manifests carry only the identity digest** to stay compact; a
flipped copy is still caught because its identity hash matches the original's
mirror entry in the index. Which variant a hit lands on is what tells the
matcher a match is mirrored - orientation is a match-time property, not a
stored frame property.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

HASH_BYTES = 32  # 256-bit hashes throughout

# Per-frame flag bits (stored in the index and in manifests)
FLAG_LOW_QUALITY = 0x01  # excluded from ANN candidate generation


@dataclass(frozen=True)
class FrameHash:
    """Result of hashing one frame: both orientation variants plus quality."""

    hash: bytes  # HASH_BYTES identity digest
    mirror: bytes  # HASH_BYTES digest of the horizontally-flipped frame
    quality: int  # 0..100 (image-domain quality metric)
    flags: int


class HashKernel(Protocol):
    """Stable interface for perceptual frame hashing."""

    # Identifies the algorithm + parameters; part of fingerprint identity.
    # Streams hashed with different algo_ids never compare.
    algo_id: str
    hash_bits: int

    def hash_frame(self, gray: np.ndarray) -> FrameHash:
        """Hash one grayscale frame (HxW uint8, already prep-normalized)."""
        ...
