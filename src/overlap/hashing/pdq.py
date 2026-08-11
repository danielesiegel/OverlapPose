"""The default hash kernel: overlap's mirror-symmetric PDQ implementation.

``algo_id = "pdq2"`` names this exact implementation
(:mod:`overlap.hashing.pdq_numpy`). It is deterministic across platforms and
Python versions; vendor and lab sides therefore produce bit-identical hashes
for identical decoded frames.
"""

from __future__ import annotations

import numpy as np

from overlap.hashing.base import FLAG_LOW_QUALITY, FrameHash
from overlap.hashing.pdq_numpy import compute_pdq_dihedral, pack_bits

# Frames below this quality are hashed and stored but flagged and excluded
# from ANN candidate generation: near-featureless frames (blank walls, black
# frames) collide massively and poison matching.
QUALITY_FLOOR = 20


class PdqKernel:
    """Default :class:`~overlap.hashing.base.HashKernel` implementation."""

    # pdq2: 256x256 working resolution in float32. pdq1 hashed at 512x512 in
    # float64 and is not comparable - 92% of its cost was smoothing pixels it
    # had upsampled to get there.
    algo_id = "pdq2"
    hash_bits = 256

    def __init__(self, quality_floor: int = QUALITY_FLOOR) -> None:
        self.quality_floor = quality_floor

    def hash_frame(self, gray: np.ndarray) -> FrameHash:
        bits_id, bits_mx, quality = compute_pdq_dihedral(gray)
        flags = 0
        if quality < self.quality_floor:
            flags |= FLAG_LOW_QUALITY
        return FrameHash(
            hash=pack_bits(bits_id),
            mirror=pack_bits(bits_mx),
            quality=quality,
            flags=flags,
        )
