"""File digests and the dataset Merkle root.

The Merkle root binds a manifest to the exact set of files it describes:
``overlap verify`` recomputes it over delivered files after purchase. Spec
(also documented in docs/manifest-spec.md):

- leaf   = SHA-256( 0x00 || relpath_utf8 || 0x00 || file_sha256 )
- node   = SHA-256( 0x01 || left || right )
- levels pair left-to-right over leaves sorted by relpath; an unpaired node
  is promoted unchanged
- empty dataset root = 32 zero bytes

The 0x00/0x01 domain prefixes prevent second-preimage attacks (a leaf can
never be re-interpreted as an interior node).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

_CHUNK = 1 << 20


def sha256_file(path: Path) -> bytes:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(_CHUNK)
            if not block:
                break
            h.update(block)
    return h.digest()


def merkle_root(leaves: list[tuple[str, bytes]]) -> bytes:
    """Root over (relpath, file_sha256) pairs. Order-independent: sorted here."""
    if not leaves:
        return bytes(32)
    level = [
        hashlib.sha256(b"\x00" + relpath.encode("utf-8") + b"\x00" + digest).digest()
        for relpath, digest in sorted(leaves)
    ]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(hashlib.sha256(b"\x01" + level[i] + level[i + 1]).digest())
        if len(level) % 2 == 1:
            nxt.append(level[-1])
        level = nxt
    return level[0]
