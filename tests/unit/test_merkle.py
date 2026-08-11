from __future__ import annotations

import hashlib
from pathlib import Path

from overlap.ingest.merkle import merkle_root, sha256_file


def test_sha256_file(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello overlap")
    assert sha256_file(p) == hashlib.sha256(b"hello overlap").digest()


def test_empty_root_is_zero() -> None:
    assert merkle_root([]) == bytes(32)


def test_root_is_order_independent() -> None:
    leaves = [("a.mp4", bytes(32)), ("b.mp4", bytes([1]) * 32), ("c.mp4", bytes([2]) * 32)]
    assert merkle_root(leaves) == merkle_root(list(reversed(leaves)))


def test_root_changes_on_content_change() -> None:
    base = [("a.mp4", bytes(32)), ("b.mp4", bytes([1]) * 32)]
    tampered = [("a.mp4", bytes(32)), ("b.mp4", bytes([9]) * 32)]
    assert merkle_root(base) != merkle_root(tampered)


def test_root_changes_on_rename() -> None:
    base = [("a.mp4", bytes(32))]
    renamed = [("z.mp4", bytes(32))]
    assert merkle_root(base) != merkle_root(renamed)


def test_single_leaf_and_odd_counts() -> None:
    one = merkle_root([("a", bytes(32))])
    three = merkle_root([("a", bytes(32)), ("b", bytes(32)), ("c", bytes(32))])
    assert len(one) == 32
    assert len(three) == 32
    assert one != three
