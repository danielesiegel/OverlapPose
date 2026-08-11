"""Post-purchase verification: do delivered bytes match the manifest?

Matching is content-addressed (sha256), so renamed or anonymized-path
deliveries still verify; the manifest's Merkle root is recomputed as an
integrity check of the manifest itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from overlap.ingest.merkle import merkle_root, sha256_file
from overlap.store.manifest import read_manifest

if TYPE_CHECKING:
    from pathlib import Path


def verify_delivery(manifest_path: Path, data_dir: Path) -> dict[str, Any]:
    manifest = read_manifest(manifest_path)

    delivered: dict[bytes, list[str]] = {}
    for path in sorted(p for p in data_dir.rglob("*") if p.is_file()):
        digest = sha256_file(path)
        delivered.setdefault(digest, []).append(path.relative_to(data_dir).as_posix())

    matched: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    manifest_shas = set()
    for mfile in manifest.files:
        manifest_shas.add(mfile.sha256)
        paths = delivered.get(mfile.sha256)
        if paths:
            matched.append({"relpath": mfile.relpath, "delivered_as": paths[0]})
        else:
            missing.append({"relpath": mfile.relpath, "sha256": mfile.sha256.hex()})

    extra = sorted(
        path for digest, paths in delivered.items() if digest not in manifest_shas for path in paths
    )

    recomputed_root = merkle_root([(f.relpath, f.sha256) for f in manifest.files])
    manifest_consistent = recomputed_root == manifest.merkle_root

    return {
        "schema": "verify/1",
        "manifest": str(manifest_path),
        "data_dir": str(data_dir),
        "manifest_merkle_ok": manifest_consistent,
        "files_expected": len(manifest.files),
        "files_matched": len(matched),
        "files_missing": len(missing),
        "files_extra": len(extra),
        "ok": manifest_consistent and not missing,
        "missing": missing,
        "extra": extra,
        "matched": matched,
    }
