"""Ingestion: file discovery, fingerprinting pipeline, and dataset digests."""

from overlap.ingest.merkle import merkle_root, sha256_file
from overlap.ingest.model import FileResult, StreamResult
from overlap.ingest.pipeline import IndexStats, discover_files, index_paths

__all__ = [
    "FileResult",
    "IndexStats",
    "StreamResult",
    "discover_files",
    "index_paths",
    "merkle_root",
    "sha256_file",
]
