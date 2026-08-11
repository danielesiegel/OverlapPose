"""Matching: candidate generation, temporal chaining, scoring, comparison."""

from overlap.match.audit import audit_sample
from overlap.match.chain import ChainParams, HoughChainMatcher, RunGeometry
from overlap.match.compare import (
    compare_manifest,
    compare_manifest_file,
    compare_two_manifests,
    self_dedupe,
)
from overlap.match.score import ScoreParams, assign_tier, confidence, union_ms
from overlap.match.verify import verify_delivery

__all__ = [
    "ChainParams",
    "HoughChainMatcher",
    "RunGeometry",
    "ScoreParams",
    "assign_tier",
    "audit_sample",
    "compare_manifest",
    "compare_manifest_file",
    "compare_two_manifests",
    "confidence",
    "self_dedupe",
    "union_ms",
    "verify_delivery",
]
