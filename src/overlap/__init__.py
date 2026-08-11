"""overlap - perceptual fingerprinting and overlap detection for robotics datasets.

Public API:

- :func:`overlap.index_paths` - fingerprint media files into a local index
- :func:`overlap.export_manifest` - write a shareable fingerprint manifest
- :func:`overlap.compare` - compare a manifest against a local index
- :func:`overlap.verify` - verify delivered files against a manifest's Merkle root

Everything else is internal and may change between minor versions.
"""

from importlib.metadata import PackageNotFoundError, version
from typing import Any

try:
    __version__ = version("overlap-cli")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0.dev0"

__all__ = ["__version__", "compare", "export_manifest", "index_paths", "verify"]

_API = {
    "index_paths": ("overlap.ingest", "index_paths"),
    "export_manifest": ("overlap.store.manifest", "export_manifest"),
    "compare": ("overlap.match", "compare_manifest_file"),
    "verify": ("overlap.match", "verify_delivery"),
}


def __getattr__(name: str) -> Any:
    """Lazy public API: keeps `import overlap` light (no cv2/faiss import)."""
    if name in _API:
        module_name, attr = _API[name]
        import importlib

        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
