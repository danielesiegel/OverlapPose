"""Exception hierarchy. All overlap-raised errors derive from OverlapError."""


class OverlapError(Exception):
    """Base class for all errors raised by overlap."""


class IndexError_(OverlapError):
    """The index directory is missing, locked, or has an incompatible schema."""


class ManifestError(OverlapError):
    """A manifest file is malformed, truncated, or schema-incompatible."""


class ReaderError(OverlapError):
    """A media file could not be opened or decoded."""


class ConfigError(OverlapError):
    """Invalid configuration value or unreadable config file."""
