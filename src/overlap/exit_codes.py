"""Process exit codes. These are a documented, stable part of the CLI contract."""

from enum import IntEnum


class ExitCode(IntEnum):
    OK = 0
    RUNTIME_ERROR = 1
    USAGE_ERROR = 2  # Click/Typer convention for bad flags or arguments
    OVERLAP_FOUND = 3  # `compare --fail-over`: overlap at or above the given percentage
    PARTIAL_FAILURE = 4  # completed, but some inputs failed (details on stderr / --json)
    INTERRUPTED = 130  # SIGINT after a graceful checkpoint
