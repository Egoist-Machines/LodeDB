"""lodedb is the public API for LodeDB.

Re-exports the local-first, on-disk SDK (:mod:`lodedb.local`) and the ``lodedb``
CLI, so you can write ``import lodedb`` / ``from lodedb import LodeDB``. LodeDB
runs in-process with no server, no network, and no authentication: your data
stays on local disk.
"""

from __future__ import annotations

from lodedb.local import (
    LOCAL_MODEL_PRESETS,
    AnnOptions,
    Appender,
    Checkpointer,
    ConcurrentWriterError,
    ImageEmbeddingUnsupportedError,
    LodeCollection,
    LodeDB,
    LodeLateInteractionHit,
    LodeLateInteractionIndex,
    LodeSearchHit,
    ReadOnlyError,
    RescoreOptions,
    local_capability_report,
)
from lodedb.local.cli import app, main

# Keep in sync with `version` in pyproject.toml (the release workflow asserts they match).
__version__ = "1.4.0"

__all__ = [
    "LOCAL_MODEL_PRESETS",
    "AnnOptions",
    "Appender",
    "Checkpointer",
    "ConcurrentWriterError",
    "ImageEmbeddingUnsupportedError",
    "LodeCollection",
    "LodeDB",
    "LodeLateInteractionHit",
    "LodeLateInteractionIndex",
    "LodeSearchHit",
    "ReadOnlyError",
    "RescoreOptions",
    "app",
    "local_capability_report",
    "main",
]
