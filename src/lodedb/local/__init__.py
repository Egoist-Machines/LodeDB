"""The local-first LodeDB SDK: embedded, on-disk, no-auth.

Public surface:

- :class:`LodeDB` — the embedded SDK entrypoint.
- :class:`LodeSearchHit` — one redacted ``(score, id, metadata)`` result row.
- :func:`resolve_local_device` / :func:`build_local_embedding_backend` —
  embedding device selection (MPS / CUDA / CPU).
- :func:`local_capability_report` — the data behind ``lodedb doctor``.
"""

from lodedb.local.backends import (
    LocalEmbeddingResolution,
    build_local_embedding_backend,
    resolve_local_device,
)
from lodedb.local.db import LodeDB, LodeSearchHit
from lodedb.local.doctor import local_capability_report
from lodedb.local.presets import LOCAL_MODEL_PRESETS, LocalModelPreset, resolve_preset

__all__ = [
    "LOCAL_MODEL_PRESETS",
    "LodeDB",
    "LodeSearchHit",
    "LocalEmbeddingResolution",
    "LocalModelPreset",
    "build_local_embedding_backend",
    "local_capability_report",
    "resolve_local_device",
    "resolve_preset",
]
