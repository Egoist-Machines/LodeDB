"""The read-only source-export contract shared by every importer.

An *importer* knows how to read one source store (a LangChain ``InMemoryVectorStore``
save file, a LlamaIndex ``StorageContext``, a mem0 Qdrant collection, a pgvector
table) and yield a uniform stream of :class:`ExportedRow` without ever mutating the
source. The runner then replays those rows into a fresh LodeDB target.

Two replay modes mirror the shipped adapters:

- ``text-replay`` — the row carries canonical text + metadata; LodeDB embeds it.
- ``vector-preserve`` — the row carries the source's own embedding (and payload
  text/metadata); LodeDB stores the vector verbatim.

Importers are read-only by construction: they open the source, iterate, and close.
Nothing here issues a write, delete, drop, or schema change against a source.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

# Replay modes. ``text-replay`` is the default for LangChain/LlamaIndex (LodeDB
# owns embedding); ``vector-preserve`` is the default for mem0 and direct pgvector
# (the source/app owns embedding and the vectors must be carried over verbatim).
MODE_TEXT_REPLAY = "text-replay"
MODE_VECTOR_PRESERVE = "vector-preserve"


@dataclass
class ExportedRow:
    """One source row in the uniform export shape.

    ``id`` is the source's stable id. ``text`` is the canonical document text when
    available (required for text-replay; optional payload otherwise). ``metadata``
    is the scalar/JSON-ish metadata to carry into LodeDB. ``vector`` is the
    source's embedding for vector-preserve mode (``None`` for text-replay).
    ``ref_doc_id`` is the LlamaIndex source-document relationship, carried so the
    adapter can rebuild it. ``raw_payload`` is the full opaque payload (mem0) kept
    in LodeDB's raw-text sidecar, never in redacted metadata.
    """

    id: str
    text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    vector: list[float] | None = None
    ref_doc_id: str | None = None
    raw_payload: dict[str, Any] | None = None


@dataclass
class SourceExport:
    """A read-only handle over one source store ready to stream rows.

    ``framework`` is one of ``langchain`` / ``llama-index`` / ``mem0`` / ``None``
    (a direct provider). ``provider`` is the concrete source store
    (``in-memory`` / ``simple`` / ``qdrant`` / ``pgvector`` / …). ``mode`` is the
    replay mode the rows are shaped for. ``vector_dim`` is the embedding dimension
    for vector-preserve (``None`` for text-replay). ``count`` is the row total when
    cheaply known (``None`` if it can only be learned by streaming). ``location``
    is an opaque, possibly-secret source locator that the report layer fingerprints
    and never stores verbatim.
    """

    framework: str | None
    provider: str
    mode: str
    location: str
    vector_dim: int | None = None
    count: int | None = None
    warnings: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    def iter_rows(self) -> Iterator[ExportedRow]:  # pragma: no cover - overridden
        """Yields exported rows. Implementations override this."""

        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - default no-op
        """Releases any source handle. Default no-op for in-memory sources."""

        return None


class SourceExportError(RuntimeError):
    """Raised when a source cannot be opened or exported (read-only failure)."""
