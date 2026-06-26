"""Read-only exporter for a persisted LlamaIndex ``StorageContext`` (SimpleVectorStore).

LlamaIndex persists an index as a directory of JSON files (``docstore.json``,
``index_store.json``, ``default__vector_store.json``). The docstore is the source
of truth for node text and metadata, so this importer loads the persisted
``StorageContext`` and streams each stored ``TextNode`` as a text-replay
:class:`ExportedRow`, preserving the node id, text, metadata, and the SOURCE
``ref_doc_id`` relationship. LodeDB re-embeds the text with the chosen preset; the
LodeDB LlamaIndex adapter rebuilds the SOURCE relationship from ``ref_doc_id`` on
read. The persist directory is opened read-only and never rewritten.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from lodedb.local.migrate.sources.base import (
    MODE_TEXT_REPLAY,
    ExportedRow,
    SourceExport,
    SourceExportError,
)


class LlamaIndexSimpleExport(SourceExport):
    """Streams the nodes of a persisted LlamaIndex docstore as text-replay rows."""

    def __init__(self, persist_dir: str) -> None:
        """Loads the persisted docstore and prepares the node stream."""

        nodes = _load_docstore_nodes(persist_dir)
        super().__init__(
            framework="llama-index",
            provider="simple",
            mode=MODE_TEXT_REPLAY,
            location=str(persist_dir),
            vector_dim=None,
            count=len(nodes),
            notes={
                "fidelity": (
                    "node id, text, metadata, and the SOURCE ref_doc_id are preserved; other "
                    "node relationships and non-metadata node fields are not round-tripped"
                )
            },
        )
        self._nodes = nodes

    def iter_rows(self) -> Iterator[ExportedRow]:
        """Yields one text-replay row per stored node."""

        from llama_index.core.schema import MetadataMode

        for node in self._nodes:
            try:
                text = node.get_content(metadata_mode=MetadataMode.NONE)
            except Exception:  # noqa: BLE001 - non-text node: skip rather than fail the run
                text = None
            if not text or not str(text).strip():
                # The LodeDB LlamaIndex adapter is text-path and cannot index empty
                # nodes; surface them as skipped rows by carrying no text.
                yield ExportedRow(
                    id=str(node.node_id),
                    text=None,
                    metadata=dict(node.metadata or {}),
                    ref_doc_id=node.ref_doc_id,
                )
                continue
            yield ExportedRow(
                id=str(node.node_id),
                text=str(text),
                metadata=dict(node.metadata or {}),
                ref_doc_id=node.ref_doc_id,
            )


def _load_docstore_nodes(persist_dir: str) -> list[Any]:
    """Returns the list of nodes from a persisted LlamaIndex docstore.

    Loads only the docstore (not the vector store), since the LodeDB adapter is
    text-path and re-embeds text. Raises a clear error when llama-index-core is
    absent or the directory is not a valid persist dir.
    """

    directory = Path(persist_dir)
    if not directory.is_dir():
        raise SourceExportError(f"LlamaIndex persist directory not found: {directory}")
    if not (directory / "docstore.json").is_file():
        raise SourceExportError(
            f"{directory} does not look like a LlamaIndex persist dir (no docstore.json)"
        )

    try:
        from llama_index.core.storage.docstore import SimpleDocumentStore
    except ImportError as exc:
        raise SourceExportError(
            "exporting a LlamaIndex store needs llama-index-core: pip install 'lodedb[llama-index]'"
        ) from exc

    try:
        docstore = SimpleDocumentStore.from_persist_dir(str(directory))
    except Exception as exc:  # noqa: BLE001 - any load failure is a clean export error
        raise SourceExportError(f"could not load the LlamaIndex docstore at {directory}: {exc}") \
            from exc
    docs = getattr(docstore, "docs", None)
    if not isinstance(docs, dict):
        raise SourceExportError("loaded LlamaIndex docstore exposes no 'docs' mapping")
    return list(docs.values())
