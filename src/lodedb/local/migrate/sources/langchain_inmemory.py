"""Read-only exporter for a LangChain ``InMemoryVectorStore`` save file.

``InMemoryVectorStore.dump(path)`` writes a JSON file whose deserialized ``store``
is a ``{id: {"id", "vector", "text", "metadata"}}`` map. This importer loads that
file through LangChain's own ``InMemoryVectorStore.load`` (so the on-disk format is
read by the library that wrote it, not re-parsed by hand) and streams each entry as
a text-replay :class:`ExportedRow`: LodeDB re-embeds the text with the chosen
preset. The source file is opened read-only and never rewritten.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from lodedb.local.migrate.sources.base import (
    MODE_TEXT_REPLAY,
    ExportedRow,
    SourceExport,
    SourceExportError,
)


class _NullEmbeddings:
    """A no-op ``Embeddings`` stand-in.

    ``InMemoryVectorStore.load`` requires an embedding object, but a pure read of
    the persisted ``store`` never calls it. We pass this so the import needs no real
    embedding model (and no network) just to read documents back out.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - unused
        raise RuntimeError("embedding is not used when exporting an InMemoryVectorStore")

    def embed_query(self, text: str) -> list[float]:  # pragma: no cover - unused
        raise RuntimeError("embedding is not used when exporting an InMemoryVectorStore")


class LangChainInMemoryExport(SourceExport):
    """Streams a LangChain ``InMemoryVectorStore`` dump as text-replay rows."""

    def __init__(self, path: str) -> None:
        """Loads the save file via LangChain and prepares the row stream."""

        store_map = _load_store_map(path)
        super().__init__(
            framework="langchain",
            provider="in-memory",
            mode=MODE_TEXT_REPLAY,
            location=str(path),
            vector_dim=None,
            count=len(store_map),
        )
        self._store_map = store_map

    def iter_rows(self) -> Iterator[ExportedRow]:
        """Yields one text-replay row per stored document."""

        for key, entry in self._store_map.items():
            if not isinstance(entry, dict):
                continue
            doc_id = str(entry.get("id") or key)
            text = entry.get("text")
            metadata = entry.get("metadata") or {}
            yield ExportedRow(
                id=doc_id,
                text=text if isinstance(text, str) else None,
                metadata=dict(metadata) if isinstance(metadata, dict) else {},
                vector=None,
            )


def _load_store_map(path: str) -> dict[str, Any]:
    """Returns the ``{id: entry}`` store map from a dump, via LangChain's loader.

    Tries the public ``InMemoryVectorStore.load`` first (version-robust: LangChain
    deserializes its own format). Falls back to reading the JSON directly for a
    plain dump, so a missing optional langchain-core install still exports.
    """

    save_path = Path(path)
    if not save_path.is_file():
        raise SourceExportError(f"LangChain InMemoryVectorStore save file not found: {save_path}")

    try:
        from langchain_core.vectorstores import InMemoryVectorStore
    except ImportError:
        return _load_store_map_from_json(save_path)

    try:
        store = InMemoryVectorStore.load(str(save_path), _NullEmbeddings())
    except Exception as exc:  # noqa: BLE001 - fall back to direct JSON on any loader issue
        try:
            return _load_store_map_from_json(save_path)
        except SourceExportError:
            raise SourceExportError(
                f"could not load the LangChain InMemoryVectorStore at {save_path}: {exc}"
            ) from exc
    store_map = getattr(store, "store", None)
    if not isinstance(store_map, dict):
        raise SourceExportError("loaded InMemoryVectorStore has no usable 'store' mapping")
    return store_map


def _load_store_map_from_json(save_path: Path) -> dict[str, Any]:
    """Reads a plain ``InMemoryVectorStore`` dump JSON without langchain-core.

    Handles both the bare ``{id: entry}`` map and a top-level ``{"store": {...}}``
    wrapper. Entries that are LangChain-serialized objects (``{"lc": ...}``) cannot
    be decoded without the library and raise a clear error.
    """

    try:
        data = json.loads(save_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceExportError(f"could not read the dump file {save_path}: {exc}") from exc
    if isinstance(data, dict) and "store" in data and isinstance(data["store"], dict):
        data = data["store"]
    if not isinstance(data, dict):
        raise SourceExportError("dump file is not an InMemoryVectorStore store mapping")
    if any(isinstance(v, dict) and "lc" in v for v in data.values()):
        raise SourceExportError(
            "this dump uses LangChain object serialization; install lodedb[langchain] so the "
            "file can be loaded by langchain-core"
        )
    return data
