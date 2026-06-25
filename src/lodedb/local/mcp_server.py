"""MCP server exposing LodeDB as local agent memory (no auth, stdio transport).

Lets coding agents (Claude Code, Cursor, etc.) use LodeDB as a local, on-disk
vector store / memory over the Model Context Protocol. It reuses the LodeDB SDK
and adds **no** storage logic of its own. Data stays on the machine. By default the search
tool runs a hybrid BM25-plus-vector ranking and returns each hit's stored text next to its
score, id, and metadata, so an agent can rank and answer in one call, and the get-by-id tool
returns a memory's text by id. The stats
tool is always metrics-only (counts, bytes) and raw query text never leaves the process.
Start the server with ``--exclude-text`` (or ``store_text=False``) to redact text: the search
tool then returns metrics only and the get-by-id tool is withdrawn.

Optional dependency: ``pip install 'lodedb[mcp]'``. Run via ``lodedb mcp`` (stdio),
e.g. as an MCP server entry in a coding agent's config.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from lodedb.local.db import LodeDB

# Tool bodies are module-level (and thin) so they are unit-testable without
# standing up the MCP stdio transport; the FastMCP tools just call these.


def _add(
    db: LodeDB,
    text: str,
    id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Adds (or replaces) one document; returns its id and the new count."""

    doc_id = db.add(text, id=id, metadata=metadata)
    return {"id": doc_id, "count": db.count()}


def _search(
    db: LodeDB,
    query: str,
    k: int = 10,
    filter: dict[str, Any] | None = None,
    *,
    mode: str = "vector",
    include_text: bool = False,
) -> list[dict[str, Any]]:
    """Returns the top-k hits as ``[{score, id, metadata}]``.

    ``mode`` is forwarded to :meth:`LodeDB.search` (``"vector"``, ``"hybrid"``, or
    ``"lexical"``). With ``include_text`` each row also carries the hit's stored
    ``text`` (read from the same raw-text store as :meth:`LodeDB.get`), so a caller
    can rank and read in one pass. The caller sets ``include_text`` only when text
    retention is on.
    """

    rows: list[dict[str, Any]] = []
    for hit in db.search(query, k=k, filter=filter, mode=mode):
        row: dict[str, Any] = {"score": hit.score, "id": hit.id, "metadata": hit.metadata}
        if include_text:
            row["text"] = db.get(hit.id)
        rows.append(row)
    return rows


def _get(db: LodeDB, id: str) -> dict[str, Any]:
    """Returns one memory's stored raw text by id (``found`` False when absent)."""

    text = db.get(id)
    return {"id": id, "found": text is not None, "text": text}


def _remove(db: LodeDB, id: str) -> dict[str, Any]:
    """Removes one document by id; returns whether it existed and the new count."""

    return {"removed": db.remove(id), "count": db.count()}


def _stats(db: LodeDB) -> dict[str, Any]:
    """Returns redacted store stats (counts, storage bytes) — never document text."""

    return db.stats()


def _default_search_mode(db: LodeDB) -> str:
    """Picks the search mode: ``"hybrid"`` when a lexical source is available, else ``"vector"``.

    Hybrid fuses a BM25 lexical ranker with the vector scan, which recovers exact tokens
    (error codes, serials, dates) that the embedding misses, so it is the better default for
    local RAG. It needs a lexical source (``store_text`` or ``index_text``), so with neither
    the server falls back to a plain vector scan instead of raising.
    """

    return "hybrid" if (db.store_text or db.index_text) else "vector"


def build_mcp_server(
    path: str | Path,
    *,
    model: str = "minilm",
    device: str = "auto",
    embedding_runtime: str = "auto",
    name: str = "lodedb",
    store_text: bool = True,
    exclude_text: bool = False,
    _embedding_backend: Any | None = None,
):
    """Builds a FastMCP server exposing LodeDB tools over one on-disk DB.

    Returns ``(server, db)``. Raises a clear :class:`ImportError` if the ``mcp``
    SDK is not installed (``pip install 'lodedb[mcp]'``).

    ``lodedb_search`` runs a hybrid (BM25 lexical + vector) ranking whenever a lexical
    source is available, falling back to a plain vector scan otherwise. Text retention
    is on by default, so search returns each hit's stored text and a ``lodedb_get`` tool
    returns a memory's text by id. Pass ``exclude_text=True`` to redact text from this
    server (search returns metrics only and ``lodedb_get`` is omitted) while still
    retaining it on disk for hybrid search, or ``store_text=False`` to keep no text on
    disk at all (same redaction, and search then falls back to a vector scan).
    """

    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - clear install hint
        raise ImportError(
            "the LodeDB MCP server needs the `mcp` SDK: pip install 'lodedb[mcp]'"
        ) from exc

    db = LodeDB(
        path=path,
        model=model,
        device=device,
        embedding_runtime=embedding_runtime,
        store_text=store_text,
        _embedding_backend=_embedding_backend,
    )
    # Text is exposed when it is retained and not explicitly redacted; this gates
    # both the inline text in search results and the get-by-id tool.
    include_text = store_text and not exclude_text
    # Prefer hybrid ranking for RAG; falls back to vector when no lexical source exists.
    search_mode = _default_search_mode(db)
    server = FastMCP(name)

    @server.tool()
    def lodedb_add(
        text: str,
        id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add (or replace) a document in local memory. Returns its id and count."""

        return _add(db, text, id=id, metadata=metadata)

    if include_text:

        @server.tool()
        def lodedb_search(
            query: str,
            k: int = 10,
            filter: dict[str, Any] | None = None,
        ) -> list[dict[str, Any]]:
            """Hybrid (BM25 + vector) search. Returns [{score, id, metadata, text}].

            Each hit carries its stored text, so an agent can rank and answer in one call.
            """

            return _search(db, query, k=k, filter=filter, mode=search_mode, include_text=True)

    else:

        @server.tool()
        def lodedb_search(
            query: str,
            k: int = 10,
            filter: dict[str, Any] | None = None,
        ) -> list[dict[str, Any]]:
            """Semantic search over local memory. Returns [{score, id, metadata}].

            Hybrid (BM25 + vector) when text is retained, else vector. Text is redacted on
            this server (started with --exclude-text or no retention).
            """

            return _search(db, query, k=k, filter=filter, mode=search_mode, include_text=False)

    if include_text:

        @server.tool()
        def lodedb_get(id: str) -> dict[str, Any]:
            """Return a memory's stored raw text by id: {id, found, text}."""

            return _get(db, id)

    @server.tool()
    def lodedb_remove(id: str) -> dict[str, Any]:
        """Remove a document by id. Returns whether it existed and the new count."""

        return _remove(db, id)

    @server.tool()
    def lodedb_stats() -> dict[str, Any]:
        """Return redacted store stats (counts, storage bytes) — never raw text."""

        return _stats(db)

    return server, db


def main() -> None:
    """Entry point for ``lodedb mcp``: serve LodeDB over stdio MCP (env-configurable)."""

    server, _db = build_mcp_server(
        os.environ.get("LODEDB_PATH", "./data"),
        model=os.environ.get("LODEDB_MODEL", "minilm"),
        device=os.environ.get("LODEDB_DEVICE", "auto"),
        embedding_runtime=os.environ.get("LODEDB_EMBEDDING_RUNTIME", "auto"),
    )
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
