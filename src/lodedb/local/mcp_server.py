"""MCP server exposing LodeDB as local agent memory (no auth, stdio transport).

Lets coding agents (Claude Code, Cursor, etc.) use LodeDB as a local, on-disk
vector store / memory over the Model Context Protocol. It reuses the LodeDB SDK
and adds **no** storage logic of its own. Data stays on the machine. The search and
stats tools surface only metrics (scores, ids, counts, bytes) — never document or query
text; the get-by-id tool returns a memory's stored text only when text retention is on.

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
) -> list[dict[str, Any]]:
    """Returns the top-k hits as ``[{score, id, metadata}]`` (no raw text)."""

    return [
        {"score": hit.score, "id": hit.id, "metadata": hit.metadata}
        for hit in db.search(query, k=k, filter=filter)
    ]


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


def build_mcp_server(
    path: str | Path,
    *,
    model: str = "minilm",
    device: str = "auto",
    name: str = "lodedb",
    store_text: bool = True,
    _embedding_backend: Any | None = None,
):
    """Builds a FastMCP server exposing LodeDB tools over one on-disk DB.

    Returns ``(server, db)``. Raises a clear :class:`ImportError` if the ``mcp``
    SDK is not installed (``pip install 'lodedb[mcp]'``). By default the server
    exposes a ``lodedb_get`` tool that returns a memory's original text by id; pass
    ``store_text=False`` to stop retaining text and omit the tool.
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
        store_text=store_text,
        _embedding_backend=_embedding_backend,
    )
    server = FastMCP(name)

    @server.tool()
    def lodedb_add(
        text: str,
        id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add (or replace) a document in local memory. Returns its id and count."""

        return _add(db, text, id=id, metadata=metadata)

    @server.tool()
    def lodedb_search(
        query: str,
        k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search over local memory. Returns [{score, id, metadata}]."""

        return _search(db, query, k=k, filter=filter)

    if store_text:

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
    )
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
