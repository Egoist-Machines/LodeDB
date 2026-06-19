"""Tests for the LodeDB MCP server tool layer (no stdio transport stood up).

The tool bodies are exercised directly with a deterministic hash backend; the
FastMCP wiring is checked by building the server and listing its registered
tools (gated on the optional `mcp` SDK).
"""

from __future__ import annotations

import pytest

from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.local.db import LodeDB
from lodedb.local.mcp_server import (
    _add,
    _get,
    _remove,
    _search,
    _stats,
    build_mcp_server,
)


def _db(tmp_path, *, store_text: bool = False) -> LodeDB:
    """Opens a LodeDB with an injected deterministic hash backend."""

    return LodeDB(
        path=tmp_path,
        model="minilm",
        store_text=store_text,
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )


def _tool_names(server) -> set[str]:
    """Returns the registered FastMCP tool names, tolerating SDK API drift."""

    try:
        import asyncio

        return {tool.name for tool in asyncio.run(server.list_tools())}
    except Exception:  # noqa: BLE001 - fall back to the internal registry
        return set(server._tool_manager._tools.keys())


def test_tool_helpers_add_search_remove_stats(tmp_path):
    """The MCP tool helpers add/search/remove and report metrics-only stats."""

    db = _db(tmp_path)
    added = _add(db, "the quick brown fox", id="fox", metadata={"topic": "animals"})
    assert added == {"id": "fox", "count": 1}

    hits = _search(db, "fox", k=5)
    assert hits and hits[0]["id"] == "fox"
    assert all({"score", "id", "metadata"} <= set(h) for h in hits)

    stats = _stats(db)
    assert stats["document_count"] == 1
    assert stats["raw_payload_text_present"] is False  # never raw text

    removed = _remove(db, "fox")
    assert removed == {"removed": True, "count": 0}
    db.close()


def test_build_mcp_server_registers_the_four_tools(tmp_path):
    """build_mcp_server wires lodedb_add/lodedb_search/lodedb_remove/lodedb_stats into FastMCP."""

    pytest.importorskip("mcp")  # needs lodedb[mcp]
    server, db = build_mcp_server(
        tmp_path, _embedding_backend=HashEmbeddingBackend(native_dim=384)
    )
    try:
        import asyncio

        names = {tool.name for tool in asyncio.run(server.list_tools())}
    except Exception:  # noqa: BLE001 - fall back to the internal registry
        names = set(server._tool_manager._tools.keys())
    assert {"lodedb_add", "lodedb_search", "lodedb_remove", "lodedb_stats"} <= names
    db.close()


def test_get_helper_returns_stored_text(tmp_path):
    """The get-by-id helper returns a memory's raw text by id when storage is on."""

    db = _db(tmp_path, store_text=True)
    _add(db, "the original memory body", id="m1")
    assert _get(db, "m1") == {"id": "m1", "found": True, "text": "the original memory body"}
    assert _get(db, "absent") == {"id": "absent", "found": False, "text": None}
    db.close()


def test_lodedb_get_registered_only_when_store_text_enabled(tmp_path):
    """lodedb_get is exposed only when the server opts into raw-text storage."""

    pytest.importorskip("mcp")  # needs lodedb[mcp]
    backend = HashEmbeddingBackend(native_dim=384)
    server_on, db_on = build_mcp_server(
        tmp_path / "on", store_text=True, _embedding_backend=backend
    )
    server_off, db_off = build_mcp_server(
        tmp_path / "off", store_text=False, _embedding_backend=HashEmbeddingBackend(native_dim=384)
    )
    try:
        assert "lodedb_get" in _tool_names(server_on)
        assert "lodedb_get" not in _tool_names(server_off)
    finally:
        db_on.close()
        db_off.close()
