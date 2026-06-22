"""SQLite topology store — the source-of-truth adjacency for a knowledge graph.

LodeDB itself is an exact vector index, not a graph engine: ``search`` ranks the
top-``k`` semantically similar items, which is the wrong primitive for "every
edge whose ``src`` is X". So a :class:`~lodedb.graph.KnowledgeGraph` keeps the
*topology* (nodes, typed edges, properties) here, in a small embedded SQLite
sidecar built for deterministic traversal, and uses LodeDB as the rebuildable
*semantic* index over node/edge text. This module owns the topology half: CRUD
plus the adjacency queries (`neighbors`, `edges_for`) that back k-hop traversal.

It is stdlib-only (``sqlite3``) and holds no embeddings — vectors live in LodeDB.
"""

from __future__ import annotations

import json
import sqlite3
from array import array
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id          TEXT PRIMARY KEY,
    type        TEXT NOT NULL DEFAULT '',
    label       TEXT NOT NULL DEFAULT '',
    properties  TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS edges (
    id          TEXT PRIMARY KEY,
    src         TEXT NOT NULL,
    relation    TEXT NOT NULL,
    dst         TEXT NOT NULL,
    properties  TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS node_vectors (
    node_id TEXT PRIMARY KEY REFERENCES nodes(id) ON DELETE CASCADE,
    vector  BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
CREATE INDEX IF NOT EXISTS idx_edges_rel ON edges(relation);
CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
"""

_DIRECTIONS = frozenset({"out", "in", "both"})

# Max ids per IN-clause chunk; kept well under SQLite's default bound-parameter
# limit (~32k) since direction="both" binds each id twice.
_IN_CHUNK = 400


@dataclass(frozen=True)
class Node:
    """One graph node: a stable id, a type, a label (the text indexed for
    semantic retrieval), and arbitrary JSON-able properties."""

    id: str
    type: str = ""
    label: str = ""
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    """One typed, directed edge ``(src) -[relation]-> (dst)`` with properties."""

    id: str
    src: str
    relation: str
    dst: str
    properties: dict[str, Any] = field(default_factory=dict)


class TopologyStore:
    """Embedded SQLite store of nodes/edges with traversal queries."""

    def __init__(self, path: str | Path) -> None:
        """Opens (creating if needed) the topology database at ``path``."""

        self.path = Path(path)
        # check_same_thread=False: the KnowledgeGraph mirrors LodeDB's
        # single-writer model, but a loopback dev server may touch it from a
        # worker thread; SQLite still serializes writes internally.
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        """Closes the underlying connection (state stays durable on disk)."""

        self._conn.close()

    # -- nodes --------------------------------------------------------------

    def upsert_node(self, node: Node) -> None:
        """Inserts or replaces a node by id."""

        self.upsert_nodes((node,))

    def upsert_nodes(self, nodes: Iterable[Node]) -> None:
        """Inserts or replaces many nodes in one transaction (executemany)."""

        rows = [(n.id, n.type, n.label, json.dumps(n.properties)) for n in nodes]
        if not rows:
            return
        with self._conn:
            self._conn.executemany(
                "INSERT INTO nodes (id, type, label, properties) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET type=excluded.type, label=excluded.label, "
                "properties=excluded.properties",
                rows,
            )

    def get_node(self, node_id: str) -> Node | None:
        """Returns the node with ``node_id`` or ``None`` if absent."""

        row = self._conn.execute(
            "SELECT id, type, label, properties FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        return _node_from_row(row) if row is not None else None

    def get_nodes(self, node_ids: Iterable[str]) -> list[Node]:
        """Returns the nodes for ``node_ids`` (chunked IN; missing ids omitted).

        The batched read behind k-hop and hybrid retrieval: one query per
        ``_IN_CHUNK`` ids instead of one round-trip per visited node.
        """

        ids = [str(value) for value in node_ids]
        if not ids:
            return []
        found: list[Node] = []
        for start in range(0, len(ids), _IN_CHUNK):
            batch = ids[start : start + _IN_CHUNK]
            placeholders = ",".join("?" for _ in batch)
            rows = self._conn.execute(
                f"SELECT id, type, label, properties FROM nodes WHERE id IN ({placeholders})",
                batch,
            ).fetchall()
            found.extend(_node_from_row(row) for row in rows)
        return found

    def set_node_vectors(self, items: Iterable[tuple[str, Sequence[float]]]) -> None:
        """Persists raw float32 node vectors (opt-in), so the semantic index is
        rebuildable from this source-of-truth store, including vector-in nodes."""

        rows = [
            (str(node_id), array("f", [float(value) for value in vector]).tobytes())
            for node_id, vector in items
        ]
        if not rows:
            return
        with self._conn:
            self._conn.executemany(
                "INSERT INTO node_vectors (node_id, vector) VALUES (?, ?) "
                "ON CONFLICT(node_id) DO UPDATE SET vector=excluded.vector",
                rows,
            )

    def get_node_vector(self, node_id: str) -> list[float] | None:
        """Returns a node's retained raw vector, or ``None`` if none is stored."""

        row = self._conn.execute(
            "SELECT vector FROM node_vectors WHERE node_id = ?", (node_id,)
        ).fetchone()
        if row is None:
            return None
        values = array("f")
        values.frombytes(row["vector"])
        return list(values)

    def iter_node_vectors(self) -> Iterator[tuple[str, list[float]]]:
        """Iterates over all retained ``(node_id, vector)`` pairs."""

        for row in self._conn.execute("SELECT node_id, vector FROM node_vectors").fetchall():
            values = array("f")
            values.frombytes(row["vector"])
            yield str(row["node_id"]), list(values)

    def remove_node_vector(self, node_id: str) -> None:
        """Drops a node's retained vector (e.g. when it switches to label indexing)."""

        with self._conn:
            self._conn.execute("DELETE FROM node_vectors WHERE node_id = ?", (str(node_id),))

    def remove_node(self, node_id: str) -> tuple[bool, list[str]]:
        """Removes a node and all incident edges.

        Returns ``(existed, removed_edge_ids)`` so the caller can also drop the
        node's and incident edges' entries from the semantic index.
        """

        with self._conn:
            incident = [
                str(row["id"])
                for row in self._conn.execute(
                    "SELECT id FROM edges WHERE src = ? OR dst = ?", (node_id, node_id)
                ).fetchall()
            ]
            self._conn.execute("DELETE FROM edges WHERE src = ? OR dst = ?", (node_id, node_id))
            cursor = self._conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
            return cursor.rowcount > 0, incident

    def all_node_ids(self) -> list[str]:
        """Returns every node id (used by reindex to detect orphaned index docs)."""

        return [str(row["id"]) for row in self._conn.execute("SELECT id FROM nodes").fetchall()]

    def iter_nodes(self) -> Iterator[Node]:
        """Iterates over all nodes."""

        for row in self._conn.execute(
            "SELECT id, type, label, properties FROM nodes"
        ).fetchall():
            yield _node_from_row(row)

    def node_count(self) -> int:
        """Returns the number of nodes."""

        return int(self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0])

    # -- edges --------------------------------------------------------------

    def upsert_edge(self, edge: Edge) -> None:
        """Inserts or replaces an edge by id."""

        self.upsert_edges((edge,))

    def upsert_edges(self, edges: Iterable[Edge]) -> None:
        """Inserts or replaces many edges in one transaction (executemany)."""

        rows = [
            (e.id, e.src, e.relation, e.dst, json.dumps(e.properties)) for e in edges
        ]
        if not rows:
            return
        with self._conn:
            self._conn.executemany(
                "INSERT INTO edges (id, src, relation, dst, properties) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET src=excluded.src, relation=excluded.relation, "
                "dst=excluded.dst, properties=excluded.properties",
                rows,
            )

    def get_edge(self, edge_id: str) -> Edge | None:
        """Returns the edge with ``edge_id`` or ``None`` if absent."""

        row = self._conn.execute(
            "SELECT id, src, relation, dst, properties FROM edges WHERE id = ?", (edge_id,)
        ).fetchone()
        return _edge_from_row(row) if row is not None else None

    def remove_edge(self, edge_id: str) -> bool:
        """Removes one edge by id. Returns True if it existed."""

        with self._conn:
            cursor = self._conn.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
            return cursor.rowcount > 0

    def iter_edges(self) -> Iterator[Edge]:
        """Iterates over all edges."""

        for row in self._conn.execute(
            "SELECT id, src, relation, dst, properties FROM edges"
        ).fetchall():
            yield _edge_from_row(row)

    def edge_count(self) -> int:
        """Returns the number of edges."""

        return int(self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0])

    # -- traversal ----------------------------------------------------------

    def neighbors(
        self,
        node_id: str,
        *,
        direction: str = "out",
        relation: str | None = None,
    ) -> list[Edge]:
        """Returns edges incident to ``node_id`` in ``direction`` (out/in/both)."""

        return self.edges_for([node_id], direction=direction, relation=relation)

    def edges_for(
        self,
        node_ids: Iterable[str],
        *,
        direction: str = "out",
        relation: str | None = None,
    ) -> list[Edge]:
        """Returns edges incident to any of ``node_ids`` (the batched frontier
        expansion that backs k-hop traversal).

        ``direction`` is ``"out"`` (``src`` in the set), ``"in"`` (``dst`` in the
        set), or ``"both"``. ``relation`` optionally restricts the edge type.
        """

        if direction not in _DIRECTIONS:
            raise ValueError(f"direction must be one of {sorted(_DIRECTIONS)}, got {direction!r}")
        ids = [str(value) for value in node_ids]
        if not ids:
            return []
        # Chunk the IN-list so a large traversal frontier never exceeds SQLite's
        # bound-parameter limit (each id binds up to twice for direction="both").
        # Dedup by edge id across chunks (an edge can match via src in one chunk
        # and dst in another).
        found: dict[str, Edge] = {}
        for start in range(0, len(ids), _IN_CHUNK):
            batch = ids[start : start + _IN_CHUNK]
            placeholders = ",".join("?" for _ in batch)
            if direction == "out":
                where = f"src IN ({placeholders})"
                params: list[Any] = list(batch)
            elif direction == "in":
                where = f"dst IN ({placeholders})"
                params = list(batch)
            else:
                where = f"src IN ({placeholders}) OR dst IN ({placeholders})"
                params = list(batch) + list(batch)
            if relation is not None:
                where = f"({where}) AND relation = ?"
                params.append(relation)
            for row in self._conn.execute(
                f"SELECT id, src, relation, dst, properties FROM edges WHERE {where}", params
            ).fetchall():
                edge = _edge_from_row(row)
                found[edge.id] = edge
        return list(found.values())


def _node_from_row(row: sqlite3.Row) -> Node:
    return Node(
        id=str(row["id"]),
        type=str(row["type"]),
        label=str(row["label"]),
        properties=json.loads(row["properties"]),
    )


def _edge_from_row(row: sqlite3.Row) -> Edge:
    return Edge(
        id=str(row["id"]),
        src=str(row["src"]),
        relation=str(row["relation"]),
        dst=str(row["dst"]),
        properties=json.loads(row["properties"]),
    )
