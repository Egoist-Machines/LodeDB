"""lodedb.graph — a hybrid knowledge-graph layer over LodeDB.

Topology (nodes, typed edges) lives in an embedded SQLite sidecar built for
deterministic traversal; LodeDB serves as the rebuildable semantic index for
entry-point retrieval. See :class:`KnowledgeGraph`.
"""

from __future__ import annotations

from ._store import Edge, Node
from .knowledge_graph import KnowledgeGraph, Subgraph

__all__ = ["Edge", "KnowledgeGraph", "Node", "Subgraph"]
