"""lodedb.graph is a hybrid knowledge-graph layer over LodeDB.

Topology (nodes, typed edges) lives in an embedded SQLite sidecar built for
deterministic traversal; LodeDB serves as the rebuildable semantic index for
entry-point retrieval. See :class:`KnowledgeGraph`.

:class:`TemporalKnowledgeGraph` is the bi-temporal successor, backed by the native
``lodedb-graph`` Rust crate: every fact carries event time (``valid_at``/
``invalid_at``) and transaction time (``created_at``/``expired_at``), contradictions
invalidate rather than delete, and reads take an ``as_of`` frame. Being native, it
also runs on-device (Swift), not just in Python.
"""

from __future__ import annotations

from ._store import Edge, Node
from .knowledge_graph import KnowledgeGraph, Subgraph
from .temporal import (
    Embedder,
    TemporalKnowledgeGraph,
    episode_mentions_reranker,
    maximal_marginal_relevance,
    node_distance_reranker,
    rrf,
)

__all__ = [
    "Edge",
    "Embedder",
    "KnowledgeGraph",
    "Node",
    "Subgraph",
    "TemporalKnowledgeGraph",
    "episode_mentions_reranker",
    "maximal_marginal_relevance",
    "node_distance_reranker",
    "rrf",
]
