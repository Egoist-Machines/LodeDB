"""TemporalKnowledgeGraph — a Pythonic wrapper over the native ``lodedb-graph``
bi-temporal knowledge graph (the ``_turbovec.graph`` submodule).

This is the temporal successor to :class:`lodedb.graph.KnowledgeGraph`: same
architecture (an authoritative topology store + a rebuildable LodeDB semantic
index), plus Graphiti-style bi-temporality — every fact carries event time
(``valid_at``/``invalid_at``) and transaction time (``created_at``/``expired_at``),
contradictions invalidate rather than delete, and reads take an ``as_of`` frame.

Unlike Graphiti, this performs no LLM extraction/resolution/contradiction detection:
it stores what it is given (:meth:`add_fact` is the analogue of Graphiti's
``add_triplet``) and answers temporal queries. Embedding is caller-supplied — pass an
``embedder`` with ``dimension`` and ``embed(texts, role)`` — mirroring how the native
core keeps embedding in the binding layer.

The heavy lifting (SQLite topology, the LodeDB index, invalidation, as-of) runs in
Rust; this layer only marshals dict ⇄ JSON properties and the ``as_of`` frame.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

__all__ = ["TemporalKnowledgeGraph", "Embedder"]


@runtime_checkable
class Embedder(Protocol):
    """The embedder contract the graph calls back into.

    ``dimension`` may be an int attribute/property or a zero-arg method. ``embed``
    receives a list of texts and a role (``"document"`` on ingest, ``"query"`` on
    search) and returns one vector per text.
    """

    dimension: int

    def embed(self, texts: Sequence[str], role: str) -> list[list[float]]: ...


def _dumps(properties: Mapping[str, Any] | None) -> str | None:
    """Serialize dict properties to the JSON string the native layer accepts."""
    if properties is None:
        return None
    return json.dumps(dict(properties))


def _parse_record(record: dict | None) -> dict | None:
    """Parse a native record's ``properties`` JSON string back into a dict."""
    if record is None:
        return None
    raw = record.get("properties")
    if isinstance(raw, str):
        try:
            record["properties"] = json.loads(raw) if raw and raw != "null" else {}
        except json.JSONDecodeError:
            record["properties"] = {}
    return record


def _parse_records(records: list[dict]) -> list[dict]:
    return [_parse_record(r) for r in records]


def _as_of(as_of: Any) -> tuple[int | None, bool]:
    """Translate a friendly ``as_of`` into the native ``(as_of_ms, all_time)`` pair.

    ``None`` → current view; an int (epoch ms) → as-of that instant; ``"all"`` /
    ``"history"`` → every version.
    """
    if as_of is None:
        return None, False
    if isinstance(as_of, bool):  # bool is an int subclass — reject before it maps to epoch 0/1
        raise TypeError(f"as_of must be None, an int (epoch ms), or 'all'/'history', not bool: {as_of!r}")
    if isinstance(as_of, str):
        if as_of.lower() in ("all", "history"):
            return None, True
        raise ValueError(f"as_of string must be 'all'/'history', got {as_of!r}")
    return int(as_of), False


def _resolve_dimension(embedder: Any, vector_dim: int | None) -> int:
    if vector_dim is not None:
        return int(vector_dim)
    if embedder is not None and hasattr(embedder, "dimension"):
        dim = embedder.dimension
        return int(dim() if callable(dim) else dim)
    return 384


class TemporalKnowledgeGraph:
    """A bi-temporal knowledge graph backed by native ``lodedb-graph``.

    Example::

        kg = TemporalKnowledgeGraph(embedder=my_embedder)          # in-memory
        kg.upsert_entity("alice", "Person", "Alice, engineer")
        kg.upsert_entity("acme", "Org", "Acme Corp")
        f = kg.add_fact("alice", "works_at", "acme", "Alice works at Acme",
                        valid_at=1000)
        kg.add_fact("alice", "works_at", "globex", "Alice works at Globex",
                    valid_at=2000, invalidates=[f])

        kg.neighbors("alice", relation="works_at")                 # -> Globex (now)
        kg.neighbors("alice", relation="works_at", as_of=1500)     # -> Acme
        kg.history("alice")                                        # both, preserved
    """

    def __init__(
        self,
        path: str | None = None,
        *,
        embedder: Embedder | None = None,
        vector_dim: int | None = None,
        index_facts: bool = True,
    ) -> None:
        from lodedb import _turbovec  # bundled native extension

        self._embedder = embedder
        dim = _resolve_dimension(embedder, vector_dim)
        self._g = _turbovec.graph.TemporalKnowledgeGraph(
            path, dim, embedder, index_facts
        )

    # -- episodes -----------------------------------------------------------

    def add_episode(
        self,
        source: str,
        body: str,
        occurred_at: int,
        *,
        properties: Mapping[str, Any] | None = None,
        mentions: Sequence[str] = (),
    ) -> str:
        """Store a raw observation (no extraction); returns its id."""
        return self._g.add_episode(source, body, occurred_at, _dumps(properties), list(mentions))

    def get_episode(self, id: str) -> dict | None:
        return _parse_record(self._g.get_episode(id))

    # -- entities -----------------------------------------------------------

    def upsert_entity(
        self,
        id: str,
        type: str,
        label: str,
        *,
        properties: Mapping[str, Any] | None = None,
        valid_at: int | None = None,
        invalid_at: int | None = None,
    ) -> str:
        """Create or replace an entity (upsert by id) and (re)index it."""
        return self._g.upsert_entity(id, type, label, _dumps(properties), valid_at, invalid_at)

    def upsert_entity_vec(
        self,
        id: str,
        type: str,
        label: str,
        embedding: Sequence[float],
        *,
        properties: Mapping[str, Any] | None = None,
        valid_at: int | None = None,
        invalid_at: int | None = None,
    ) -> str:
        """Create or replace an entity indexed by a precomputed vector (vector-in)."""
        return self._g.upsert_entity_vec(
            id, type, label, list(embedding), _dumps(properties), valid_at, invalid_at
        )

    def get_entity(self, id: str) -> dict | None:
        return _parse_record(self._g.get_entity(id))

    def entities(self, type: str | None = None, *, as_of: Any = None) -> list[dict]:
        """Complete-set enumeration by type (None = all) in a temporal frame."""
        ms, all_time = _as_of(as_of)
        return _parse_records(self._g.entities(type, ms, all_time))

    def remove_entity(self, id: str) -> bool:
        return self._g.remove_entity(id)

    # -- facts --------------------------------------------------------------

    def add_fact(
        self,
        src: str,
        relation: str,
        dst: str,
        fact: str,
        *,
        properties: Mapping[str, Any] | None = None,
        episodes: Sequence[str] = (),
        valid_at: int | None = None,
        invalidates: Sequence[str] = (),
    ) -> str:
        """Assert a fact (a distinct, uniquely-id'd assertion, so history survives).
        ``invalidates`` closes prior facts using Graphiti's rule. Returns the fact id.
        """
        return self._g.add_fact(
            src, relation, dst, fact, _dumps(properties),
            list(episodes), valid_at, list(invalidates),
        )

    def invalidate_fact(self, id: str, *, invalid_at: int | None = None) -> bool:
        """Close a fact's validity without a replacement (an explicit end)."""
        return self._g.invalidate_fact(id, invalid_at)

    def get_fact(self, id: str) -> dict | None:
        return _parse_record(self._g.get_fact(id))

    def remove_fact(self, id: str) -> bool:
        """Hard-remove a fact (prefer :meth:`invalidate_fact` to keep history)."""
        return self._g.remove_fact(id)

    # -- traversal ----------------------------------------------------------

    def neighbors(
        self,
        id: str,
        *,
        direction: str = "out",
        relation: str | None = None,
        as_of: Any = None,
    ) -> list[dict]:
        """Facts incident to ``id`` (out/in/both, optional relation), as-of a frame."""
        ms, all_time = _as_of(as_of)
        return _parse_records(self._g.neighbors(id, direction, relation, ms, all_time))

    def k_hop(
        self,
        seeds: str | Sequence[str],
        *,
        k: int = 1,
        direction: str = "both",
        as_of: Any = None,
    ) -> dict:
        """Deterministic k-hop neighbourhood around ``seeds``, in a temporal frame."""
        seed_ids = [seeds] if isinstance(seeds, str) else list(seeds)
        ms, all_time = _as_of(as_of)
        return self._parse_subgraph(self._g.k_hop(seed_ids, k, direction, ms, all_time))

    # -- semantic retrieval -------------------------------------------------

    def semantic_entities(
        self,
        query: str | None = None,
        *,
        embedding: Sequence[float] | None = None,
        k: int = 10,
        type: str | None = None,
        as_of: Any = None,
    ) -> list[tuple[float, dict]]:
        """Top-``k`` entities for ``query``/``embedding``, optionally by type, as-of."""
        ms, all_time = _as_of(as_of)
        emb = list(embedding) if embedding is not None else None
        hits = self._g.semantic_entities(query, emb, k, type, ms, all_time)
        return [(score, _parse_record(rec)) for score, rec in hits]

    def semantic_facts(
        self,
        query: str | None = None,
        *,
        embedding: Sequence[float] | None = None,
        k: int = 10,
        relation: str | None = None,
        as_of: Any = None,
    ) -> list[tuple[float, dict]]:
        """Top-``k`` facts for ``query``/``embedding`` (Graphiti's default shape)."""
        ms, all_time = _as_of(as_of)
        emb = list(embedding) if embedding is not None else None
        hits = self._g.semantic_facts(query, emb, k, relation, ms, all_time)
        return [(score, _parse_record(rec)) for score, rec in hits]

    def search_subgraph(
        self,
        query: str | None = None,
        *,
        embedding: Sequence[float] | None = None,
        k: int = 5,
        hops: int = 1,
        direction: str = "both",
        type: str | None = None,
        as_of: Any = None,
    ) -> dict:
        """Semantic seed entities + k-hop expansion (the headline query)."""
        ms, all_time = _as_of(as_of)
        emb = list(embedding) if embedding is not None else None
        return self._parse_subgraph(
            self._g.search_subgraph(query, emb, k, hops, direction, type, ms, all_time)
        )

    def resolve_entity(self, name: str, *, k: int = 5) -> list[tuple[float, dict]]:
        """Candidate entities matching ``name`` for the caller's resolution step."""
        return [(score, _parse_record(rec)) for score, rec in self._g.resolve_entity(name, k)]

    def history(self, entity_id: str) -> list[dict]:
        """Every fact ever touching an entity, all frames (history)."""
        return _parse_records(self._g.history(entity_id))

    # -- maintenance --------------------------------------------------------

    def reindex(self) -> dict:
        """Rebuild the semantic index from the topology truth store."""
        return self._g.reindex()

    def stats(self) -> dict:
        """Entity/fact counts and the index document count."""
        return self._g.stats()

    def persist(self) -> None:
        """Checkpoint the semantic index (the topology store autocommits)."""
        self._g.persist()

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _parse_subgraph(sub: dict) -> dict:
        return {
            "entities": _parse_records(sub.get("entities", [])),
            "facts": _parse_records(sub.get("facts", [])),
            "seeds": sub.get("seeds", []),
        }
