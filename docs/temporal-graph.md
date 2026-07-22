# Bi-temporal knowledge graphs on LodeDB (`lodedb-graph`)

`lodedb-graph` is the **bi-temporal successor** to [`lodedb.graph.KnowledgeGraph`](graph.md).
It keeps the same architecture — an authoritative topology store paired with a
rebuildable LodeDB semantic index — and adds the thing agent memory actually needs
from a graph: **time**. Every fact records when it became true, when it stopped being
true, and when the system learned each of those; contradictions **invalidate** old
facts instead of deleting them; and every read can be taken "as of" any instant.

It is a native Rust crate (`crates/lodedb-graph`) over `lodedb-core`, so unlike the
Python-only `KnowledgeGraph` it binds to **both Python and Swift** — the same engine
runs on a server and on a phone.

This is a faithful, storage-oriented port of the temporal core of
[Graphiti](https://github.com/getzep/graphiti). It is **not** an LLM pipeline: it
stores what it is given and answers temporal queries. Entity extraction, entity
resolution, temporal extraction, and contradiction detection stay with the caller
(an LLM layer) — `add_fact` is the analogue of Graphiti's LLM-free `add_triplet`.
The deeper design rationale is in [`temporal-graph-design.html`](temporal-graph-design.html).

## The model: episodes → entities → facts

Three record kinds, mirroring Graphiti's `graphiti_core`:

- **Episode** — a raw observation the graph was built from (a note, a chat turn, a
  connector event), with provenance and an `occurred_at` (Graphiti's `reference_time`).
- **Entity** — a resolved thing in the world: an id, a caller-defined `type`, a
  `label` (embedded for semantic search), JSON `properties`, and — for entities that
  begin and end — an optional validity interval.
- **Fact** — a typed, directed, **labeled** relationship `(src) —[relation]→ (dst)`
  with a natural-language `fact` string and **four timestamps**:

| Timestamp | Clock | Meaning |
| --- | --- | --- |
| `valid_at` | event | when the fact became true in the world |
| `invalid_at` | event | when it stopped being true |
| `created_at` | transaction | when the system recorded it |
| `expired_at` | transaction | when the system superseded / retracted it |

A "live" fact is one with `expired_at` and `invalid_at` both open. Adding a
contradicting fact closes the prior one — its `invalid_at` is set to the new fact's
`valid_at` and its `expired_at` to now — exactly as Graphiti does, so nothing is lost.

## Quickstart (Python)

```python
from lodedb.graph import TemporalKnowledgeGraph

# Bring an embedder: any object with `dimension` and `embed(texts, role)`.
kg = TemporalKnowledgeGraph(path="./life", embedder=my_embedder)   # path=None → in-memory

kg.upsert_entity("alice", "Person", "Alice, software engineer")
kg.upsert_entity("acme",  "Org",    "Acme Corp")
kg.upsert_entity("globex","Org",    "Globex Corp")

# Alice worked at Acme from t=1000 (epoch ms; use real timestamps in practice).
f = kg.add_fact("alice", "works_at", "acme", "Alice works at Acme", valid_at=1000)
# At t=2000 she moves — this invalidates the Acme fact rather than deleting it.
kg.add_fact("alice", "works_at", "globex", "Alice works at Globex",
            valid_at=2000, invalidates=[f])
```

## As-of queries and history

Every read takes an `as_of` frame: `None` (the current view), an epoch-ms instant
(as of that moment), or `"all"` (every version).

```python
kg.neighbors("alice", relation="works_at")               # -> Globex   (current)
kg.neighbors("alice", relation="works_at", as_of=1500)   # -> Acme     (as of 2019)
kg.neighbors("alice", relation="works_at", as_of=2500)   # -> Globex
kg.history("alice")                                       # both facts, preserved

# Semantic entry points + k-hop expansion, time-scoped:
sub = kg.search_subgraph("who does Alice work for?", k=3, hops=1, as_of=1500)
for entity_id, score in sub["seeds"]: ...
for fact in sub["facts"]: ...
```

`entities(type, as_of=...)`, `semantic_entities(...)`, and `semantic_facts(...)` (the
Graphiti-style fact search) all take the same frame. `resolve_entity(name)` returns
candidate entities for a caller-driven resolution step (embedding + lexical match),
leaving the merge decision to you.

## Bring your own embedder

`lodedb-core` does not embed — embedding lives in the binding layer — so the graph is
driven by a caller-supplied embedder, mirroring the Swift `LodeEmbedder` contract. In
Python it is any object with a `dimension` and an `embed(texts, role)` method (`role`
is `"document"` on ingest, `"query"` on search). Callers who already hold vectors can
skip it and use the vector-in verbs (`upsert_entity_vec`, and the `embedding=`
argument on the `semantic_*` calls).

## The Rust crate

The engine is `crates/lodedb-graph`, a small crate over `lodedb-core`:

- an embedded SQLite **topology truth store** (episodes, entities, typed facts,
  provenance, bi-temporal validity) — the authoritative adjacency;
- a `lodedb-core` **semantic index** driven as a rebuildable derived artifact
  (`reindex()` restores it from the truth store);
- the bi-temporal logic (as-of resolution, invalidation) and the Graphiti rerankers
  (RRF, MMR, node-distance, episode-mentions) ported as pure functions.

The public `TemporalGraph` type mirrors this Python surface. See the crate's tests
(`crates/lodedb-graph/tests/temporal_graph.rs`) for the invariants it upholds.

## Relationship to `KnowledgeGraph`

`KnowledgeGraph` (Python, SQLite + LodeDB, non-temporal) is unchanged and remains for
label/property graphs that do not need time. `TemporalKnowledgeGraph` is the superset:
temporal, native, and on-device. It does not replace `KnowledgeGraph`; pick the
temporal one when validity, invalidation, or history matter.

## Not in scope (by design)

The engine is deliberately the storage-and-query half of Graphiti. It has **no** LLM
extraction, entity resolution, or contradiction detection; **no** community tier; and
it targets embedded SQLite + LodeDB rather than a server graph database, so it runs on
device. Compose the LLM steps around it — the primitives (`add_episode`,
`upsert_entity`, `add_fact`/`invalidate_fact`, `resolve_entity`, the as-of reads) are
exactly the surface such a pipeline needs.
