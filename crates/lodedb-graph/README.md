<h1 align="center">lodedb-graph</h1>
<p align="center">a <b>bi-temporal knowledge graph</b> over LodeDB — Graphiti's temporal core, local and on-device 🕰️</p>

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](../../LICENSE)

*Part of [LodeDB](../../README.md), by [Egoist Machines, Inc.](https://egoistmachines.com)*

`lodedb-graph` is a native Rust crate that gives agent memory the one thing a plain
label/property graph lacks: **time**. Every fact records when it became true, when it
stopped being true, and when the system learned each of those. Contradictions
**invalidate** old facts instead of deleting them, so history is preserved — and every
read can be taken "as of" any instant.

It is a faithful, storage-oriented port of the temporal core of
[Graphiti](https://github.com/getzep/graphiti), built to run **embedded and
on-device** — the same engine on a server and on a phone. Because it is native, it
binds to both **Python** and **Swift**, not Python alone.

> **It is not an LLM pipeline.** It stores what it is given and answers temporal
> queries. Entity extraction, entity resolution, temporal extraction, and
> contradiction detection stay with the caller (an LLM layer). `add_fact` is the
> analogue of Graphiti's LLM-free `add_triplet`. The design rationale is in
> [`docs/temporal-graph-design.html`](../../docs/temporal-graph-design.html).

## Architecture

Two stores behind one `TemporalGraph` handle:

- a **topology truth store** — embedded SQLite (bundled; nothing to install):
  episodes, entities, typed facts, provenance, and bi-temporal validity. This is the
  authoritative adjacency.
- a **semantic index** — a private [`lodedb-core`](../lodedb-core) engine (vector +
  BM25) over entity labels and fact text, for hybrid entry-point search. It is a
  *rebuildable derived artifact*: `reindex()` restores it from the truth store, so it
  is safe to throw away.

The bi-temporal logic (as-of resolution, invalidation) and Graphiti's rerankers (RRF,
MMR, node-distance, episode-mentions) are ported as pure functions in
[`src/`](src/), exposed under the [`rerank`](src/search.rs) module for advanced
callers.

## The model: episodes → entities → facts

Three record kinds, mirroring Graphiti's `graphiti_core` (`nodes.py`, `edges.py`):

- **Episode** ← `EpisodicNode` — a raw observation the graph was built from (a note, a
  chat turn, a connector event), with provenance and an `occurred_at`
  (Graphiti's `reference_time`).
- **Entity** ← `EntityNode` — a resolved thing in the world: an id, a caller-defined
  `type`, a `label` (embedded for semantic search), JSON `properties`, and — for
  entities that begin and end — an optional validity interval.
- **Fact** ← `EntityEdge` — a typed, directed, labeled relationship
  `(src) —[relation]→ (dst)` with a natural-language `fact` string and **four
  timestamps**:

| Timestamp | Clock | Meaning |
| --- | --- | --- |
| `valid_at` | event | when the fact became true in the world |
| `invalid_at` | event | when it stopped being true |
| `created_at` | transaction | when the system recorded it |
| `expired_at` | transaction | when the system superseded / retracted it |

A **live** fact has `expired_at` and `invalid_at` both open. Asserting a contradicting
fact closes the prior one — `invalid_at` set to the new fact's `valid_at`, `expired_at`
set to now — atomically, in a single topology transaction, exactly as Graphiti does. So
nothing is lost, and "as-of T" queries fall out of the model.

Timestamps are epoch **milliseconds** (`i64`); `None` encodes an open interval.

## Quickstart (Rust)

```rust
use lodedb_graph::{TemporalGraph, GraphConfig, Direction, AsOf};
use serde_json::json;

// Bring an embedder (implement the `Embedder` trait) for the text-in path,
// or pass `None` and use the `*_vec` verbs with precomputed vectors.
let mut g = TemporalGraph::open_in_memory(GraphConfig::default(), Some(Box::new(my_embedder)))?;

g.upsert_entity("alice",  "Person", "Alice, software engineer", json!({}), None, None)?;
g.upsert_entity("acme",   "Org",    "Acme Corp",   json!({}), None, None)?;
g.upsert_entity("globex", "Org",    "Globex Corp", json!({}), None, None)?;

// Alice works at Acme from t=1000 (epoch ms; use real timestamps in practice).
let f = g.add_fact("alice", "works_at", "acme", "Alice works at Acme",
                   json!({}), vec![], Some(1000), &[])?;
// At t=2000 she moves — this invalidates the Acme fact rather than deleting it.
g.add_fact("alice", "works_at", "globex", "Alice works at Globex",
           json!({}), vec![], Some(2000), &[f])?;

g.neighbors("alice", Direction::Out, Some("works_at"), AsOf::Now)?;      // → Globex  (current)
g.neighbors("alice", Direction::Out, Some("works_at"), AsOf::At(1500))?; // → Acme    (as of t=1500)
g.history("alice")?;                                                     // both facts, preserved
```

## Quickstart (Python)

```python
from lodedb.graph import TemporalKnowledgeGraph

# Any object with `dimension` and `embed(texts, role)`; path=None → in-memory.
kg = TemporalKnowledgeGraph(path="./life", embedder=my_embedder)

kg.upsert_entity("alice", "Person", "Alice, software engineer")
kg.upsert_entity("acme",  "Org",    "Acme Corp")
kg.upsert_entity("globex","Org",    "Globex Corp")

f = kg.add_fact("alice", "works_at", "acme", "Alice works at Acme", valid_at=1000)
kg.add_fact("alice", "works_at", "globex", "Alice works at Globex",
            valid_at=2000, invalidates=[f])

kg.neighbors("alice", relation="works_at")               # → Globex   (current)
kg.neighbors("alice", relation="works_at", as_of=1500)   # → Acme     (as of t=1500)
kg.history("alice")                                       # both facts, preserved

# Semantic entry points + k-hop expansion, time-scoped:
sub = kg.search_subgraph("who does Alice work for?", k=3, hops=1, as_of=1500)
```

## Quickstart (Swift, on-device)

```swift
import LodeDBCore

// Swift embeds fact/label text and passes vectors, so the graph needs no embedder
// over the FFI — the on-device path.
let g = try LodeGraph(path: url, embedder: myEmbedder)

try g.upsertEntity(id: "alice",  type: "Person", label: "Alice, software engineer")
try g.upsertEntity(id: "acme",   type: "Org",    label: "Acme Corp")
try g.upsertEntity(id: "globex", type: "Org",    label: "Globex Corp")

let f = try g.addFact(src: "alice", relation: "works_at", dst: "acme",
                      fact: "Alice works at Acme", validAt: 1000)
try g.addFact(src: "alice", relation: "works_at", dst: "globex",
              fact: "Alice works at Globex", validAt: 2000, invalidates: [f])

try g.neighbors(id: "alice", direction: "out", relation: "works_at", asOf: .now)      // → Globex
try g.neighbors(id: "alice", direction: "out", relation: "works_at", asOf: .at(1500)) // → Acme
```

## As-of queries and history

Every read resolves under a temporal frame:

| Frame | Rust | Python / Swift | Meaning |
| --- | --- | --- | --- |
| current view (live facts) | `AsOf::Now` | `None` / `.now` | Graphiti's default |
| as of an instant `T` | `AsOf::At(T)` | `1500` / `.at(1500)` | facts valid at event-time `T` |
| every version | `AsOf::All` | `"all"` / `.all` | full history, no temporal filter |

`entities(...)`, `semantic_entities(...)`, `semantic_facts(...)`, `neighbors(...)`,
`k_hop(...)`, and `search_subgraph(...)` all take the same frame. `resolve_entity(name)`
returns candidate entities for a caller-driven resolution step (embedding + lexical
match), leaving the merge decision to you.

## Bring your own embedder

`lodedb-core` does not embed — embedding lives in the binding layer — so the graph is
driven by a caller-supplied embedder, mirroring the Swift `LodeEmbedder` contract:

- **Rust** — implement the [`Embedder`](src/model.rs) trait (`dimension()` +
  `embed(texts, role)`), or pass `None` and use the vector-in verbs
  (`upsert_entity_vec`, `add_fact_vec`, and the `embedding=` argument on the
  `semantic_*` calls).
- **Python** — any object with a `dimension` and an `embed(texts, role)` method
  (`role` is `"document"` on ingest, `"query"` on search).
- **Swift** — a `LodeEmbedder`; the binding embeds and passes vectors, so no embedder
  crosses the FFI.

## API surface

`TemporalGraph` mirrors the same surface across all three bindings:

| Area | Verbs |
| --- | --- |
| **Episodes** | `add_episode`, `get_episode` |
| **Entities** | `upsert_entity`(`_vec`), `get_entity`, `entities`, `remove_entity` |
| **Facts** | `add_fact`(`_vec`), `invalidate_fact`, `get_fact`, `remove_fact` |
| **Traversal** | `neighbors`, `k_hop`, `history` |
| **Semantic** | `semantic_entities`, `semantic_facts`, `search_subgraph`, `resolve_entity` |
| **Maintenance** | `reindex`, `stats`, `persist` |

`add_fact` / `invalidate_fact` preserve history; `remove_entity` / `remove_fact` are
hard deletes (Graphiti edge/node deletion) — prefer invalidation when history matters.

## Scope — not in scope, by design

The engine is deliberately the **storage-and-query half** of Graphiti. It has:

- **no** LLM extraction, entity resolution, or contradiction detection — compose those
  around it; the primitives (`add_episode`, `upsert_entity`, `add_fact` /
  `invalidate_fact`, `resolve_entity`, the as-of reads) are exactly the surface such a
  pipeline needs;
- **no** community tier;
- an embedded SQLite + LodeDB target rather than a server graph database, so it runs on
  device.

## Relationship to `KnowledgeGraph`

The Python-only [`lodedb.graph.KnowledgeGraph`](../../docs/graph.md) (SQLite + LodeDB,
non-temporal) is unchanged and remains for label/property graphs that do not need time.
`TemporalKnowledgeGraph` is the superset — temporal, native, and on-device. It does not
replace `KnowledgeGraph`; reach for the temporal one when validity, invalidation, or
history matter.

## Tests

The invariants the crate upholds — invalidation preserving history and as-of reads,
event-axis supersession, k-hop traversal, reindex-rebuilds-from-truth, episode
reference-time inheritance, open-start as-of consistency — are exercised in
[`tests/temporal_graph.rs`](tests/temporal_graph.rs):

```console
cargo test -p lodedb-graph
```

## License

Apache-2.0 © [Egoist Machines, Inc.](https://egoistmachines.com)
