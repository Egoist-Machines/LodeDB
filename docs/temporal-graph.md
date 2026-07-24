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
kg = TemporalKnowledgeGraph(path="./kg", embedder=my_embedder)     # path=None → in-memory

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

Reads support two time axes. Event time answers when a fact was true in the
world. Transaction time answers what the database knew at a given time.

| Python `as_of` | Rust | Meaning |
| --- | --- | --- |
| `None` | `AsOf::Now` | Graphiti-compatible current view |
| `"now_valid"` | `AsOf::NowValid(now_ms)` | Strict validity on both clocks |
| `event_ms` | `AsOf::At(event_ms)` | Event-time travel only |
| `(event_ms, known_ms)` | `AsOf::AtKnown { ... }` | Event and knowledge time |
| `"all"` | `AsOf::All` | Every stored version |

The compatibility current view intentionally follows Graphiti's open-ended
semantics: it requires `expired_at IS NULL AND invalid_at IS NULL`, but does not
check `valid_at <= now`. A future-dated fact can therefore appear under
`as_of=None`. Use `as_of="now_valid"` for "what is true now" reads. Graphiti
likewise exposes temporal fields as optional search filters and leaves them unset
by default in
[`SearchFilters`](https://github.com/getzep/graphiti/blob/main/graphiti_core/search/search_filters.py).

```python
kg.neighbors("alice", relation="works_at")               # -> Globex   (current)
kg.neighbors("alice", relation="works_at",
             as_of="now_valid")                          # strict current validity
kg.neighbors("alice", relation="works_at", as_of=1500)   # -> Acme     (as of t=1500)
kg.neighbors("alice", relation="works_at", as_of=2500)   # -> Globex
kg.neighbors("alice", relation="works_at",
             as_of=(2500, learned_at))                   # what was known then
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

Swift exposes the same frames as `.now`, `.nowValid`, `.at(eventMs)`,
`.atKnown(validAt: eventMs, knownAt: learnedMs)`, and `.all`.

## Stable ingest and provenance rollback

Supply a stable id when an upstream event or extraction job may retry:

```python
episode_id = kg.add_episode(
    "connector",
    payload,
    occurred_at,
    id="mailbox/message-42",
)
fact_id = kg.add_fact(
    "alice",
    "works_at",
    "acme",
    "Alice works at Acme",
    episodes=[episode_id],
    id="mailbox/message-42#fact-1",
)
```

Repeating an id with the same payload returns the original id without creating a
second derivation. Reusing it with a different payload raises an error.
`episodes()` enumerates source records, `facts_by_episode(id)` exposes provenance,
and `remove_episode(id)` rolls back facts originated by that episode. If an episode
only added support to a fact originated elsewhere, removal detaches that support
and keeps the fact.

## Authorization predicates

Semantic predicates run in the index before candidate generation and ranking.
They are not post-filters. The same predicate is applied before every subgraph
expansion frontier.

```python
scope = {"owner_id": {"$in": authorized_owner_ids}}

entities = kg.semantic_entities("billing contact", predicate=scope)
facts = kg.semantic_facts("approved access", predicate=scope)
resolved = kg.resolve_entity("Alice", predicate=scope)
subgraph = kg.search_subgraph(
    "approved access",
    predicate=scope,
    seed_kind="fact",
    relation="can_access",
)
```

Predicates use LodeDB's metadata grammar (`$eq`, `$ne`, `$in`, `$nin`, ordered
comparisons, `$exists`, `$and`, `$or`, and `$not`) over top-level scalar
properties. Put the authorization fields on both entities and facts when a
protected query expands through topology. Swift provides `GraphPropertyPredicate`
for equality, set membership, existence, and logical composition.

## Per-property entity lineage

Entity snapshots remain convenient to read, but each changed top-level property
now has its own version history. A version records its value, event-time bounds,
transaction-time bounds, and an optional source episode.

```python
kg.upsert_entity(
    "device-7",
    "Device",
    "Device 7",
    properties={"status": "active", "region": "us-west"},
    property_sources={"status": episode_id},
)
status_history = kg.entity_property_history("device-7", "status")
```

Unchanged properties do not create new versions. Removing a property closes its
current version without erasing prior values. Swift exposes the same records
through `entityPropertyHistory(_:key:)`.

## Rerankers and fact seeds

The native Graphiti-compatible rerankers are exported as `rrf`,
`maximal_marginal_relevance`, `node_distance_reranker`, and
`episode_mentions_reranker`. `search_subgraph(..., seed_kind="entity")` preserves
the original behavior. Use `"fact"` to start from semantic facts, or `"both"` to
use both entry-point types.

## Bring your own embedder

`lodedb-core` does not embed — embedding lives in the binding layer — so the graph is
driven by a caller-supplied embedder, mirroring the Swift `LodeEmbedder` contract. In
Python it is any object with a `dimension` and an `embed(texts, role)` method (`role`
is `"document"` on ingest, `"query"` on search). Callers who already hold vectors can
skip it and use the vector-in verbs (`upsert_entity_vec`, `add_fact_vec`, and the
`embedding=` argument on the `semantic_*` calls).

Both write paths validate at the boundary: fact endpoints must be existing
entities, provenance must reference existing episodes, and every id named in
`invalidates` must close, or the whole call fails and nothing persists. Pass
`index_text=False` at open to keep the semantic index vector-only (no label/fact
text retained on the index side; lexical hybrid search degrades to pure vector).
The topology store still holds the text you pass in — that store is the graph's
data.

## The Rust crate

The engine is `crates/lodedb-graph`, a small crate over `lodedb-core`:

- an embedded SQLite **topology truth store** (episodes, entities, typed facts,
  provenance, bi-temporal validity, and caller-supplied vectors) — the
  authoritative adjacency;
- a `lodedb-core` **semantic index** driven as a rebuildable derived artifact.
  `reindex()` restores both text-in records (through the embedder) and vector-in
  records (from vectors retained in the topology store);
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
