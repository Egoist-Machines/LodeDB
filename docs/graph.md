# Knowledge graphs & graph-backed memory on LodeDB

LodeDB is an exact, embedded vector index — not a graph engine. But a knowledge
graph (and the graph-based agent memory built on Zep/Graphiti, cognee, Letta) does
not need LodeDB *to be* a graph engine; it needs a fast, local, crash-atomic
**semantic-retrieval layer over nodes and edges**, paired with a store built for
traversal. This page covers the three capabilities that make LodeDB a first-class
substrate for that, and the `lodedb.graph` layer that ties them together.

The design principle throughout: **the graph store is the source of truth; LodeDB
is a rebuildable semantic index over it.** That dissolves cross-store atomicity —
if an index write is lost, the topology is still correct and the index rebuilds
from it.

---

## 1. Enumeration — `list_documents` / `get_document`

`search` ranks the top-`k` most similar documents. Traversal needs the opposite:
the **complete** set matching a structural predicate, regardless of similarity —
"every edge whose `src` is X", "every node of `type` Person". That is enumeration,
and it's now on the public API:

```python
from lodedb import LodeDB

db = LodeDB("./data")

# every document, payload-free: {"id", "metadata", "chunk_count", "content_hash"}
for record in db.list_documents():
    ...

# the COMPLETE matching set — no k cap, no query vector, no scoring
out_edges = db.list_documents(filter={"src": "node:alice"})
people     = db.list_documents(filter={"type": "Person", "year": {"$gte": 2020}})

# by-id metadata read (resolve an edge's endpoints without a search)
rec = db.get_document("edge:alice-worksAt-acme")   # or None if absent
```

`filter` takes the same exact-match-or-predicate grammar as `search`
(`$eq/$ne/$gt/$gte/$lt/$lte/$in/$nin/$exists` + `$and/$or/$not`, plus a
`document_ids` allowlist). Records are payload-free — never text or vectors.

> Scale note: filtered enumeration currently matches in-process over enumerated
> records. Pushing it into the engine's posting index (O(matches)) is tracked in
> [`research-prompts/02`](research-prompts/02-engine-side-filtered-enumeration.md).

---

## 2. Vector-in — bring your own embeddings

By default LodeDB embeds text internally. The **vector-in** API lets you store and
query **precomputed** embeddings instead — for reusing an external embedding model,
or for a graph layer that embeds once and reuses the vectors:

```python
# add precomputed vectors (must match the index dim: minilm=384, bge=768)
db.add_vectors(my_vector, id="alice", metadata={"kind": "node", "type": "Person"})
db.add_vectors_many([
    {"vector": v0, "id": "n0", "metadata": {...}},
    {"vector": v1, "id": "n1"},
])

# query by a precomputed query vector (skips internal embedding)
hits = db.search_by_vector(my_query_vector, k=10, filter={"kind": "node"})
batched = db.search_many_by_vector([qv0, qv1, qv2], k=10)
```

- Vectors are **L2-normalized by default** (`normalize=False` to skip) so cosine
  scores stay comparable with the text path and with self-embedded docs.
- A vector-in document stores **no raw text** — `get(id)` returns `None`.
- Vector-in reuses the same atomic-commit + O(changed) persistence + TurboVec scan
  as the text path; it is byte-identical, so `search` and `search_by_vector` over
  the same vectors return identical results (verified: `topk_overlap == 1.0` in the
  benchmark), and ingest/query are much faster when you already have the vectors
  (no embedding step).
- Only mix vectors from the **same** embedding model in one index — mixing models
  makes similarity meaningless.

> Today vector-in requires the index's preset dimension. A bring-your-own-dimension,
> no-embedder index (for 1536-d/3072-d frameworks like Graphiti/cognee) is the
> integration unlock tracked in
> [`research-prompts/03`](research-prompts/03-arbitrary-dim-vector-only-index.md).

---

## 3. `lodedb.graph.KnowledgeGraph`

A hybrid layer: an embedded **SQLite topology sidecar** (nodes, typed edges,
properties — the source of truth, built for traversal) plus **LodeDB as the
semantic index** for entry-point retrieval. Pure Python, stdlib-only sidecar.

```python
from lodedb.graph import KnowledgeGraph

kg = KnowledgeGraph("./kg")                      # ./kg/topology.sqlite3 + ./kg/index

kg.add_node(id="alice", type="Person", label="Alice, software engineer at Acme")
kg.add_node(id="acme",  type="Org",    label="Acme Corp, a robotics company")
kg.add_node(id="nyc",   type="Place",  label="New York City")
kg.add_edge("alice", "works_at", "acme")
kg.add_edge("acme",  "hq_in",    "nyc")

# deterministic traversal over the topology
kg.neighbors("alice", direction="out")           # [Edge(alice -works_at-> acme)]
sub = kg.k_hop("alice", k=2, direction="both")   # 2-hop neighbourhood (nodes + edges)

# semantic entry points (LodeDB search scoped to nodes)
hits = kg.semantic_nodes("who builds robots?", k=3, node_type="Org")

# hybrid retrieval: semantic seeds + structural expansion (the whole point)
sub = kg.search_subgraph("robotics", k=3, hops=1)
for node_id, score in sub.seeds:                 # semantic entry points + scores
    ...
for edge in sub.edges:                            # their 1-hop neighbourhood
    ...
```

Nodes are embedded by their `label`, or by a caller-supplied vector
(`add_node(..., embedding=[...])`, the vector-in path). Pass `index_edges=True` to
also index edge "facts" for `semantic_edges`. `reindex()` rebuilds the LodeDB index
from the SQLite source of truth (using enumeration to drop orphans), making the
index a derived, throwaway artifact.

### Why hybrid, not LodeDB-only

A property graph needs typed-edge traversal, k-hop/path queries, and (often)
temporal validity — primitives a top-`k` vector search can't express. Keeping
topology in SQLite and using LodeDB for the "which nodes/edges are relevant"
step is exactly how Zep/Graphiti and cognee are built; LodeDB is a strong local,
exact, crash-atomic vector half of that pairing, and its O(changed) delta
persistence matches incremental fact accrual in agent memory.

---

## Benchmarks

`benchmarks/graph_memory/` measures all three (metrics-only): vector-in vs text-in
ingest/query throughput + parity, predicate-filter latency by selectivity, and
graph traversal + hybrid retrieval at scale.

```bash
# local (synthetic, no network)
python benchmarks/graph_memory/graph_memory_bench.py --dataset synthetic \
    --max-documents 2000 --graph-nodes 2000

# Modal (GovReport at scale)
modal run benchmarks/graph_memory/modal_bench.py::smoke
modal run benchmarks/graph_memory/modal_bench.py::main_a10
```

---

## Integration roadmap (memory systems)

The strategic framing is to treat knowledge-graph memory as an **integration
target**, not a LodeDB subsystem: LodeDB as the pluggable vector backend inside an
existing graph-memory framework. The prerequisites that gated it are now in place
(public enumeration + by-id read; vector-in), with the remaining unlock — an
arbitrary-dim vector-only index for frameworks that own their embedder — scoped in
[`research-prompts/03`](research-prompts/03-arbitrary-dim-vector-only-index.md).
When the integrations roadmap (`docs/integrations.md`, PR #8) lands, add a row:
*"knowledge-graph memory — LodeDB as the vector backend for Graphiti/cognee;
prereqs satisfied: enumeration, by-id read, vector-in."*

See [`research-prompts/`](research-prompts/) for the full set of optimization
follow-ups surfaced while building this stack.
