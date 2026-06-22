# Findings: graph bulk-load and batched topology reads

Answers [`../research-prompts/06-graph-bulk-load-and-batched-reads.md`](../research-prompts/06-graph-bulk-load-and-batched-reads.md).
Code-grounded, measured on this branch (`feat/graph-knowledge-memory`), prior art cited.
Design input — not a committed decision.

---

## 1. TL;DR

The `KnowledgeGraph` layer is fast where it batches (edge build is pure SQLite,
~10.7k edges/s) and slow where it loops one round-trip per entity:

- **Node build pays one LodeDB commit per node.** `add_node` →
  `_index_node` → `add_vectors`/`add` ([`knowledge_graph.py:433`](../../src/lodedb/graph/knowledge_graph.py#L433),
  [`:435`](../../src/lodedb/graph/knowledge_graph.py#L435)), and each
  `add_vectors`/`add` commits one index generation atomically before returning
  ([`db.py:322`](../../src/lodedb/local/db.py#L322), engine
  [`core.py:3062`](../../src/lodedb/engine/core.py#L3062)). A bulk load of *N*
  nodes therefore does *N* commits. Modal A10 measured **122 nodes/s**.
- **`k_hop` materializes the visited set with one `get_node` per node**
  ([`knowledge_graph.py:249-252`](../../src/lodedb/graph/knowledge_graph.py#L249)),
  i.e. O(frontier) SQLite round-trips. On the Modal A10 hybrid run the seed
  expansion reached ~7.8k nodes and `search_subgraph` p50 was **183 ms**,
  dominated by that loop.

Both are O(N)-round-trip patterns. The fix is the same one the layer already
uses for edges:

1. **Bulk ingest** — add `add_nodes(list)` / `add_edges(list)` (and an
   `ingest()` context manager for the streaming case) that group the SQLite
   side into one `executemany` transaction and the LodeDB side into **one**
   `add_many` / `add_vectors_many` — a single atomic commit for the batch.
2. **Batched reads** — add `TopologyStore.get_nodes(ids)` (one chunked
   `SELECT … WHERE id IN (…)`, mirroring `edges_for`
   [`_store.py:205-249`](../../src/lodedb/graph/_store.py#L205)) and have
   `k_hop`/`search_subgraph` materialize the visited set in one batched read.
3. **Frontier budget** — an optional `max_nodes` cap on `k_hop` that stops BFS
   when the visited set would exceed the budget and `log()`s the truncation (no
   silent caps).

**Measured on this machine** (CPU, `HashEmbeddingBackend(native_dim=384)`,
bounded ≤10k nodes; PoC `/tmp/poc_graph_bulk.py`):

| operation | current (per-call) | batched | speedup |
|-----------|-------------------:|--------:|--------:|
| node build, 2 000 nodes | 317 nodes/s | 6 283 nodes/s | **19.8×** |
| node build, 8 000 nodes | 253 nodes/s | 7 881 nodes/s | **31.1×** |
| frontier read, 964 unique | 5.07 ms | 2.48 ms (`IN`) | 2.0× |
| frontier read, 5 417 unique | 26.8 ms | 14.2 ms (`IN`) | 1.9× |

The batched build rate (~6.3–7.9k nodes/s) lands on the in-repo vector-in
ceiling (`add_vectors_many` measured **6 609 docs/s** on A10,
[`results_a10.json`](../../benchmarks/graph_memory/results/results_a10.json))
and the edge-build rate (10.7k/s) — exactly the order-of-magnitude the success
criteria targets. Batched reads return a **byte-identical** node set to the
per-node loop (`match: true` at both frontier sizes). All three changes are
additive and preserve the public `Node`/`Edge`/`Subgraph` shapes and the
atomic-commit + SQLite-first invariants.

---

## 2. Current behavior (code-grounded + measured)

### 2.1 Node build is one commit per node

`add_node` writes SQLite first, then the index:

```
add_node            knowledge_graph.py:135-136
  self._store.upsert_node(node)        # SQLite (autocommit txn)
  self._index_node(node, embedding=…)  # LodeDB
```

`_index_node` routes to `add_vectors` (vector-in) or `add` (text-in):

```
_index_node         knowledge_graph.py:432-435
  if embedding is not None:
      self._db.add_vectors(embedding, id=doc_id, metadata=metadata)
  elif node.label.strip():
      self._db.add(node.label, id=doc_id, metadata=metadata)
```

Both single-doc verbs commit atomically per call — `add_vectors`'s docstring is
explicit: *"The mutation commits atomically before returning"*
([`db.py:322`](../../src/lodedb/local/db.py#L322)); `add` likewise *"commits
this mutation atomically before returning"* ([`db.py:255`](../../src/lodedb/local/db.py#L255)).
Each lands one index generation through `upsert_vectors_batch` /
`upsert_batch` → the atomic commit route ([`core.py:3062`](../../src/lodedb/engine/core.py#L3062),
which writes the `.txd` journal then swaps the root `<key>.commit.json`). So a
bulk load of *N* nodes does *N* full commit cycles. This is directly observable:
running the per-call PoC emitted a `turbovec_build`/`turbovec_update` +
`snapshot_persist` telemetry cycle **per node** (3.6 MB of per-commit logs for
the per-call leg vs. one cycle for the batched leg).

Edge build, by contrast, is pure SQLite (`upsert_edge`, [`_store.py:153-162`](../../src/lodedb/graph/_store.py#L153))
and only touches LodeDB when `index_edges=True`
([`knowledge_graph.py:168-169`](../../src/lodedb/graph/knowledge_graph.py#L168)).
That is why the same Modal run measured edge build at **10 718 edges/s** —
~88× the node rate — for structurally identical work minus the per-entity
commit.

> **The benchmark already isolates this.** `run_graph_bench` precomputes node
> vectors and calls `kg.add_node(embedding=…)` per node
> ([`graph_memory_bench.py:299-306`](../../benchmarks/graph_memory/graph_memory_bench.py#L299)).
> The embedding cost is *already removed* from the 122 nodes/s — it is almost
> pure per-commit overhead. The same file's `vector_in` sub-benchmark stores the
> same kind of precomputed vectors via **one** `add_vectors_many`
> ([`:161-168`](../../benchmarks/graph_memory/graph_memory_bench.py#L161)) and
> measured **6 609 docs/s** on the same A10. The gap between those two numbers in
> one results file *is* the prize: 6 609 / 122 ≈ **54×** of headroom that is
> purely the per-call-vs-batched commit difference.

### 2.2 Traversal does one `get_node` per visited node

`k_hop` already batches the *edge* expansion through `edges_for` (chunked `IN`),
but then materializes nodes one at a time:

```
k_hop               knowledge_graph.py:248-253
  nodes: dict[str, Node] = {}
  for node_id in visited:
      node = self._store.get_node(node_id)   # one SELECT … WHERE id = ? per node
      if node is not None:
          nodes[node_id] = node
```

`get_node` is a single-row `SELECT … WHERE id = ?` ([`_store.py:107-113`](../../src/lodedb/graph/_store.py#L107)).
`search_subgraph` calls `k_hop` after the semantic seed step
([`knowledge_graph.py:340`](../../src/lodedb/graph/knowledge_graph.py#L340)), so
it inherits the same loop over the *expanded* frontier. On the Modal A10 hybrid
run that frontier averaged **7 838 nodes** and `search_subgraph` p50/p95 were
**182.6 ms / 451.6 ms** (mean 234.3 ms); the cheaper pure-`k_hop` traversal
averaged 1 020 nodes at p50 **18.4 ms**
([`results_a10.json`](../../benchmarks/graph_memory/results/results_a10.json)).
The node-materialization loop scales with frontier size and is the dominant term
in the hybrid number.

`semantic_nodes` adds a *second* per-hit `get_node` over the *k* seeds
([`knowledge_graph.py:284-286`](../../src/lodedb/graph/knowledge_graph.py#L284)) —
small at `k≤10`, but it batches for free under the same `get_nodes` and pairs
with the search-hydration work in
[`05-batched-metadata-hydration`](05-batched-metadata-hydration.md).

### 2.3 The batched-read primitive already exists for edges

`edges_for` ([`_store.py:205-249`](../../src/lodedb/graph/_store.py#L205)) is the
template: it chunks the id list at `_IN_CHUNK = 400`
([`_store.py:47`](../../src/lodedb/graph/_store.py#L47)), builds one
`SELECT … WHERE … IN (placeholders)` per chunk, and dedups by id across chunks.
`get_nodes` is the same shape with a simpler predicate (`id IN (…)`, one bind per
id, so the chunk size is comfortable). The engine even has the batched-read
precedent on the LodeDB side: `get_document_texts` takes a tuple of ids, returns
a `{id: text}` map, and *omits* missing ids rather than failing
([`core.py:1922-1956`](../../src/lodedb/engine/core.py#L1922); SDK
`get_texts`, [`db.py:574`](../../src/lodedb/local/db.py#L574)). `get_nodes`
should follow that contract: missing ids omitted, never raise.

### 2.4 My own before/after measurements

PoC (`/tmp/poc_graph_bulk.py`, CPU, `HashEmbeddingBackend(native_dim=384)`,
`TemporaryDirectory` under `/tmp`, median of repeats for reads):

**Build — per-call `add_node` vs. SQLite `executemany` + one `add_vectors_many`:**

| nodes | per-call ms | per-call nodes/s | batched ms | batched nodes/s | speedup |
|------:|------------:|-----------------:|-----------:|----------------:|--------:|
| 2 000 | 6 304 | 317 | 318 | 6 283 | **19.8×** |
| 8 000 | 31 562 | 253 | 1 015 | 7 881 | **31.1×** |

The speedup *grows* with N (per-call throughput sags as the index grows — each
commit rewrites/extends generation state — while the single batched commit
amortizes). The absolute batched rate (~6.3–7.9k nodes/s) matches the in-repo
A10 `add_vectors_many` ceiling (6 609 docs/s), confirming the batched path is
LodeDB-ingest-bound, not graph-layer-bound.

> The per-call rate here (253–317/s) is *higher* than Modal's 122/s because this
> is CPU + hash backend on a warm local disk; the A10 leg additionally paid
> CUDA-side per-commit overhead. The portable, machine-independent result is the
> **ratio** (≈20–31× locally), which projects the A10 node build from 122 nodes/s
> toward the 6 600 nodes/s `add_vectors_many` ceiling already measured on that
> same host.

**Reads — per-node `get_node` loop vs. one chunked `IN` query:**

| frontier (unique) | per-node loop p50 | chunked `IN` p50 | speedup | identical set? |
|------------------:|------------------:|-----------------:|--------:|:--------------:|
| 964 | 5.07 ms | 2.48 ms | 2.0× | yes |
| 5 417 | 26.8 ms | 14.2 ms | 1.9× | yes |

The in-process read win is modest (~2×) because a local SQLite point-lookup is
cheap and Python loop overhead dominates both legs. The structural change is
what matters at scale: **O(frontier) prepared statements → O(frontier / 400)**.
On the Modal A10 host, where the 7 838-node frontier drove the 183 ms hybrid
p50, collapsing ~7 838 individual `get_node` calls into ~20 `IN` queries removes
the term that the latency is made of; the local 2× is a floor, not the ceiling
of the production gain. (Result-set parity is exact — see `match: true`.)

---

## 3. Prior art (cited)

The "batch the writes into one transaction, batch the reads into one `IN`" shape
is the standard answer across both graph and vector systems.

- **Neo4j — batch updates, don't transact per node.** Idiomatic bulk write is
  `UNWIND $rows AS row … ` so one transaction applies thousands of mutations;
  Neo4j recommends **10 000–100 000 updates per transaction**, and a query run as
  one-transaction-per-statement degrades badly at scale (cited example: ~15 ms
  early vs. ~6 000 ms late without batching). For first-time bulk load the
  offline `neo4j-admin import` tool is preferred over transactional inserts.
  ([batched updates / `UNWIND`](https://medium.com/neo4j/5-tips-tricks-for-fast-batched-updates-of-graph-structures-with-neo4j-and-cypher-73c7f693c8cc),
  [driver performance manual](https://neo4j.com/docs/python-manual/current/performance/),
  [bulk import](https://neo4j.com/blog/cypher-and-gql/bulk-data-import-neo4j-3-0/))
- **Kùzu — `COPY FROM` ≫ per-statement CREATE/MERGE.** Kùzu's docs state
  `COPY FROM` is the *fastest* way to bulk-insert and explicitly scope
  `CREATE`/`MERGE` to "small additions or updates on a sporadic basis," steering
  millions-of-nodes loads to the bulk path. This is precisely the `add_node`
  (sporadic) vs. `add_nodes`/`ingest()` (bulk) split proposed here.
  ([Kùzu import docs](https://docs.kuzudb.com/import/))
- **Qdrant — every upsert is a transaction; batch the points.** Qdrant's guidance:
  *"every individual upsert call initiates a transaction that consumes memory and
  disk I/O … at scale this naive approach can overwhelm your system."* Recommended
  batch ~100–1 000 points/request; `upload_collection`/`upload_points` chunk large
  lists internally. This is the vector-DB mirror of the per-`add_vectors`-commit
  cost LodeDB has, and validates `add_vectors_many`-as-one-commit as the fix.
  ([Qdrant points / batching](https://qdrant.tech/documentation/manage-data/points/),
  [large-scale ingestion](https://qdrant.tech/course/essentials/day-4/large-scale-ingestion/))
- **NetworkX — bulk constructors are first-class.** The in-memory analogue ships
  `add_nodes_from` / `add_edges_from` alongside `add_node`/`add_edge`, so callers
  building a graph from a container use the bulk verb by default — the API-shape
  precedent for `add_nodes`/`add_edges`.
  ([add_nodes_from](https://networkx.org/documentation/stable/reference/classes/generated/networkx.Graph.add_nodes_from.html),
  [add_edges_from](https://networkx.org/documentation/stable/reference/classes/generated/networkx.Graph.add_edges_from.html))
- **SQLite — one transaction = one fsync.** In autocommit mode every write is its
  own transaction (one fsync each); wrapping many inserts in a single transaction
  pays one fsync for the batch, and `executemany` prepares the statement once and
  binds per row — "the fastest approach." This is exactly the SQLite half of
  `add_nodes`/`add_edges`.
  ([Python sqlite3 perf tips](https://remusao.github.io/posts/few-tips-sqlite-perf.html),
  [sqlite3 docs](https://docs.python.org/3/library/sqlite3.html))

**Synthesis.** Both halves of LodeDB's graph layer (the SQLite topology and the
LodeDB index) independently converge on the same rule the rest of the ecosystem
follows: *batch writes into one commit/transaction; fetch a set with one `IN`/
batched read, not a loop.* The design below is the minimal application of that
rule, reusing primitives that already exist in-repo (`add_vectors_many`,
`executemany`, the `edges_for` chunked-`IN` pattern, `get_document_texts`'s
omit-missing contract).

---

## 4. Recommended design

Three additive changes. No change to `Node`/`Edge`/`Subgraph`, no engine change.

### 4.1 `TopologyStore.get_nodes(ids)` — batched read (foundation)

Mirror `edges_for` exactly; simpler predicate.

```python
# src/lodedb/graph/_store.py
def get_nodes(self, node_ids: Iterable[str]) -> dict[str, Node]:
    """Returns {id: Node} for the ids that exist (missing ids omitted).

    The batched counterpart to get_node, backing k-hop/search_subgraph
    materialization. Chunks the IN-list like edges_for so a large frontier
    never exceeds SQLite's bound-parameter limit.
    """
    ids = [str(v) for v in node_ids]
    if not ids:
        return {}
    found: dict[str, Node] = {}
    for start in range(0, len(ids), _IN_CHUNK):
        batch = ids[start : start + _IN_CHUNK]
        placeholders = ",".join("?" for _ in batch)
        for row in self._conn.execute(
            f"SELECT id, type, label, properties FROM nodes WHERE id IN ({placeholders})",
            batch,
        ).fetchall():
            node = _node_from_row(row)
            found[node.id] = node
    return found
```

Contract: returns found-only, never raises on a missing id (matches
`get_document_texts`, [`core.py:1922`](../../src/lodedb/engine/core.py#L1922)).
`get_node` stays as the single-id convenience (`get_nodes([id]).get(id)` would
work but the dedicated single-row query is marginally cheaper for the hot
single-id path).

Then `k_hop` replaces its loop ([`knowledge_graph.py:248-252`](../../src/lodedb/graph/knowledge_graph.py#L248)):

```python
nodes = self._store.get_nodes(visited)   # one batched read, was len(visited) round-trips
return Subgraph(nodes=nodes, edges=list(edges_seen.values()))
```

`semantic_nodes` ([`knowledge_graph.py:281-287`](../../src/lodedb/graph/knowledge_graph.py#L281))
similarly collects the hit ids and does one `get_nodes`, preserving score order
by zipping after the fetch. `Subgraph` is returned unchanged.

### 4.2 `add_nodes` / `add_edges` — bulk ingest (one commit)

Bulk methods that keep **SQLite-first, then the rebuildable index** and collapse
the LodeDB side to one commit.

```python
# src/lodedb/graph/knowledge_graph.py
def add_nodes(self, nodes: Sequence[Mapping[str, Any]]) -> list[str]:
    """Adds/replaces many nodes in one SQLite transaction + one index commit.

    Each item: {"id"?, "type"?, "label"?, "properties"?, "embedding"?}.
    Source-of-truth order: all nodes land in SQLite (one executemany txn),
    then a single add_many / add_vectors_many indexes them (one atomic commit).
    A crash between the two is recoverable via reindex() — identical to the
    per-call invariant, just once per batch instead of once per node.
    """
```

Implementation outline (no new engine surface):

1. Build `Node` objects; assign ids (`secrets.token_hex` for missing, as today
   [`knowledge_graph.py:128`](../../src/lodedb/graph/knowledge_graph.py#L128)).
2. **SQLite first:** one `self._store.upsert_nodes(nodes)` doing
   `with self._conn: self._conn.executemany(<the existing upsert SQL>, rows)` —
   the same `INSERT … ON CONFLICT DO UPDATE` already in `upsert_node`
   ([`_store.py:100-105`](../../src/lodedb/graph/_store.py#L100)), one
   transaction.
3. **Then the index, partitioned by path** to preserve the existing per-node
   routing ([`_index_node`](../../src/lodedb/graph/knowledge_graph.py#L427)):
   - items with an `embedding` → one `self._db.add_vectors_many([...])`;
   - items with a non-empty `label` and no embedding → one `self._db.add_many([...])`;
   - items with neither → `remove` (clear any stale doc), as today.
   That is **at most two commits** (one vector batch, one text batch). If a batch
   mixes both kinds and a strict single-commit is desired, prefer requiring a
   homogeneous batch, or document the two-commit case (both are still
   all-or-nothing per call; see §5). For the common bulk-load case (all-embedding
   or all-label) it is exactly one commit.

`add_edges(list)` is simpler — one `executemany` on the edges table
(`upsert_edges`), and, only when `index_edges`, one `add_many`/`add_vectors_many`
for edges carrying a `fact`/`embedding`. Edge ids derive from the triple as today
([`knowledge_graph.py:159`](../../src/lodedb/graph/knowledge_graph.py#L159)).

**`ingest()` context manager (recommended companion, for the streaming case).**
When the caller can't pre-materialize the whole list (e.g. fact accrual from a
stream), a context manager buffers and flushes once on exit:

```python
with kg.ingest() as batch:           # buffers; defers the index commit
    for fact in stream:
        batch.add_node(id=…, label=…, embedding=…)
        batch.add_edge(src, rel, dst)
# __exit__ -> one executemany txn (nodes, then edges), then one add_*_many commit
```

`ingest()` is sugar over `add_nodes`/`add_edges` (accumulate, then call them
once). Recommendation: ship **`add_nodes`/`add_edges` as the primitive** (matches
NetworkX `*_from` and is the unit the benchmark needs) and **`ingest()` as the
ergonomic streaming wrapper**. Both compile to the same one-transaction +
one-commit flush, so there is no second code path to keep correct.

### 4.3 `max_nodes` frontier budget on `k_hop` (no silent caps)

A hub node or dense graph can expand a 2-hop BFS to most of the graph (the Modal
run: avg 7 838 nodes from `k≤10` seeds). Add an **opt-in** budget:

```python
def k_hop(self, seeds, *, k=1, direction="both", relation=None,
          max_nodes: int | None = None) -> Subgraph:
```

Semantics: BFS as today; before adding a hop's new endpoints, if
`len(visited) + len(new) > max_nodes`, admit nodes up to the budget, stop
expanding, and **log** the truncation — never silently cap:

```python
import logging
_log = logging.getLogger(__name__)
...
if max_nodes is not None and len(visited) >= max_nodes:
    _log.warning("k_hop frontier truncated at max_nodes=%d (hop %d, seeds=%d)",
                 max_nodes, _hop, len(seed_ids))
    break
```

`max_nodes=None` (default) = today's unbounded behavior, so existing callers are
unchanged. `search_subgraph` forwards an optional `max_nodes` to `k_hop`. This
uses stdlib `logging` — the graph layer imports none today, and `logging` matches
how the engine reports progress (`_log_large_ingest_progress`,
[`core.py:1304`](../../src/lodedb/engine/core.py#L1304)). The `Subgraph` shape is
unchanged; a truncated result is simply a smaller valid neighbourhood.

### 4.4 Public surface (unchanged shapes)

`lodedb.graph` continues to export `Edge`, `Node`, `KnowledgeGraph`, `Subgraph`
([`__init__.py`](../../src/lodedb/graph/__init__.py)). New methods are additive:
`KnowledgeGraph.add_nodes`, `.add_edges`, `.ingest`; `TopologyStore.get_nodes`,
`.upsert_nodes`, `.upsert_edges`; new keyword `max_nodes` on `k_hop` /
`search_subgraph`. No field is added to `Node`/`Edge`/`Subgraph`.

---

## 5. Tradeoffs / risks / invariant compliance

**Atomic-commit invariant — preserved.** `add_vectors_many` / `add_many`
validate the whole batch up front and apply it as one atomic commit (vector
path: *"a bad vector fails atomically with nothing applied … the same direct
TurboVec sync + atomic commit the text path uses,"*
[`core.py:1331-1335`](../../src/lodedb/engine/core.py#L1331); commit route
[`core.py:3062`](../../src/lodedb/engine/core.py#L3062)). So a bulk index write is
all-or-nothing, exactly like a single `add`. The SQLite side is one explicit
`with self._conn:` transaction (`executemany`), also all-or-nothing.

**Source-of-truth ordering — preserved.** `add_nodes`/`add_edges` keep
SQLite-write-then-index, the same order as `add_node`
([`knowledge_graph.py:135-136`](../../src/lodedb/graph/knowledge_graph.py#L135)).
A crash between the SQLite commit and the index commit leaves topology correct
and the index rebuildable via `reindex()`
([`knowledge_graph.py:354`](../../src/lodedb/graph/knowledge_graph.py#L354)) —
the crash window is *narrower* than the per-call path (one window per batch vs.
one per node).

**The one genuine tradeoff: mixed-kind batches.** A batch mixing embedding-nodes
and label-nodes needs two index commits (one `add_vectors_many`, one `add_many`).
Each is independently atomic, but the *batch* is then two commits, not one. Three
mitigations, in order of preference: (a) for bulk load, batches are typically
homogeneous (the benchmark is all-embedding), so this is one commit in practice;
(b) document that a mixed batch commits the vector partition then the text
partition (still recoverable — both reach SQLite first); (c) if strict
single-commit-per-`add_nodes` is required, validate the batch is homogeneous and
raise otherwise. **Recommend (a)+(b)**: don't over-constrain the API; the
recoverability invariant holds regardless because SQLite is written first.

**Read parity — verified.** The batched `get_nodes` returns a node set
byte-identical to the per-node loop (`match: true` at 964 and 5 417 unique ids in
the PoC). `Subgraph.nodes` is a dict, so ordering is irrelevant; edges are
already batched (`edges_for`) and unchanged.

**`max_nodes` changes results when set.** A truncated traversal returns fewer
nodes than an unbounded one — by design, and only when the caller opts in. The
mandatory `log()` means truncation is never silent. Default `None` is fully
backward-compatible.

**Memory.** `add_nodes` materializes the batch in memory before the flush (same
as `add_many` today) and `get_nodes` materializes the frontier dict (the loop
already did). For unbounded streams, `ingest()` could flush in sub-batches (e.g.
every 10 000, matching the Neo4j/Qdrant guidance) rather than buffering all —
worth a `flush_every` knob if huge ingests appear, but out of scope for the
order-of-magnitude win.

**Concurrency.** Unchanged. The bulk path holds the same single-writer LodeDB
lock for one commit instead of N, and the SQLite WAL serializes the one
`executemany` txn ([`_store.py:82-84`](../../src/lodedb/graph/_store.py#L82)). One
batch commit *reduces* lock churn vs. N per-node commits.

---

## 6. Prototype / validation plan

**PoC already run** (`/tmp/poc_graph_bulk.py`, this branch, CPU,
`HashEmbeddingBackend(native_dim=384)`, bounded ≤10k nodes — scratch only, no
repo files touched):

- Build: per-call `add_node` 253–317 nodes/s vs. batched (`executemany` + one
  `add_vectors_many`) 6 283–7 881 nodes/s → **19.8–31.1×**, batched rate at the
  in-repo `add_vectors_many` ceiling.
- Reads: per-node `get_node` loop vs. chunked `IN` → **1.9–2.0×** in-process,
  with **identical** result sets, and an O(frontier)→O(frontier/400) statement-count
  reduction that is the real lever on the Modal frontier scale.

**Implementation PoC (next step, behind the hard constraints of a real change):**

1. Add `TopologyStore.get_nodes` + `upsert_nodes`/`upsert_edges`; rewire `k_hop`
   and `semantic_nodes` to one `get_nodes`. Lowest-risk, highest-leverage on the
   183 ms hybrid number; pairs with [`05`](05-batched-metadata-hydration.md).
2. Add `KnowledgeGraph.add_nodes`/`add_edges`; add `ingest()` as the streaming
   wrapper over them.
3. Add the `max_nodes` budget + `log()` to `k_hop`/`search_subgraph`.
4. Tests (mirror [`tests/test_graph_knowledge_graph.py`](../../tests/test_graph_knowledge_graph.py)):
   bulk == per-call equivalence (same nodes/edges/index docs), `get_nodes` ==
   loop parity (incl. missing-id omission), one-commit assertion (count index
   generations or `stats()` deltas), `reindex()` recovery after a simulated
   crash between the SQLite and index writes, `max_nodes` truncation emits a log
   and returns a valid sub-neighbourhood.

**Expected `benchmarks/graph_memory/` (graph sub-bench) delta on Modal A10.**
The benchmark loop ([`graph_memory_bench.py:299-306`](../../benchmarks/graph_memory/graph_memory_bench.py#L299))
would switch from per-node `kg.add_node(embedding=…)` to one
`kg.add_nodes([... embedding ...])`, and `search_subgraph`/`k_hop` would
materialize via `get_nodes`:

| metric | A10 today | expected after | basis |
|--------|----------:|---------------:|-------|
| `node_build_nodes_per_s` | 122 | ~6 000–6 600 | A10 `add_vectors_many` ceiling 6 609 docs/s; local ratio 20–31× |
| `node_build_ms` (17 517 nodes) | 143 019 | ~2 700 | one commit instead of 17 517 |
| `hybrid_latency` p50 (~7.8k frontier) | 182.6 | ≈ semantic search + ~1 batched read (small tens of ms) | 7 838 `get_node`→ ~20 `IN`; local 2× floor, larger at scale |
| `hybrid_latency` p95 | 451.6 | sharply lower | tail was the long per-node loops |
| `khop_latency` p50 (~1k frontier) | 18.4 | lower | 1 020 `get_node`→ ~3 `IN` |
| `edge_build_edges_per_s` | 10 718 | ~unchanged or slightly up | already batched; `executemany` removes per-edge txn overhead |

Node build should rise ~an order of magnitude (toward the `add_many` /
edge-build rate) and hybrid latency should drop to roughly *semantic search +
one batched read* — exactly the success criteria.

---

## 7. Open questions

1. **Single-commit guarantee for mixed-kind batches.** Accept the two-commit
   case for embedding+label mixes (recoverable, recommended), or constrain
   `add_nodes` to homogeneous batches? Leaning accept-and-document (§5).
2. **`ingest()` flush granularity.** Buffer the whole batch (simplest, matches
   `add_many`) or auto-flush every *M* entities for unbounded streams (Neo4j/
   Qdrant suggest 10k–100k)? Propose: buffer-all now, add `flush_every` only if
   huge ingests appear.
3. **`get_nodes` for `semantic_edges` too.** `semantic_edges` has the same
   per-hit `get_edge` loop ([`knowledge_graph.py:309-314`](../../src/lodedb/graph/knowledge_graph.py#L309));
   a parallel `get_edges(ids)` is trivial and worth bundling, though `k≤10`
   makes it minor. Bundle for symmetry?
4. **Default `max_nodes`?** Keep unbounded by default (chosen, for
   compatibility), or ship a generous default (e.g. 50 000) so a pathological hub
   can't OOM an unsuspecting caller? A default would be a behavior change; prefer
   opt-in + a documented recommendation.
5. **`add_node`/`add_edge` on top of the batch path.** Reimplement the singular
   verbs as `add_nodes([one])` to delete the duplicated routing in `_index_node`,
   or keep the fast single-doc path? The single-doc `add_vectors`/`add` are
   marginally cheaper per call, so keep both but share the routing helper.
6. **Engine-side `get_documents(ids)` batched metadata read.** Orthogonal but
   adjacent: `reindex()` and the hybrid metadata path would also benefit from a
   batched by-id metadata read on LodeDB itself — tracked by
   [`05`](05-batched-metadata-hydration.md); the graph `get_nodes` is the SQLite
   half of the same idea.
