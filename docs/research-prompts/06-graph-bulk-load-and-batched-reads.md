# Research prompt: graph bulk-load and batched topology reads

## Context

The `lodedb.graph.KnowledgeGraph` layer (`src/lodedb/graph/knowledge_graph.py`)
is built on per-entity operations: `add_node` / `add_edge` each write to SQLite and
then to LodeDB, and traversal (`k_hop`, `search_subgraph`) calls
`TopologyStore.get_node(id)` once per visited node.

The `benchmarks/graph_memory/` run on Modal A10 (GovReport, 17.5k nodes, 280k
edges, degree 16) surfaced two concrete costs:

- **Node build ≈ 122 nodes/s.** `add_node` calls `LodeDB.add_vectors` (or `add`)
  which commits atomically **per call** — so a bulk graph load pays one commit per
  node. Edge build, which is pure SQLite, runs at ~10,700 edges/s by comparison.
- **Hybrid `search_subgraph` p50 ≈ 183 ms** on a dense 2-hop expansion reaching
  ~7,800 nodes — dominated by ~7,800 individual `get_node` SQLite round-trips in
  `k_hop`.

Both are O(N)-round-trip patterns that a batched path removes.

## The problem to investigate

Design batched paths for the graph layer:

1. **Bulk ingest** — `add_nodes(list)` / `add_edges(list)` that group the LodeDB
   side into one `add_many` / `add_vectors_many` (a single commit) and the SQLite
   side into one `executemany` transaction, instead of one commit/transaction per
   entity. Quantify the build-throughput gain vs the per-call path. Decide the API
   shape (a bulk method, or an `ingest()` context manager that defers the LodeDB
   commit until exit).
2. **Batched topology reads** — `TopologyStore.get_nodes(ids)` (one `SELECT ...
   WHERE id IN (...)`, chunked like `edges_for`) and have `k_hop` /
   `search_subgraph` materialize the visited set in one batched read instead of
   per-node `get_node`. Pairs naturally with the search-hydration batching in
   `docs/research-prompts/05`.

Consider whether `k_hop` should also cap/stream very large frontiers (a `max_nodes`
budget) so a hub node or dense graph can't expand to the whole graph unboundedly,
and `log()` when it truncates (no silent caps).

## Invariants to respect

- Bulk ingest must keep the **atomic-commit** guarantee (one commit for the batch,
  all-or-nothing) and the source-of-truth ordering (SQLite first, then the
  rebuildable index) so a partial failure is still recoverable via `reindex()`.
- No change to the public `Node`/`Edge`/`Subgraph` shapes.

## Success criteria

Graph build throughput rises by ~an order of magnitude (toward the edge-build /
`add_many` rate rather than the per-commit rate), and `k_hop` / `search_subgraph`
latency on large frontiers drops to roughly one batched read instead of O(frontier)
round-trips. Re-run `benchmarks/graph_memory/` (graph sub-benchmark) on Modal A10
before/after and report node-build nodes/s and hybrid p50/p95.
