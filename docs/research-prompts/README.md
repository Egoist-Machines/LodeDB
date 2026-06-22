# Deep-research prompts — graph/knowledge-graph & memory stack

Optimization opportunities surfaced while building the graph-knowledge-memory
stack (vector-in API, public enumeration, and the `lodedb.graph` layer) on top of
LodeDB v0.1.2. Each file is a **self-contained prompt for a deep-research agent**:
it states the problem, points at the relevant code, lists the invariants to
respect, and defines what a good answer looks like. They are handoff artifacts —
none of them is started.

Read these alongside `AGENTS.md` (the hard invariants: payload-free artifacts,
O(changed) commit path, lean deps) and `docs/architecture.md`.

| # | Prompt | One-line | Rough leverage |
|---|--------|----------|----------------|
| 01 | [filter-predicate-planner](01-filter-predicate-planner.md) | Push `$gte`/`$ne`/`$exists` filters into the posting index / secondary indexes instead of per-doc scan | High — gates temporal memory & edge-type traversal at scale |
| 02 | [engine-side-filtered-enumeration](02-engine-side-filtered-enumeration.md) | Make `list_documents(filter=)` O(matches) via posting-set resolution, not O(corpus) in Python | High — the 1-hop traversal primitive |
| 03 | [arbitrary-dim-vector-only-index](03-arbitrary-dim-vector-only-index.md) | A bring-your-own-vectors index mode (any dim, no internal embedder) for Graphiti/cognee/Letta | High — the memory-systems integration unlock |
| 04 | [durable-vector-rebuildable-index](04-durable-vector-rebuildable-index.md) | Make the graph's semantic index fully rebuildable even for vector-in nodes | Medium |
| 05 | [batched-metadata-hydration](05-batched-metadata-hydration.md) | Remove the per-hit `get_document` N+1 in search result hydration | Medium |

Quantify candidate fixes with `benchmarks/graph_memory/` (vector-in throughput,
filter-predicate latency by selectivity, graph traversal) on Modal A10/L40S.
