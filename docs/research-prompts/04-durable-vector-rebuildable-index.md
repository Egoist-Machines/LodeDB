# Research prompt: make the graph's semantic index fully rebuildable

## Context

The `lodedb.graph.KnowledgeGraph` layer (`src/lodedb/graph/knowledge_graph.py`)
treats SQLite as the **source of truth** for topology and LodeDB as a **rebuildable
semantic index**. That framing is what dissolves cross-store atomicity worries: if
an index write is lost (crash between the SQLite write and the LodeDB write),
`reindex()` rebuilds the index from SQLite.

**The gap:** `reindex()` rebuilds a node's index entry by re-embedding its `label`
text. But a node added via the vector-in path (`add_node(..., embedding=[...])`,
which calls `LodeDB.add_vectors`) has **no retained raw vector** — the embedding is
transient and only quantized codes persist in LodeDB, and the topology store
(`src/lodedb/graph/_store.py`) holds only `label`/`properties`, not the vector. So
vector-in nodes cannot be faithfully rebuilt from the source of truth; `reindex()`
documents this limitation today.

This breaks the "fully rebuildable derived index" guarantee for exactly the callers
who most want vector-in (those bringing their own embeddings).

## The problem to investigate

How to make the semantic index rebuildable for vector-in nodes too. Options to
evaluate (with trade-offs):

1. **Persist node vectors in the topology store** — add a `node_vectors` table/blob
   in `_store.py` so `reindex()` re-`add_vectors` from SQLite. Cost: storage
   duplication (raw f32 vectors), but the topology DB is already the source of
   truth, so this is consistent. Quantify the size overhead.
2. **Engine export/import of encoded vectors** — add a LodeDB path to export the
   quantized codes for an id and re-import them into a fresh index without
   re-embedding (the TurboVec binding already has `export_encoded`/`add_encoded` per
   `tests/conftest.py`'s mock — check the real binding in
   `third_party/turbovec/turbovec-python/`). Then "rebuild" = copy encoded rows,
   no source vectors needed. Cleaner but couples reindex to engine internals.
3. **A reindex hook/callback** — let the caller re-supply embeddings during
   `reindex()` (they own the embedder). Lowest storage cost, pushes work to the user.

Recommend one, considering the AGENTS.md payload-free boundary (option 1 stores raw
vectors in the *topology* DB, which is the user's own store — is that acceptable, or
should it also be redaction-aware?).

## Success criteria

`KnowledgeGraph.reindex()` faithfully reconstructs the semantic index for *all*
nodes — text-labelled and vector-in — such that post-reindex `semantic_nodes` /
`search_subgraph` results are identical to pre-reindex. Add a test that builds a KG
with vector-in nodes, drops the index dir, reindexes, and asserts identical
retrieval. Measure reindex throughput (nodes/s) in `benchmarks/graph_memory/`.
