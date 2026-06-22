# Research prompt: a bring-your-own-vectors index mode (arbitrary dim, no embedder)

## Context

LodeDB v0.1.2 added a **vector-in** API — `LodeDB.add_vectors`,
`add_vectors_many`, `search_by_vector`, `search_many_by_vector` in
`src/lodedb/local/db.py`, backed by `LodeEngine.upsert_vectors` /
`_ingest_vectors` and an `embedding`-carrying `EngineQuery` in
`src/lodedb/engine/core.py`. It lets a caller supply precomputed embeddings and
skip LodeDB's internal embedder.

**Today's constraint:** vectors must match the index's `native_dim`, which is
pinned by the model preset (`minilm` → 384, `bge` → 768) and enforced by several
validators (`EngineRoutePolicy.validate_index_request`, the `native_dim` checks in
`create_index` and `_chunks_have_full_embeddings`). So vector-in works only for an
external embedder whose dimension happens to match a preset.

That blocks the highest-leverage integration. Graph-memory frameworks
(Graphiti/Zep, cognee, Letta) **own their embedder** — often 1536-d or 3072-d — and
want a local, exact, crash-atomic vector store to plug in underneath. To be that
backend, LodeDB needs an index that holds **arbitrary-dimension** vectors with **no
internal embedding model at all**.

## The problem to investigate

Design a **vector-only index mode**:

- A construction path (e.g. `LodeDB(path, vector_dim=1536)` or
  `embedding="none"`) that builds an index pinned to a caller-chosen dim with **no**
  `SentenceTransformerEmbeddingBackend` attached.
- `add_vectors` / `search_by_vector` work as today but at the chosen dim; the
  text-in methods (`add`, `search`) raise a clear "this index is vector-only" error.
- Decide how this interacts with the route policy / model-consistency validators
  (`EngineRoutePolicy`, `create_index` at the `native_dim`/model checks): likely a
  dedicated route profile or a documented bypass that still records a stable,
  redacted "model identity" so reopening validates dim consistency.
- Persisted snapshot must record the dim/mode so reopening enforces it, and mixing
  models in one index stays prevented.

Investigate the TurboVec bit-width/quantizer behavior at large dims (1536/3072):
calibration cost, memory, and scan latency vs the 384/768 presets — does the
quantizer hold recall? Quantify.

## Invariants to respect

- **Payload-free artifacts**: raw input vectors stay transient (see
  `_discard_direct_turbovec_transient_embeddings`); only quantized codes persist.
- **O(changed) commits** and the **generation-addressed atomic manifest** must be
  reused unchanged (vector-in already does — extend, don't fork).
- **Lean deps**: a vector-only index should *reduce* the dependency surface (no
  sentence-transformers needed), not add to it — consider making the embedder import
  lazy so a vector-only install can skip it.

## Success criteria

A LodeDB index can be created at an arbitrary dim with no embedder, accept and
search 1536-d vectors, persist and reopen correctly, and preserve recall vs a
brute-force oracle. Demonstrate a thin **Graphiti or cognee vector-store adapter**
backed by it (even if only a proof of concept), and benchmark ingest/query at
100k–1M external vectors on Modal A10/L40S. This is the concrete step that turns
"LodeDB as the vector backend for knowledge-graph memory" into reality (see the
roadmap row in `docs/integrations.md`).
