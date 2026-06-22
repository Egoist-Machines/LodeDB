# Research prompt: engine-side filtered enumeration (O(matches), not O(corpus))

## Context

LodeDB v0.1.2 now exposes enumeration on the public SDK:
`LodeDB.list_documents(filter=...)` and `LodeDB.get_document(id)` in
`src/lodedb/local/db.py` (added for the graph/knowledge-graph layer). Enumeration —
the *complete* set of documents matching a filter, with no `k` cap and no query
vector — is the primitive a graph layer needs for deterministic traversal: "all
edges whose `src` is X", "all nodes of `type` Person". The `lodedb.graph` layer
(`src/lodedb/graph/`) and its `reindex()` depend on it.

**Current implementation is intentionally simple and O(corpus):**
`LodeDB.list_documents(filter=...)` calls the engine's `list_documents()` to
materialize *every* record, then applies the compiled predicate
(`_predicate.compile_metadata_filter`) in Python. Fine at thousands of docs;
wasteful at 10^6.

Meanwhile the engine already has the data structure to do this in O(matches): the
generation-keyed `_MetadataPostingIndex` in `src/lodedb/engine/core.py` resolves a
filter to an id set for the query allowlist. Enumeration is "return that id set
(and its redacted records)" — without ranking, without a query vector, without
materializing the whole corpus.

## The problem to investigate

Design and (ideally) implement an **engine-side filtered enumeration** API:

- `LodeEngine`/`LodeIndex` method that resolves a validated filter to the complete
  matching id set via the posting index and returns the redacted records for just
  those ids (reuse `_document_resource_payload`), paged/streamed if large.
- A `count(filter=...)` that returns `len(matching set)` without materializing
  records at all.
- Wire `LodeDB.list_documents(filter=...)` / a new `LodeDB.count(filter=...)` to it,
  keeping the current Python fallback only for predicates the index can't yet
  resolve (coordinate with prompt 01's planner).

Cover: streaming/iterator semantics for very large result sets; interaction with the
predicate planner (01) so both share one resolution path; and behavior on read-only
snapshot handles.

## Invariants to respect

- **Payload-free**: records stay `{document_id, metadata, chunk_count,
  content_hash}` — no text/vectors (see `_document_resource_payload`).
- **Snapshot isolation**: enumeration must read one committed generation
  consistently (works on `read_only=True` handles).
- **O(changed) commits**: no new per-write cost.

## Success criteria

`list_documents(filter=)` / `count(filter=)` latency scales with the result-set
size, not the corpus size. Demonstrate with a benchmark (extend
`benchmarks/graph_memory/` with an enumeration case) showing 1-hop traversal
("all edges of a node") staying flat as the graph grows from 10k → 1M edges on
Modal. No change to on-disk format; existing readers unaffected.
