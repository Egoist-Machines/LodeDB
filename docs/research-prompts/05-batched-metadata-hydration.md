# Research prompt: remove the per-hit metadata N+1 in search hydration

## Context

In `src/lodedb/local/db.py`, `LodeDB.search` / `search_many` (and the new
`search_by_vector` / `search_many_by_vector`) hydrate results via
`_hits_from_result_rows`, which calls `self._metadata_for_document(document_id)` —
i.e. `self._index.get_document(id)` — **once per hit**. So a single `search(k=100)`
issues ~100 engine `get_document` round-trips after the scan, and a batched
`search_many` multiplies that by the batch size.

The graph/knowledge-graph layer makes this hotter: `semantic_nodes` and
`search_subgraph` (`src/lodedb/graph/knowledge_graph.py`) call search and then read
hit metadata for every seed, on every query.

## The problem to investigate

Eliminate the N+1. Options:

1. **Batched metadata fetch** — add an engine/`LodeIndex` method that returns
   redacted records for a *list* of ids in one call (mirroring the existing
   `get_document_texts` batch shape), and have `_hits_from_result_rows` call it once
   per search instead of per hit.
2. **Return metadata in the query payload** — the engine's
   `_materialize_query_results` already touches `state.document_metadata` to apply
   filters; consider attaching the (already in-memory) redacted metadata to each
   result row so no second lookup is needed at all. Check this stays within the
   payload-free result contract (`_search_result_from_payload` forbids text/vector
   fields — metadata is already allowed).

Measure the per-query overhead today (it's pure Python dict lookups, so likely
small per call but real at high k / large batches) and the improvement.

## Invariants to respect

- **Payload-free**: results carry redacted metadata only — never text/vectors
  (`_search_result_from_payload`'s forbidden-key check must still pass).
- No change to the public hit shape `(score, id, metadata)` / `LodeSearchHit`.

## Success criteria

`search(k=N)` issues O(1) metadata lookups instead of O(N); `search_many` issues
O(1) per batch instead of O(N·batch). Show the latency improvement at high k and
large batch sizes in `benchmarks/graph_memory/` (the `vector_in` sub-benchmark
already times search latency — extend it to sweep k) on Modal. No change to results.
