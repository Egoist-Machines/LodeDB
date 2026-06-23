# graph_memory benchmark

Measures the three capabilities added for the graph and knowledge-graph memory
stack on LodeDB. Metrics only (counts, latency, throughput, recall and overlap).
No raw documents, queries, or embeddings are written, matching the repo's
benchmark provenance rules (`benchmarks/README.md`).

## What it measures

1. **`vector_in`.** Text-in ingest (LodeDB embeds internally) versus vector-in
   ingest (the caller supplies precomputed vectors), plus query latency and an
   exact parity check: `search` versus `search_by_vector` over byte-identical
   indexes, where `topk_overlap` should be 1.0. Documents are capped to one chunk
   so the two indexes are identical and the comparison isolates the embedding cost
   that vector-in removes.
2. **`filters`.** `search` latency across predicate kinds and selectivities: exact
   `$eq` and `$in` (posting-allowlist pushdown) versus `$gte`, range, `$ne`, and
   `$exists` (predicate evaluation). Compares the per-field filter planner against
   a per-document scan.
3. **`graph`.** A synthetic knowledge graph over the corpus (`lodedb.graph`):
   node and edge build throughput, k-hop traversal latency (SQLite topology), and
   hybrid `search_subgraph` latency (semantic seed plus structural expansion).

## Reproduce

Local (synthetic, no network, for quick validation):

```bash
python benchmarks/graph_memory/graph_memory_bench.py \
    --dataset synthetic --max-documents 2000 --query-count 64 \
    --graph-nodes 2000 --avg-degree 8 --hops 2 --device cpu
```

Modal (GovReport, downloaded on Modal; embedding on GPU):

```bash
modal run benchmarks/graph_memory/modal_bench.py::smoke      # tiny synthetic (A10)
modal run benchmarks/graph_memory/modal_bench.py::main_a10   # 50k docs/nodes (A10)
modal run benchmarks/graph_memory/modal_bench.py::main_l40s  # 50k docs/nodes (L40S)
```

The Modal image compiles the working tree's LodeDB (vendored TurboVec via
maturin), so it benchmarks this branch's code, matching `uv sync` and CI.

## Results

Headline numbers (full JSON in `results/`). `minilm` (384-d), CPU TurboVec scan;
embedding device as noted. Provenance: `measured`.

### Smoke: synthetic, 500 docs, Modal A10 (CUDA embedding)

| metric | value |
|---|---|
| vector-in ingest speedup vs text-in | 10.9x |
| top-k overlap (search vs search_by_vector) | 1.0 (exact parity) |
| search latency p50, text then vector | 4.33 ms to 0.28 ms |
| k-hop p50 (deg 8, 2 hops) | 3.32 ms |
| hybrid `search_subgraph` p50 | 15.8 ms |

### Full: GovReport (17.5k docs, full train split), Modal A10 (CUDA embedding)

Full JSON in `results/results_a10.json`. `minilm` (384-d), CPU TurboVec scan. This
are the current numbers; the "baseline" columns are the pre-optimization v0.1.2
figures, shown for the metrics that changed.

vector-in (docs capped to one chunk for exact parity):

| metric | value |
|---|---|
| ingest, vector-in vs text-in | 6.6x faster (no embedding step) |
| search p50, text / vector | 6.4 ms / 0.9 ms |
| top-k overlap | 1.0 (exact parity) |

filters (p50, k=10, 256 queries): the per-field planner versus the prior
O(corpus) compiled-matcher scan.

| predicate | baseline (scan) | planner | speedup |
|---|---|---|---|
| `no_filter` | 7.5 ms | 9.7 ms | n/a |
| `eq_topic` (`$eq`) | 14.0 ms | 13.9 ms | 1.0x |
| `exists_topic` (`$exists`) | 47.0 ms | 43.6 ms | 1.1x |
| `ne_topic` (`$ne`) | 51.8 ms | 40.6 ms | 1.3x |
| `gte_year` (`$gte`) | 56.6 ms | 26.8 ms | 2.1x |
| `range_year` (`$gte` and `$lt`) | 73.0 ms | 17.7 ms | 4.1x |

Exact `$eq` / `$in` already rode the posting allowlist and are unchanged. Ordered
and range predicates were O(corpus); the planner makes them O(matches + log V), so
they drop 2 to 4x at 17.5k docs and the gap widens with corpus size.

graph (17.5k nodes, 280k edges, avg degree 16, 2 hops):

| op | baseline | optimized |
|---|---|---|
| node build | 122 nodes/s (per-node `add_node`) | 4,526 nodes/s (`add_nodes`), about 24x |
| k-hop p50 | 18.4 ms | 16.4 ms |
| hybrid `search_subgraph` p50 | 182.6 ms | 175.0 ms |

The batched `add_nodes` / `add_edges` lift build from the per-node commit rate
toward the `add_vectors_many` ceiling. Traversal now uses a batched `get_nodes`;
on this dense 2-hop stress case (a frontier of about 7.8k nodes, roughly 45% of
the graph) latency stays frontier-bound, so the read-batching gain is modest. The
metadata-inline optimization is not visible here: the bench queries at k=10 (ten
per-hit reads), so it shows at high k (a measured 57% drop at k=100) and is
covered by a unit test.

## Notes

- Vector-in's speedup is the embedding cost it removes. The scan path is shared, so
  results are identical (overlap 1.0). Vector-in is a faster ingest and query path
  for callers who already hold embeddings, not a different index.
- k-hop and hybrid latency depend on graph density (avg degree) and `hops`. A dense
  random graph expands to a large fraction of nodes within 2 hops, which is the
  upper-bound stress case rather than a typical sparse knowledge graph.
