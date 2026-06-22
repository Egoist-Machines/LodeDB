# graph_memory benchmark

Measures the three capabilities added for the graph / knowledge-graph & memory
stack on LodeDB. **Metrics-only** (counts, latency, throughput, recall/overlap) —
no raw documents, queries, or embeddings are written, per the repo's benchmark
provenance rules (`benchmarks/README.md`).

## What it measures

1. **`vector_in`** — text-in ingest (LodeDB embeds internally) vs vector-in ingest
   (caller supplies precomputed vectors), plus query latency and an exact **parity**
   check: `search` vs `search_by_vector` over byte-identical indexes
   (`topk_overlap` should be `1.0`). Documents are capped to one chunk so the two
   indexes are identical and the comparison isolates the embedding cost vector-in
   removes.
2. **`filters`** — `search` latency across predicate kinds/selectivities: exact
   `$eq` / `$in` (posting-allowlist pushdown) vs `$gte` / range / `$ne` / `$exists`
   / `$and` (predicate evaluation). Quantifies the filter-planner opportunity in
   `docs/research-prompts/01`.
3. **`graph`** — a synthetic knowledge graph over the corpus (`lodedb.graph`):
   node/edge build throughput, k-hop traversal latency (SQLite topology), and hybrid
   `search_subgraph` latency (semantic seed + structural expansion).

## Reproduce

Local (synthetic, no network — quick validation):

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

The Modal image compiles the working tree's LodeDB (vendored TurboVec via maturin),
so it benchmarks **this branch's** code, matching `uv sync` / CI.

## Results

Headline numbers (full JSON in `results/`). `minilm` (384-d), CPU TurboVec scan;
embedding device as noted. Provenance: `measured`.

### Smoke — synthetic, 500 docs, Modal A10 (CUDA embedding)

| metric | value |
|---|---|
| vector-in ingest speedup vs text-in | **10.9×** |
| top-k overlap (search vs search_by_vector) | **1.0** (exact parity) |
| search latency p50 — text vs vector | 4.33 ms → **0.28 ms** |
| k-hop p50 (deg 8, 2 hops) | 3.32 ms |
| hybrid `search_subgraph` p50 | 15.8 ms |

### Full — GovReport, 50k docs/nodes, deg 16, Modal A10

_Populated by `modal run ...::main_a10` (`results/results_a10.json`)._

## Notes

- Vector-in's speedup is the embedding cost it removes; the scan path is shared, so
  results are identical (overlap `1.0`) — vector-in is a faster ingest/query path
  for callers who already hold embeddings, not a different index.
- k-hop / hybrid latency depends on graph density (avg degree) and `hops`; a dense
  random graph expands to a large fraction of nodes within 2 hops, which is the
  upper-bound stress case rather than a typical sparse knowledge graph.
