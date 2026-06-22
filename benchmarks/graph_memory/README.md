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

### Full — GovReport (17.5k docs, full train split), Modal A10 (CUDA embedding)

Full JSON in `results/results_a10.json`. `minilm` (384-d), CPU TurboVec scan.

**vector-in** (docs capped to one chunk for exact parity):

| metric | text-in | vector-in |
|---|---|---|
| ingest throughput | 811 docs/s | **6,609 docs/s** (8.1×) |
| search p50 | 5.16 ms | **0.63 ms** (8×) |
| top-k overlap (mean / min) | — | **0.9996 / 0.90** |

The few <1.0 overlaps are top-k **boundary ties** (equal scores ordered differently),
not divergence — the indexes are byte-identical.

**filters** (p50, k=10, 256 queries):

| predicate | p50 | vs `$eq` |
|---|---|---|
| `no_filter` | 7.5 ms | — |
| `eq_topic` (`$eq`, posting allowlist) | 14.0 ms | 1.0× |
| `and_topic_year` (`$and`) | 34.1 ms | 2.4× |
| `in_topic_3` (`$in`) | 37.2 ms | 2.6× |
| `exists_topic` (`$exists`) | 47.0 ms | 3.3× |
| `ne_topic` (`$ne`) | 51.8 ms | 3.7× |
| `gte_year` (`$gte`) | 56.6 ms | 4.0× |
| `range_year` (`$gte`+`$lt`) | 73.0 ms | 5.2× |

Exact `$eq`/`$in` ride the posting-index allowlist; ordered/negation predicates are
3–5× slower (per-candidate evaluation) — the gradient that motivates the filter
planner in `docs/research-prompts/01`.

**graph** (17.5k nodes, 280k edges, avg degree 16, 2 hops):

| op | p50 | p95 | avg subgraph |
|---|---|---|---|
| node build | 122 nodes/s | — | per-node commit bound → batch (`research-prompts/06`) |
| edge build | 10,719 edges/s | — | — |
| k-hop | 18.4 ms | 24.4 ms | 1,021 nodes |
| hybrid `search_subgraph` | 182.6 ms | 451.6 ms | 7,838 nodes |

This is a **dense random graph** (degree 16): a 2-hop neighbourhood reaches ~45% of
all nodes, so this is the stress-case upper bound, not a typical sparse KG. Hybrid
latency is dominated by per-node topology fetches over that large frontier — see
`docs/research-prompts/06` (graph bulk-load + batched reads).

## Notes

- Vector-in's speedup is the embedding cost it removes; the scan path is shared, so
  results are identical (overlap `1.0`) — vector-in is a faster ingest/query path
  for callers who already hold embeddings, not a different index.
- k-hop / hybrid latency depends on graph density (avg degree) and `hops`; a dense
  random graph expands to a large fraction of nodes within 2 hops, which is the
  upper-bound stress case rather than a typical sparse knowledge graph.
