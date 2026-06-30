# memory_integrations benchmark

Compares LodeDB's **LangChain**, **LlamaIndex**, and **mem0** adapters against each
framework's own default and common vector stores, on realistic memory workflows.
Metrics only (counts, bytes, latency, recall, backend labels). No raw documents,
queries, payloads, or embeddings are written, matching the repo's benchmark
provenance rules (`benchmarks/README.md`).

The question: *if you use LodeDB as the memory backend behind one of these
frameworks instead of its stock store, what changes?*

## Backends

Each LodeDB adapter is run against the framework's in-memory default plus the
common production stores it supports, all through the framework's own
vector-store interface (the exact contract the adapter implements):

| Framework | Interface | Default | Also compared |
|---|---|---|---|
| LangChain | `langchain_core.vectorstores.VectorStore` | `InMemoryVectorStore` | FAISS, Chroma, Qdrant, LanceDB, sqlite-vec, pgvector |
| LlamaIndex | `BasePydanticVectorStore` | `SimpleVectorStore` | Faiss, Chroma, Qdrant |
| mem0 | `mem0.vector_stores.base.VectorStoreBase` | Qdrant | FAISS, Chroma |

The embedded local-DB competitors (LanceDB, sqlite-vec, pgvector) are compared in the
LangChain suite, which uses the uniform `VectorStore` interface they all implement.

A backend whose library is not installed is skipped and recorded, so the suite
runs with whatever is present.

## Methodology: a store comparison, not an embedder comparison

The single biggest confound in "vector DB vs vector DB" benchmarks is the
embedding model. This suite removes it:

- **One fixed model** (`minilm`, 384-d) is embedded **once**, up front, warm. Every
  backend sees the same vectors.
- **Baselines receive the precomputed vectors** (LangChain via a caching
  `Embeddings`, LlamaIndex via `TextNode.embedding`, mem0 is vector-in natively),
  so their ingest is **store-only**.
- **LodeDB's LangChain/LlamaIndex adapters are text-path** (they embed internally
  by design), so their ingest is end-to-end and the runner reports a store-only
  figure by subtracting the same warm embed time. mem0's LodeDB adapter is
  vector-in, so no subtraction is needed there.
- **Queries run by precomputed query vector** (`similarity_search_by_vector` /
  `VectorStoreQuery(query_embedding=...)` / LodeDB `search_by_vector`), so query
  latency is store search, never query embedding.
- **Recall@k** is measured against exact brute-force cosine over the same vectors.

## What it measures

The workflow is RAG for LangChain/LlamaIndex (bulk ingest a corpus, retrieve) and
agent memory for mem0 (insert scoped memories, search, filtered search by user,
update). Both then exercise the part the stock stores are weakest at:

1. **Ingest** throughput (store-only docs/s).
2. **Query** latency (p50/p95) and **recall@10**.
3. **Persist + footprint.** Force durability and measure on-disk bytes. In-RAM
   stores (`InMemoryVectorStore`, `SimpleVectorStore`, FAISS) only reach disk here,
   via a full dump; LodeDB/Chroma/Qdrant persist on every write.
4. **Durable single add.** The cost to durably add **one** memory to an existing
   store. This is the agent-memory hot path, and it is where the in-memory
   defaults fall over: durability means rewriting the whole store (O(corpus)),
   while LodeDB appends an O(changed) delta.
5. **Reopen.** Close and reopen at the same path, confirm the data (and the
   incremental adds) survived.
6. **mem0 only:** filtered search by `user_id` (latency + within-user recall) and
   `update` latency.

## Reproduce

Local (synthetic, no network; validates the harness with whatever libs are
installed):

```bash
python benchmarks/memory_integrations/run.py \
    --dataset synthetic --max-documents 2000 --query-count 128 --device cpu
```

Modal (GovReport, downloaded on Modal; embedding on GPU; full baseline matrix):

```bash
modal run benchmarks/memory_integrations/modal_bench.py::smoke      # tiny synthetic (A10)
modal run benchmarks/memory_integrations/modal_bench.py::main_a10   # 50k docs (A10)
modal run benchmarks/memory_integrations/modal_bench.py::main_l40s  # 50k docs (L40S)
```

The Modal image compiles the working tree's LodeDB (vendored TurboVec via maturin)
and installs the three frameworks + baseline stores at pinned versions (see
`modal_bench.py`), so it benchmarks this branch's code.

## Results

Full JSON in `results/` (`agg_l40s_r1.json`, `agg_l40s_r2.json`, `agg_l40s_r3.json`,
the three runs averaged below; plus `results_compare_l40s.json` for the Python-core vs
native-Rust-core comparison and a tiny `results_smoke.json`). `minilm` (384-d), CPU
TurboVec scan; embedding on an L40S GPU. Provenance: `measured`. Corpus: the full GovReport summarization train split,
which yields about 17,500 documents (the `max_documents` knob is a cap, and this
split is the same corpus the `graph_memory` benchmark uses). LodeDB uses 4-bit
TurboVec quantization, so its recall is intentionally below the exact stores' 1.0;
that trade buys footprint and durable-write cost.

**What uses the GPU.** The GPU only does two things, and only for LodeDB:

1. The shared **embedding** step (held constant across every backend, computed once,
   subtracted to give store-only ingest).
2. LodeDB's **batched** `search_many_by_vector` path in the "Batched retrieval"
   section below, which engages the GPU-resident scan at batch >= 2.

Everything else is **CPU on every backend**, including LodeDB. Single-query search
stays on the CPU kernel by design (see [`runtime_policy.py`](../../src/lodedb/engine/runtime_policy.py)
`gpu_direct_turbovec_should_use`), so the ingest / single-query / durable-add /
footprint tables are a CPU-vs-CPU store comparison: LodeDB's CPU TurboVec scan
against FAISS (CPU), Chroma, Qdrant, LanceDB, sqlite-vec, pgvector (all local), and
the in-memory defaults; the GPU never touches those store metrics, only the shared
embedding step and the batched scan.

The tables below are the **L40S** run. `docs/s` is store-only ingest,
`query p50` is by-vector search, `add p50` is the durable single-add (sampled under a
wall-clock budget, so a slow full-rewrite store is sampled fewer times), `footprint`
is durable on-disk size, and `vs LodeDB` is the footprint ratio. CPU latencies vary
run to run on Modal's shared hosts (see Caveats), so the **multipliers are the stable
read** (ratios cancel host speed); the absolute figures are the mean of 3 L40S runs of
the native Rust core (`CoreEngine`, default on).

### Headline: LodeDB vs each framework's default store

At ~17.5k docs on an L40S, mean of 3 runs. Each cell is LodeDB's figure vs the default's,
then the multiplier.

| Axis | LangChain (`InMemoryVectorStore`) | LlamaIndex (`SimpleVectorStore`) | mem0 (Qdrant) |
|---|---|---|---|
| **On-disk footprint** | 15 vs 208 MB = **13.6x smaller** | 15 vs 152 MB = **9.9x smaller** | 13 vs 73 MB = **5.7x smaller** |
| **Single-query p50** (CPU) | 0.45 vs 272 ms = **~600x faster** | 0.44 vs 272 ms = **~620x faster** | 0.59 vs 27 ms = **~46x faster** |
| **Batched retrieval, 64** (GPU) | 11,049 vs ~4 qps = **~2,880x** | 11,297 vs ~4 qps = **~3,050x** | 5,084 vs 36 qps = **~139x** |
| **Durable single add** | 0.26 ms vs 6.9 s = **~26,000x faster** | 0.26 ms vs 14.8 s = **~57,000x faster** | 0.28 vs 0.44 ms (both sub-ms) |
| **Recall@10** | 0.95 vs 1.00 | 0.95 vs 1.00 | 0.95 vs 1.00 (filtered 0.95 vs 1.00) |

Every backend, LodeDB included, is fed the same precomputed vectors (LodeDB via its
vector-in SDK), so none is charged for embedding -- a store-vs-store comparison. The
in-memory defaults rewrite the whole store to persist one memory and scan in pure
Python with no batch path. mem0's default Qdrant is a real DB, so its single add is
fast (0.44 ms), and LodeDB now matches it (0.28 ms) while reading far faster, storing
far smaller, and batching far harder.

### LangChain (default `InMemoryVectorStore`), RAG over ~17.5k docs

The LangChain suite carries the full embedded local-DB field: the in-memory default,
FAISS, Chroma, Qdrant, and the three direct competitors **LanceDB, sqlite-vec, and
pgvector**.

| backend | ingest docs/s | query p50 (ms) | recall@10 | durable add p50 | delta? | footprint | vs LodeDB |
|---|---|---|---|---|---|---|---|
| **lodedb** | 2,554 | **0.45** | 0.95 | **0.26 ms** | yes | **15 MB** | 1.0x |
| inmemory (default) | 200,029 | 271.7 | 1.00 | 6,887 ms | no | 208 MB | 13.6x |
| faiss | 32,669 | 0.47 | 1.00 | 88.7 ms | no | 45 MB | 2.9x |
| chroma | 841 | 3.35 | 1.00 | 5.9 ms | yes | 151 MB | 9.8x |
| qdrant | 1,334 | 13.9 | 1.00 | 0.48 ms | yes | 85 MB | 5.6x |
| lancedb | 5,830 | 10.6 | 1.00 | 3.4 ms | yes | 37 MB | 2.4x |
| sqlite-vec | 24,272 | 26.8 | 1.00 | 0.42 ms | yes | 101 MB | 6.6x |
| pgvector | 1,947 | 35.1 | 1.00 | 2.3 ms | yes | 50 MB | 3.3x |

Among the embedded local DBs, **LodeDB has the smallest footprint and the fastest
single query** -- LanceDB (10.6 ms), sqlite-vec (27 ms), and pgvector (35 ms) are 24x to
78x slower per query because they scan without LodeDB's quantized SIMD kernel, trading
the 5 points of recall LodeDB gives up. LanceDB is the closest on footprint (37 MB vs
15 MB). pgvector runs via an embedded `pgserver` (Postgres + pgvector, no separate
service) with no ANN index, so it is an exact seq scan like LodeDB, LanceDB, and
sqlite-vec. On durable single-add LodeDB's WAL-mode O(changed) commit (0.26 ms) leads
even the lazy-append stores (sqlite-vec 0.42 ms, qdrant 0.48 ms) -- see "Durable add"
under Reading.

### LlamaIndex (default `SimpleVectorStore`), RAG over ~17.5k docs

| backend | ingest docs/s | query p50 (ms) | recall@10 | durable add p50 | delta? | footprint | vs LodeDB |
|---|---|---|---|---|---|---|---|
| **lodedb** | 2,727 | **0.44** | 0.95 | **0.26 ms** | yes | **15 MB** | 1.0x |
| simple (default) | 20,613 | 272.4 | 1.00 | 14,774 ms | no | 152 MB | 9.9x |
| faiss | 19,176 | 0.31 | 1.00 | 15.2 ms | no | 27 MB | 1.8x |
| chroma | 800 | 3.56 | 1.00 | 6.2 ms | yes | 173 MB | 11.3x |
| qdrant | 1,923 | 24.6 | 1.00 | 0.54 ms | yes | 97 MB | 6.4x |

`SimpleVectorStore` reopens by reloading its 152 MB JSON in ~30 s; LodeDB reopens in
~3 s.

### mem0 (default Qdrant), agent-memory workflow over ~17.5k memories

This suite is vector-in (mem0 owns embeddings), so the durable-add column is
embed-free for every backend -- the fair persist-to-persist comparison.

| backend | ingest docs/s | query p50 (ms) | recall@10 | filtered recall | durable add p50 | footprint | vs LodeDB |
|---|---|---|---|---|---|---|---|
| **lodedb** | 2,472 | **0.59** | 0.95 | 0.95 | **0.28 ms** | **13 MB** | 1.0x |
| qdrant (default) | 1,526 | 27.1 | 1.00 | 1.00 | 0.44 ms | 73 MB | 5.7x |
| faiss | 56,737 | 0.34 | 1.00 | **0.04** | 103.2 ms | 31 MB | 2.4x |
| chroma | 2,398 | 3.00 | 1.00 | 1.00 | 6.4 ms | 49 MB | 3.8x |

### Batched retrieval (batch = 64): the GPU path

Single-query search is CPU. The batched path is where LodeDB engages its GPU-resident
scan: `search_many_by_vector` at batch >= 2 runs the whole batch as one resident scan.
LangChain's and LlamaIndex's retriever contracts are single-query, so their stores
answer a batch as a loop; mem0 exposes `search_batch`, so its providers use it.
Throughput is warm steady-state queries/sec (a warmup batch excludes the one-time GPU
index upload); batched recall matches single-query recall (0.95 for LodeDB).

| Framework (batch = 64, L40S) | LodeDB qps | default-store qps | LodeDB vs default | best alternative |
|---|---|---|---|---|
| LangChain | **11,049** | `InMemoryVectorStore` ~4 | **~2,800x** | FAISS-CPU 1,776* |
| LlamaIndex | **11,297** | `SimpleVectorStore` ~4 | **~2,800x** | FAISS-CPU 2,942* |
| mem0 | **5,084** | Qdrant 36 | **~140x** | FAISS 2,680* |

Reading it:

- **vs the defaults.** The in-memory defaults have no batch path, so a batch is a
  loop of ~270 ms pure-Python scans (~4 qps). LodeDB's batched GPU scan is ~5,000 to
  11,300 qps on the L40S, roughly **140x to 2,800x** depending on the default. Against
  local Qdrant (~36 to 71 qps) it is ~140x to 160x; against Chroma (~300 qps) ~16x to
  36x.
- **Self-speedup.** LodeDB's batched GPU throughput far exceeds its single-query CPU
  path; the GPU-resident scan runs 2.8x to 4.8x the all-threads CPU ceiling at scale
  (see the project's GPU benchmarks).
- **FAISS is the one close baseline, and it is noisy.** `*` FAISS-CPU is the only
  baseline that batches efficiently, but its throughput is host-dependent on Modal's
  shared instances (~1,800 to 2,900 qps across suites here) -- treat it as
  "comparable to several-x slower than LodeDB," not a fixed number.
- **Scale.** At ~17.5k vectors the GPU is not yet compute-bound, so this understates
  the GPU win. The headline GPU throughput (24k qps A10, 50k qps L40S, 2.8x to 4.8x the
  CPU ceiling) is a larger-corpus result, measured at 100k to 1M vectors in
  [`govreport_scale`](../govreport_scale) and [`direct_gpu_sweep`](../direct_gpu_sweep).

## Reading the results

- **Footprint.** LodeDB's 4-bit quantized store is the smallest durable footprint of
  any backend tested: 13.6x smaller than `InMemoryVectorStore`, 9.9x than
  `SimpleVectorStore`, 5.7x than mem0's Qdrant, and smaller than every embedded local
  DB (LanceDB 37 MB, pgvector 50 MB, sqlite-vec 101 MB vs LodeDB 15 MB). This is what
  it costs to keep a growing agent memory persisted.
- **Durable single add.** Every backend is fed precomputed vectors, so this column is
  persist-only for all of them (no embedding). LodeDB's **0.26 ms** here leads even the
  lazy-append stores (sqlite-vec 0.42 ms, qdrant 0.48 ms, pgvector 2.3 ms, lancedb
  3.4 ms), because WAL is the default commit mode: each add appends one O(changed)
  framed record to a `.wal` log and a full generation is checkpointed periodically, so
  a durable single add is sub-millisecond while staying **crash-atomic** (the WAL
  replays on reopen, never a torn store) with **lock-free reader snapshots** (one
  writer, many readers per path). It is not fsync (default `durability="fast"`;
  `"fsync"` adds ~0.4 ms) and it amortizes under batching (`add_many` is one commit,
  hence the thousands/sec ingest above). The classic `commit_mode="generation"` path,
  which publishes a whole immutable generation per write, is there when many
  out-of-process readers must see each write the instant it commits.
- **Query latency.** The in-memory defaults scan in pure Python (~270 ms); LodeDB is
  under half a millisecond. Among the embedded local DBs LodeDB is also 24x to 78x
  faster per query than LanceDB/sqlite-vec/pgvector (all exact scans without the SIMD
  kernel) and ~31x faster than local-mode Qdrant. FAISS-flat is the only one near
  LodeDB's single-query range, and it is not durable (full rewrite, no payload
  round-trip).
- **Recall.** LodeDB returns 0.95 recall@10 (4-bit quantization) versus 1.00 for the
  exact/flat stores -- the deliberate trade for footprint and query speed. The scan
  is exact (no ANN graph), so there is no recall cliff to tune.
- **mem0 filtered search.** mem0's FAISS provider has no server-side filtering: it
  over-fetches only `2*k` then post-filters, so within-user recall collapses to 0.04
  at 2% selectivity (and its `update` rebuilds the index, ~254 ms). LodeDB, Qdrant,
  and Chroma push the `user_id` predicate into the index and stay accurate.

## Caveats

- **Latencies are measured on Modal's shared-CPU instances and vary run to run**
  (FAISS-CPU single-query p50 ranged ~0.4 to 4.4 ms across runs, batch throughput 256
  to 3,418 qps). Footprint, recall, and the order-of-magnitude gaps (seconds-vs-ms
  durable add, hundreds-of-ms-vs-sub-ms query) do not move; treat the precise figures
  as a single sample and the multipliers as the stable read.
- The in-memory defaults are not durable until dumped; the durable-add and footprint
  columns charge them for that dump, which is the real cost of persisting agent
  memory. Their in-RAM ingest and query speed is real but undurable.
- The embedded local DBs run exact (no ANN index): LanceDB, sqlite-vec, and pgvector
  do exact scans, so recall is 1.00 but single-query latency is tens of ms at this
  size. pgvector with an HNSW index would query faster but approximately; it is left
  index-free to match LodeDB's exact scan.
- LlamaIndex's `FaissVectorStore` keeps no docstore, so used standalone it returns the
  faiss positional index rather than the node id (and no payload); the harness maps
  positions back through insertion order so recall is correct.
- Qdrant and Chroma run in embedded/local mode (no server) -- the apples-to-apples
  comparison against an embedded store, not their tuned server configuration.
- LodeDB's cold reopen rebuilds calibration on first open, which at small corpora
  shows as a higher reopen time that amortizes at scale.
