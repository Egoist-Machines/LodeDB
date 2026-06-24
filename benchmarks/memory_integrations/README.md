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

Full JSON in `results/` (`results_a10.json`, `results_l40s.json`, and a tiny
`results_smoke.json`). `minilm` (384-d), CPU TurboVec scan; embedding on the noted
GPU. Provenance: `measured`. Corpus: the full GovReport summarization train split,
which yields about 17,500 documents (the `max_documents` knob is a cap, and this
split is the same corpus the `graph_memory` benchmark uses). LodeDB uses 4-bit
TurboVec quantization, so its recall is intentionally below the exact stores' 1.0;
that trade buys footprint and durable-write cost.

**What uses the GPU.** The runs are named for their GPU (A10 / L40S), but the GPU
only does two things, and only for LodeDB:

1. The shared **embedding** step (held constant across every backend, computed once,
   subtracted to give store-only ingest).
2. LodeDB's **batched** `search_many_by_vector` path in the "Batched retrieval"
   section below, which engages the GPU-resident scan at batch >= 2.

Everything else is **CPU on every backend**, including LodeDB. Single-query search
stays on the CPU kernel by design (see [`runtime_policy.py`](../../src/lodedb/engine/runtime_policy.py)
`gpu_direct_turbovec_should_use`), so the ingest / single-query / durable-add /
footprint tables are a CPU-vs-CPU store comparison: LodeDB's CPU TurboVec scan
against FAISS (CPU), Chroma, Qdrant, LanceDB, sqlite-vec, pgvector (all local), and
the in-memory defaults. The proof is that the A10 and L40S store metrics match; only
embedding throughput differs (~730 docs/s on the A10 versus ~1,330 docs/s on the L40S).

The single-query tables below are the **A10** run. `docs/s` is store-only ingest,
`query p50` is by-vector search, `add p50` is the durable single-add (sampled under a
wall-clock budget, so a slow full-rewrite store is sampled fewer times), `footprint`
is durable on-disk size, and `vs LodeDB` is the footprint ratio. CPU latencies vary
run to run on Modal's shared hosts (see Caveats), so the **multipliers are the stable
read** (ratios cancel host speed) and the absolute figures are one measured sample.

### Headline: LodeDB vs each framework's default store

At ~17.5k docs on an A10. Each cell is LodeDB's figure vs the default's, then the
multiplier.

| Axis | LangChain (`InMemoryVectorStore`) | LlamaIndex (`SimpleVectorStore`) | mem0 (Qdrant) |
|---|---|---|---|
| **On-disk footprint** | 28 vs 199 MB = **7.0x smaller** | 28 vs 145 MB = **5.3x smaller** | 15 vs 70 MB = **4.6x smaller** |
| **Single-query p50** (CPU) | 0.88 vs 386 ms = **~440x faster** | 0.86 vs 427 ms = **~500x faster** | 0.93 vs 30 ms = **~32x faster** |
| **Batched retrieval, 64** (GPU) | 5,061 vs ~2 qps = **~1,000x+** | 5,047 vs ~2 qps = **~1,000x+** | 3,450 vs 32 qps = **~110x** |
| **Durable single add** | 30 ms vs 17.8 s = **~590x faster**† | 29 ms vs 28.5 s = **~990x faster**† | 19 ms vs 1.6 ms (Qdrant faster)† |
| **Recall@10** | 0.95 vs 1.00 | 0.95 vs 1.00 | 0.95 vs 1.00 (filtered 0.95 vs 1.00) |

The in-memory defaults rewrite the whole store to persist one memory and scan in pure
Python with no batch path. mem0's default Qdrant is a real DB, so its single add is
fast, but LodeDB reads far faster and stores far smaller. **†** the durable-add figure
for the text-path frameworks includes embedding the new document; the embed-free
persist cost is ~19 ms (the mem0 vector-in number) -- see "Durable add" under Reading.

### LangChain (default `InMemoryVectorStore`), RAG over ~17.5k docs

The LangChain suite carries the full embedded local-DB field: the in-memory default,
FAISS, Chroma, Qdrant, and the three direct competitors **LanceDB, sqlite-vec, and
pgvector**.

| backend | ingest docs/s | query p50 (ms) | recall@10 | durable add p50 | delta? | footprint | vs LodeDB |
|---|---|---|---|---|---|---|---|
| **lodedb** | 4,869 | **0.88** | 0.95 | 30.1 ms† | yes | **28 MB** | 1.0x |
| inmemory (default) | 30,209 | 385.9 | 1.00 | 17,775 ms | no | 199 MB | 7.0x |
| faiss | 27,323 | 4.36 | 1.00 | 222 ms | no | 43 MB | 1.5x |
| chroma | 231 | 11.7 | 1.00 | 29.1 ms | yes | 144 MB | 5.1x |
| qdrant | 406 | 29.1 | 1.00 | 3.5 ms | yes | 81 MB | 2.9x |
| lancedb | 1,496 | 27.1 | 1.00 | 9.7 ms | yes | 35 MB | 1.2x |
| sqlite-vec | 10,957 | 41.6 | 1.00 | 1.3 ms | yes | 96 MB | 3.4x |
| pgvector | 1,196 | 67.9 | 1.00 | 5.0 ms | yes | 47 MB | 1.7x |

Among the embedded local DBs, **LodeDB has the smallest footprint and the fastest
single query** -- LanceDB (27 ms), sqlite-vec (42 ms), and pgvector (68 ms) are
30x to 77x slower per query because they scan without LodeDB's quantized SIMD kernel,
trading the 5 points of recall LodeDB gives up. LanceDB is the closest on footprint
(35 MB vs 28 MB). pgvector runs via an embedded `pgserver` (Postgres + pgvector, no
separate service) with no ANN index, so it is an exact seq scan like LodeDB, LanceDB,
and sqlite-vec. **†** LodeDB's text-path add embeds the doc; embed-free persist is
~19 ms (the mem0 row).

### LlamaIndex (default `SimpleVectorStore`), RAG over ~17.5k docs

| backend | ingest docs/s | query p50 (ms) | recall@10 | durable add p50 | delta? | footprint | vs LodeDB |
|---|---|---|---|---|---|---|---|
| **lodedb** | 5,329 | **0.86** | 0.95 | 28.9 ms† | yes | **28 MB** | 1.0x |
| simple (default) | 11,709 | 426.8 | 1.00 | 28,544 ms | no | 145 MB | 5.3x |
| faiss | 13,092 | 1.54 | 1.00 | 12.7 ms | no | 26 MB | 0.9x |
| chroma | 274 | 12.5 | 1.00 | 29.8 ms | yes | 165 MB | 6.0x |
| qdrant | 524 | 34.8 | 1.00 | 3.1 ms | yes | 93 MB | 3.4x |

`SimpleVectorStore` reopens by reloading its 145 MB JSON in ~35 s; LodeDB reopens in
~1.4 s.

### mem0 (default Qdrant), agent-memory workflow over ~17.5k memories

This suite is vector-in (mem0 owns embeddings), so the durable-add column is
embed-free for every backend -- the fair persist-to-persist comparison.

| backend | ingest docs/s | query p50 (ms) | recall@10 | filtered recall | durable add p50 | footprint | vs LodeDB |
|---|---|---|---|---|---|---|---|
| **lodedb** | 3,588 | **0.93** | 0.95 | 0.95 | 19.2 ms | **15 MB** | 1.0x |
| qdrant (default) | 629 | 30.1 | 1.00 | 1.00 | 1.6 ms | 70 MB | 4.6x |
| faiss | 34,319 | 0.92 | 1.00 | **0.04** | 225.8 ms | 30 MB | 1.9x |
| chroma | 828 | 7.43 | 1.00 | 1.00 | 18.2 ms | 47 MB | 3.1x |

### Batched retrieval (batch = 64): the GPU path

Single-query search is CPU. The batched path is where LodeDB engages its GPU-resident
scan: `search_many_by_vector` at batch >= 2 runs the whole batch as one resident scan.
LangChain's and LlamaIndex's retriever contracts are single-query, so their stores
answer a batch as a loop; mem0 exposes `search_batch`, so its providers use it.
Throughput is warm steady-state queries/sec (a warmup batch excludes the one-time GPU
index upload); batched recall matches single-query recall (0.95 for LodeDB).

| Framework (batch = 64, A10) | LodeDB qps | default-store qps | LodeDB vs default | best alternative |
|---|---|---|---|---|
| LangChain | **5,061** | `InMemoryVectorStore` ~2 | **~1,000x+** | FAISS-CPU 256* |
| LlamaIndex | **5,047** | `SimpleVectorStore` ~2 | **~1,000x+** | FAISS-CPU 617* |
| mem0 | **3,450** | Qdrant 32 | **~110x** | FAISS 1,015* |

Reading it:

- **vs the defaults.** The in-memory defaults have no batch path, so a batch is a
  loop of ~300 ms pure-Python scans (~2 to 4 qps). LodeDB's batched GPU scan is
  ~3,500 to 5,000 qps on the A10 (7,282 qps on the L40S for LangChain), roughly
  **1,000x+**. Against local Qdrant (~32 qps) it is ~110x; against Chroma (~140) ~25x.
- **Self-speedup.** LodeDB's batched GPU throughput is **~3x its own single-query CPU
  rate**, inside the 2.8x to 4.8x range the project reports for the GPU-resident scan.
- **FAISS is the one close baseline, and it is noisy.** `*` FAISS-CPU is the only
  baseline that batches efficiently, but its throughput is host-dependent on Modal's
  shared instances (256 to 3,418 qps across runs and suites here) -- treat it as
  "comparable to several-x slower than LodeDB," not a fixed number.
- **Scale.** At ~17.5k vectors the GPU is not compute-bound (A10 and L40S land within
  noise), so this understates the GPU win. The headline GPU throughput (24k qps A10,
  50k qps L40S, 2.8x to 4.8x the CPU ceiling) is a larger-corpus result, measured at
  100k to 1M vectors in [`govreport_scale`](../govreport_scale) and
  [`direct_gpu_sweep`](../direct_gpu_sweep).

## Reading the results

- **Footprint.** LodeDB's 4-bit quantized store is the smallest durable footprint of
  any backend tested: 7x smaller than `InMemoryVectorStore`, 5.3x than
  `SimpleVectorStore`, 4.6x than mem0's Qdrant, and smaller than every embedded local
  DB (LanceDB 35 MB, pgvector 47 MB, sqlite-vec 96 MB vs LodeDB 28 MB). This is what
  it costs to keep a growing agent memory persisted.
- **Durable single add.** Two things drive LodeDB's per-add number. (1) The text-path
  suites (LangChain/LlamaIndex) charge LodeDB for embedding the new doc (~7 to 10 ms,
  a minilm forward pass) while the baselines get a precomputed vector; the embed-free
  persist is the mem0 vector-in figure, ~19 ms. (2) That persist is a full
  crash-atomic commit -- encode the row, append the O(changed) delta, write a new
  generation-addressed manifest, publish it atomically -- a stronger per-write
  guarantee than stores that append/fsync lazily (qdrant 1.6 ms, sqlite-vec 1.3 ms).
  It is not fsync (the default `durability="fast"` skips it; `"fsync"` adds ~0.4 ms),
  and it amortizes away under batching (`add_many` is one commit, hence the
  thousands-per-second ingest). Either way it is 500x to 1,000x faster than the
  in-memory defaults' multi-second full rewrite.
- **Query latency.** The in-memory defaults scan in pure Python (~290 to 430 ms);
  LodeDB is under 1 ms. Among the embedded local DBs LodeDB is also 30x to 77x faster
  per query than LanceDB/sqlite-vec/pgvector (all exact scans without the SIMD kernel)
  and ~30x faster than local-mode Qdrant. FAISS-flat is the only one in LodeDB's
  single-query range, and it is not durable (full rewrite, no payload round-trip).
- **Recall.** LodeDB returns 0.95 recall@10 (4-bit quantization) versus 1.00 for the
  exact/flat stores -- the deliberate trade for footprint and query speed. The scan
  is exact (no ANN graph), so there is no recall cliff to tune.
- **mem0 filtered search.** mem0's FAISS provider has no server-side filtering: it
  over-fetches only `2*k` then post-filters, so within-user recall collapses to 0.04
  at 2% selectivity (and its `update` rebuilds the index, ~226 ms). LodeDB, Qdrant,
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
