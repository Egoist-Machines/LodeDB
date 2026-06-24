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
| LangChain | `langchain_core.vectorstores.VectorStore` | `InMemoryVectorStore` | FAISS, Chroma, Qdrant |
| LlamaIndex | `BasePydanticVectorStore` | `SimpleVectorStore` | Faiss, Chroma, Qdrant |
| mem0 | `mem0.vector_stores.base.VectorStoreBase` | Qdrant | FAISS, Chroma |

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
against FAISS (CPU), Chroma, Qdrant (local), and the in-memory defaults. The proof
is that the A10 and L40S store metrics match; only embedding throughput differs
(792 docs/s on the A10 versus 1,230 docs/s on the L40S).

The single-query tables below are the **A10** run. `docs/s` is store-only ingest,
`query p50` is by-vector search, `add p50` is the durable single-add (sampled `n`
times; a slow full-rewrite store is sampled fewer times under a wall-clock budget),
and `footprint` is durable on-disk size.

### Headline: LodeDB vs each framework's default store

The store each framework ships with, at ~17.5k docs on an A10. The bold figure is
LodeDB's; the multiplier is how much better LodeDB is on that axis.

| Axis | LangChain default (`InMemoryVectorStore`) | LlamaIndex default (`SimpleVectorStore`) | mem0 default (Qdrant) |
|---|---|---|---|
| **Durable single add** | 8,590 ms to **18 ms** = **477x faster** | 19,351 ms to **13.5 ms** = **1,433x faster** | 0.5 ms to 8.2 ms (Qdrant faster) |
| **Query p50** (single, CPU) | 288 ms to **0.57 ms** = **506x faster** | 334 ms to **0.56 ms** = **597x faster** | 26.9 ms to **0.62 ms** = **43x faster** |
| **Batched retrieval (64), qps** (GPU) | 4 to **4,045** = **~1,000x** | 3 to **6,684** = **~2,200x** | 38 to **5,007** = **~130x** |
| **On-disk footprint** | 199 MB to **28 MB** = **7.0x smaller** | 145 MB to **28 MB** = **5.3x smaller** | 70 MB to **15 MB** = **4.6x smaller** |
| **Recall@10** | 1.00 to 0.95 | 1.00 to 0.95 | 1.00 to 0.95 (filtered 1.00 to 0.95) |

LangChain's and LlamaIndex's defaults are in-memory: persisting one new memory
rewrites the whole store (O(corpus)), and queries are a pure-Python scan with no
batch path. mem0's default Qdrant is already a real DB, so its single-add is fast,
but LodeDB still reads far faster and stores far smaller. The batched row is where
LodeDB's GPU-resident scan engages (see "Batched retrieval" below). LodeDB trades 5
points of recall (4-bit quantization) for all of it.

### LangChain (default `InMemoryVectorStore`), RAG over ~17.5k docs

| backend | ingest docs/s | query p50 (ms) | recall@10 | durable add p50 | n | delta? | footprint | footprint vs LodeDB |
|---|---|---|---|---|---|---|---|---|
| **lodedb** | 5,582 | **0.57** | 0.95 | **18 ms** | 30 | yes | **28 MB** | 1.0x (baseline) |
| inmemory (default) | 174,630 | 288.4 | 1.00 | 8,590 ms | 6 | no | 199 MB | **7.0x bigger** |
| faiss | 27,206 | 0.44 | 1.00 | 112.1 ms | 30 | no | 43 MB | 1.5x bigger |
| chroma | 894 | 3.26 | 1.00 | 6.5 ms | 30 | yes | 144 MB | 5.1x bigger |
| qdrant | 1,438 | 15.8 | 1.00 | 0.6 ms | 30 | yes | 81 MB | 2.9x bigger |

### LlamaIndex (default `SimpleVectorStore`), RAG over ~17.5k docs

| backend | ingest docs/s | query p50 (ms) | recall@10 | durable add p50 | n | delta? | footprint | footprint vs LodeDB |
|---|---|---|---|---|---|---|---|---|
| **lodedb** | 7,970 | **0.56** | 0.95 | **13.5 ms** | 30 | yes | **28 MB** | 1.0x (baseline) |
| simple (default) | 20,623 | 334.1 | 1.00 | 19,351 ms | 3 | no | 145 MB | **5.3x bigger** |
| faiss | 17,385 | 0.28 | 1.00 | 31.3 ms | 30 | no | 26 MB | 0.9x (similar) |
| chroma | 815 | 3.50 | 1.00 | 5.8 ms | 30 | yes | 165 MB | 6.0x bigger |
| qdrant | 1,851 | 27.4 | 1.00 | 0.6 ms | 30 | yes | 93 MB | 3.4x bigger |

`SimpleVectorStore` reopens by reloading its 145 MB JSON in about 40 s; LodeDB
reopens in about 1.4 s.

### mem0 (default Qdrant), agent-memory workflow over ~17.5k memories

| backend | ingest docs/s | query p50 (ms) | recall@10 | filtered recall | durable add p50 | footprint | footprint vs LodeDB |
|---|---|---|---|---|---|---|---|
| **lodedb** | 5,128 | **0.62** | 0.95 | 0.95 | 8.2 ms | **15 MB** | 1.0x (baseline) |
| qdrant (default) | 1,535 | 26.9 | 1.00 | 1.00 | 0.5 ms | 70 MB | **4.6x bigger** |
| faiss | 27,142 | 0.27 | 1.00 | **0.04** | 319.9 ms | 30 MB | 1.9x bigger |
| chroma | 2,164 | 2.93 | 1.00 | 1.00 | 7.7 ms | 47 MB | 3.1x bigger |

### Batched retrieval (batch = 64): the GPU path

The single-query numbers above are all CPU. The batched path is where LodeDB
engages its GPU-resident scan: `search_many_by_vector` at batch >= 2 runs the whole
batch as one resident scan. LangChain's and LlamaIndex's retriever contracts are
single-query, so their stores answer a batch as a loop; mem0 exposes `search_batch`,
so its providers use it. Throughput is warm steady-state queries/sec (the one-time
GPU index upload is excluded with a warmup batch); batched recall matches
single-query recall (0.95 for LodeDB), so the GPU path is correctness-preserving.

| Framework (batch = 64) | LodeDB qps (A10 / L40S) | default-store qps | LodeDB vs default | best alternative |
|---|---|---|---|---|
| LangChain | **4,045 / 6,294** | `InMemoryVectorStore` 4 | **~1,000x** | FAISS-CPU 881* |
| LlamaIndex | **6,684 / 6,540** | `SimpleVectorStore` 3 | **~2,200x** | FAISS-CPU 3,418* |
| mem0 | **5,007 / 4,790** | Qdrant 38 | **~130x** | FAISS 3,116* |

Reading it:

- **vs the defaults.** The in-memory defaults have no batch path, so a batch is a
  loop of ~300 ms pure-Python scans: 3 to 4 qps. LodeDB's batched GPU scan is ~5,000
  to 6,700 qps, roughly **1,000x to 2,200x**. Against local Qdrant (~40 qps) it is
  ~120x; against Chroma (~290 qps) ~20x.
- **Self-speedup.** LodeDB's batched GPU throughput is about **3x its own
  single-query CPU rate** (for example 1,786 qps single to 6,684 batched on the A10),
  inside the 2.8x to 4.8x range the project reports for the GPU-resident scan.
- **FAISS is the one close baseline, and it is noisy.** `*` FAISS-CPU is the only
  baseline that batches efficiently, but its CPU throughput is host-dependent on
  Modal's shared instances (it ranged 361 to 3,418 qps across runs and suites here),
  so treat it as "comparable to a few-x slower than LodeDB," not a fixed number.
- **Scale.** At ~17.5k vectors the GPU is not compute-bound (A10 and L40S land
  within noise of each other), so this is a conservative view of the GPU win. The
  headline GPU throughput (24k qps A10, 50k qps L40S, 2.8x to 4.8x the CPU ceiling)
  is a larger-corpus result, measured at 100k to 1M vectors in
  [`govreport_scale`](../govreport_scale) and [`direct_gpu_sweep`](../direct_gpu_sweep).

## Reading the results

- **Footprint.** LodeDB's 4-bit quantized store is the smallest durable footprint
  in every framework: about 7x smaller than `InMemoryVectorStore`, 5x smaller than
  `SimpleVectorStore`, and 4.6x smaller than mem0's default Qdrant. Chroma is the
  heaviest on disk. This is what it costs to keep a growing agent memory persisted.
- **Durable single add.** The in-memory defaults are fast to *fill* but cannot
  durably add one memory without rewriting the entire store. At ~17.5k docs that is
  about 8.6 s per single add for `InMemoryVectorStore` and about 19 s for
  `SimpleVectorStore`, and it grows O(corpus). LodeDB appends an O(changed) delta in
  ~13 to 18 ms, flat as the store grows. FAISS shares the full-rewrite model; Chroma
  and Qdrant, like LodeDB, persist incrementally.
- **Query latency.** The in-memory defaults scan in pure Python, so single-query p50
  is ~290 to 334 ms at this corpus size, versus under 1 ms for LodeDB. LodeDB is also
  ~30 to 45x faster than the *local-mode* Qdrant client (which is unindexed; server
  Qdrant would differ).
- **Recall.** LodeDB returns 0.95 recall@10 (4-bit quantization) versus 1.00 for
  the exact and flat stores, the deliberate trade for footprint and write cost.
- **mem0 filtered search.** mem0's FAISS provider has no server-side filtering: it
  over-fetches only `2*k` then post-filters, so within-user recall collapses to
  0.04 at 2% selectivity (and its `update` rebuilds the index, ~320 ms). LodeDB,
  Qdrant, and Chroma push the `user_id` predicate into the index and stay accurate.

## Caveats

- **Latencies are measured on Modal's shared-CPU instances and vary run to run**
  (FAISS-CPU's single-query p50 ranged ~0.4 to 2.3 ms across runs, its batch
  throughput 361 to 3,418 qps). The deterministic results -- footprint, recall, and
  the order-of-magnitude gaps (seconds-vs-ms durable add, hundreds-of-ms-vs-sub-ms
  query) -- do not move; treat the precise CPU-baseline figures as a single sample.
- The in-memory defaults are not durable until dumped; the durable-add and
  footprint columns charge them for that dump, which is the real cost of persisting
  agent memory. Their in-RAM ingest and query speed is real but undurable.
- LlamaIndex's `FaissVectorStore` keeps no docstore, so used standalone it returns
  the faiss positional index rather than the node id (and no payload). The harness
  maps positions back through insertion order so recall is correct, but in a real
  app FAISS needs a separate docstore for the ids, text, and metadata that LodeDB
  stores inline.
- Qdrant and Chroma run in embedded/local mode here (no server), which is the
  apples-to-apples comparison against an embedded store like LodeDB but is not their
  tuned server configuration.
- LodeDB's cold reopen rebuilds calibration on first open, which at small corpora
  shows as a higher reopen time that amortizes at scale.
