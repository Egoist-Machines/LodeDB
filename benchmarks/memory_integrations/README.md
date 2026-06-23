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

The tables below are the **A10** run. The L40S run is store-for-store equivalent
(the scan is CPU-bound); the only material difference is embedding throughput, 792
docs/s on the A10 versus 1,230 docs/s on the L40S. `docs/s` is store-only ingest,
`query p50` is by-vector search, `add p50` is the durable single-add (sampled `n`
times; a slow full-rewrite store is sampled fewer times under a wall-clock budget),
and `footprint` is durable on-disk size.

### Headline: LodeDB vs each framework's default store

The store each framework ships with, at ~17.5k docs on an A10. The bold figure is
LodeDB's; the multiplier is how much better LodeDB is on that axis.

| Axis | LangChain default (`InMemoryVectorStore`) | LlamaIndex default (`SimpleVectorStore`) | mem0 default (Qdrant) |
|---|---|---|---|
| **Durable single add** | 10,482 ms to **16.9 ms** = **620x faster** | 22,132 ms to **16.5 ms** = **1,341x faster** | 0.7 ms to 9.7 ms (Qdrant faster) |
| **Query p50** | 343 ms to **0.85 ms** = **404x faster** | 422 ms to **0.84 ms** = **502x faster** | 26.3 ms to **0.91 ms** = **29x faster** |
| **On-disk footprint** | 199 MB to **28 MB** = **7.0x smaller** | 145 MB to **28 MB** = **5.3x smaller** | 70 MB to **15 MB** = **4.6x smaller** |
| **Recall@10** | 1.00 to 0.95 | 1.00 to 0.95 | 1.00 to 0.95 (filtered 1.00 to 0.95) |

LangChain's and LlamaIndex's defaults are in-memory: persisting one new memory
rewrites the whole store (O(corpus)), and queries are a pure-Python scan. mem0's
default Qdrant is already a real DB, so its single-add is fast, but LodeDB still
reads far faster and stores far smaller. LodeDB trades 5 points of recall (4-bit
quantization) for all of it.

### LangChain (default `InMemoryVectorStore`), RAG over ~17.5k docs

| backend | ingest docs/s | query p50 (ms) | recall@10 | durable add p50 | n | delta? | footprint | footprint vs LodeDB |
|---|---|---|---|---|---|---|---|---|
| **lodedb** | 5,771 | **0.85** | 0.95 | **16.9 ms** | 30 | yes | **28 MB** | 1.0x (baseline) |
| inmemory (default) | 43,254 | 343.5 | 1.00 | 10,482 ms | 6 | no | 199 MB | **7.0x bigger** |
| faiss | 23,207 | 0.71 | 1.00 | 128.9 ms | 30 | no | 43 MB | 1.5x bigger |
| chroma | 609 | 4.47 | 1.00 | 8.7 ms | 30 | yes | 144 MB | 5.1x bigger |
| qdrant | 902 | 17.3 | 1.00 | 0.9 ms | 30 | yes | 81 MB | 2.9x bigger |

### LlamaIndex (default `SimpleVectorStore`), RAG over ~17.5k docs

| backend | ingest docs/s | query p50 (ms) | recall@10 | durable add p50 | n | delta? | footprint | footprint vs LodeDB |
|---|---|---|---|---|---|---|---|---|
| **lodedb** | 5,542 | **0.84** | 0.95 | **16.5 ms** | 30 | yes | **28 MB** | 1.0x (baseline) |
| simple (default) | 17,273 | 421.6 | 1.00 | 22,132 ms | 3 | no | 145 MB | **5.3x bigger** |
| faiss | 16,938 | 0.47 | 1.00 | 14.6 ms | 30 | no | 26 MB | 0.9x (similar) |
| chroma | 574 | 5.00 | 1.00 | 8.7 ms | 30 | yes | 165 MB | 6.0x bigger |
| qdrant | 1,273 | 25.5 | 1.00 | 0.8 ms | 30 | yes | 93 MB | 3.4x bigger |

`SimpleVectorStore` reopens by reloading its 145 MB JSON in about 40 s; LodeDB
reopens in about 1.4 s.

### mem0 (default Qdrant), agent-memory workflow over ~17.5k memories

| backend | ingest docs/s | query p50 (ms) | recall@10 | filtered recall | durable add p50 | footprint | footprint vs LodeDB |
|---|---|---|---|---|---|---|---|
| **lodedb** | 4,348 | **0.91** | 0.95 | 0.95 | 9.7 ms | **15 MB** | 1.0x (baseline) |
| qdrant (default) | 1,204 | 26.3 | 1.00 | 1.00 | 0.7 ms | 70 MB | **4.6x bigger** |
| faiss | 25,155 | 0.50 | 1.00 | **0.04** | 153 ms | 30 MB | 1.9x bigger |
| chroma | 1,582 | 4.06 | 1.00 | 1.00 | 7.7 ms | 47 MB | 3.1x bigger |

## Reading the results

- **Footprint.** LodeDB's 4-bit quantized store is the smallest durable footprint
  in every framework: about 7x smaller than `InMemoryVectorStore`, 5x smaller than
  `SimpleVectorStore`, and 4.6x smaller than mem0's default Qdrant. Chroma is the
  heaviest on disk. This is what it costs to keep a growing agent memory persisted.
- **Durable single add.** The in-memory defaults are fast to *fill* but cannot
  durably add one memory without rewriting the entire store. At ~17.5k docs that is
  about 10.5 s per single add for `InMemoryVectorStore` and about 22 s for
  `SimpleVectorStore`, and it grows O(corpus). LodeDB appends an O(changed) delta in
  about 16 ms, flat as the store grows. FAISS shares the full-rewrite model; Chroma
  and Qdrant, like LodeDB, persist incrementally.
- **Query latency.** The in-memory defaults scan in pure Python, so query p50 is
  340 to 420 ms at this corpus size, versus under 1 ms for LodeDB. LodeDB is also
  20 to 40x faster than the *local-mode* Qdrant client (which is unindexed; server
  Qdrant would differ).
- **Recall.** LodeDB returns 0.95 recall@10 (4-bit quantization) versus 1.00 for
  the exact and flat stores, the deliberate trade for footprint and write cost.
- **mem0 filtered search.** mem0's FAISS provider has no server-side filtering: it
  over-fetches only `2*k` then post-filters, so within-user recall collapses to
  0.04 at 2% selectivity (and its `update` rebuilds the index, ~150 ms). LodeDB,
  Qdrant, and Chroma push the `user_id` predicate into the index and stay accurate.

## Caveats

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
