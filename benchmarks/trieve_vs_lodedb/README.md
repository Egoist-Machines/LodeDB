# trieve_vs_lodedb benchmark

Head-to-head retrieval benchmark: LodeDB (embedded vector store) vs a self-hosted build of
Trieve at its last MIT-licensed commit (`a99b21e2`), both on Modal, with the same dense
embedding model. Metrics only (latency, throughput, recall, nDCG, footprint, counts); no raw
documents, queries, or embeddings are written, matching the repo's benchmark provenance rules
(`../README.md`).

The question: for a corpus that fits on one machine, how does LodeDB compare with the Trieve
stack (Qdrant ANN + SPLADE + cross-encoder rerank) a former Trieve user would be leaving behind?

This run is on **LodeDB 1.2.0**. The first pass of this benchmark (on 1.1.0) flagged two gaps in
LodeDB: its BM25/hybrid latency, and the lack of an approximate index for the large-corpus
exact-scan crossover. Both shipped in 1.2.0 (a native MaxScore BM25 serving index and an opt-in
cluster-prune ANN), so the numbers below are re-measured against that build. The Trieve side is
unchanged (same pinned commit, same results).

## What is compared

Both systems index the SAME pre-chunked corpus and use the SAME dense model
(`all-MiniLM-L6-v2`, 384-d, cosine), so the vector path is apples to apples. Trieve keeps its full
retrieval stack for the quality axis (dense plus SPLADE learned-sparse plus a `bge-reranker-base`
cross-encoder). LodeDB uses its shipping hybrid (BM25 fused with the vector scan via Reciprocal
Rank Fusion, no reranker) and its default 4-bit quantized codes.

| | LodeDB | Trieve |
|---|---|---|
| Vector search | exact SIMD scan (2/4-bit codes), opt-in cluster-prune ANN | Qdrant HNSW (ANN) |
| Lexical | BM25 (MaxScore serving index) | SPLADE learned-sparse (and BM25) |
| Fusion | Reciprocal Rank Fusion | dense+sparse union, cross-encoder rerank |
| Form | in-process library, on-disk | Qdrant + Postgres + Redis + workers + model servers |

## Axes

- Axis A (scale, latency, footprint): GovReport (`ccdv/govreport-summarization`) chunked at 360
  characters. Reports ingest throughput, single-query and batched latency, on-disk footprint, peak
  RSS, index-fidelity recall (LodeDB 4-bit vs its own fp32 brute force), and the opt-in ANN
  cluster-prune latency/recall trade against the exact scan.
- Axis B (retrieval quality): MLDR English (`Shitao/MLDR`, `corpus-en` plus `en/test` qrels).
  Reports doc-level recall@{10,100} and nDCG@10 for vector and hybrid on both.

## How to run (Modal)

LodeDB side:

    modal run benchmarks/trieve_vs_lodedb/modal_bench.py::smoke     # small validation
    modal run benchmarks/trieve_vs_lodedb/modal_bench.py::main      # matched 200k + MLDR-en
    modal run benchmarks/trieve_vs_lodedb/modal_bench.py::scale2m   # LodeDB-only 2M GovReport

Trieve side (builds `trieve-server` and `ingestion-worker` from source with no libtorch, then runs
the whole stack co-located in one GPU container with a Python model server for dense/SPLADE/rerank
plus a static OIDC discovery stub that `trieve-server` requires at boot):

    modal run benchmarks/trieve_vs_lodedb/trieve_modal.py::smoke
    modal run benchmarks/trieve_vs_lodedb/trieve_modal.py::main

Results land in `results/*.json`.

## Fairness controls

- Identical pre-chunked corpus into both (same `chunk_text`, 360 characters).
- Same dense model both sides (Trieve serves it via a local text-embeddings server; LodeDB embeds
  with the same weights).
- Like-for-like modes: LodeDB `vector` vs Trieve `semantic`; LodeDB `hybrid` vs Trieve `hybrid`.
- LodeDB at its shipping 4-bit default; Qdrant at f32, no quantization.
- Latency is read from Trieve's per-phase `Server-Timing` header so the vector-search phase is
  separated from query embedding.

## Results (matched run: 200k GovReport chunks, ~1.5k MLDR docs / 800 queries, one L40S each)

| Metric | LodeDB 1.2.0 | Trieve |
|---|---|---|
| Ingest throughput | 4,599 chunks/s | 85.5 chunks/s |
| Single query, end to end | ~5 ms scan / ~10-13 ms with in-process embed | ~49 ms |
| Batched-256 throughput | ~6,000 qps (CPU scan) | ~25 qps |
| On-disk footprint (200k) | 106 MB (4-bit) | not measured (Qdrant+PG; larger) |
| MLDR vector recall@10 / nDCG@10 | 0.916 / 0.862 | 0.919 / 0.866 |
| MLDR hybrid recall@10 / nDCG@10 | 0.965 / 0.918 | 0.955 / 0.861 |
| MLDR hybrid query p50 | 15.5 ms | 648 ms |
| MLDR lexical (BM25) query p50 | 3.7 ms | — |

LodeDB's vector recall matches Trieve's (same dense model, 4-bit exact ~ Qdrant f32). Its BM25+RRF
hybrid matched or beat Trieve's SPLADE+cross-encoder hybrid on MLDR (recall@10 0.965 vs 0.955;
nDCG@10 0.918 vs 0.861), now at ~40x lower latency.

## What 1.2.0 changed (same benchmark, 1.1.0 vs 1.2.0)

The two gaps the 1.1.0 pass surfaced were fixed, and it shows in the numbers (identical corpus,
identical quality):

| Metric | LodeDB 1.1.0 | LodeDB 1.2.0 | |
|---|---:|---:|---|
| Hybrid query p50 (200k-scale corpus) | 113.5 ms | **15.5 ms** | MaxScore BM25 serving index |
| Lexical (BM25) query p50 | 91.3 ms | **3.7 ms** | (same) |
| Hybrid recall@10 / nDCG@10 | 0.965 / 0.918 | 0.965 / 0.918 | quality unchanged (bit-exact BM25 parity) |
| Exact vector query p50 (200k) | 6.3 ms | **4.9 ms** | ANN-era query-path work |
| Exact vector query p50 (2M) | 131 ms | **45.5 ms** | (same) |

The hybrid speedup is the headline: same fusion, same quality, but the BM25 phase went from the
dominant cost to a few milliseconds, so `hybrid` is now a cheap default rather than a latency
trade. The exact vector scan also got roughly 3x faster at 2M (131 ms to 45.5 ms). Ingest,
footprint, and vector recall are unchanged from 1.1.0.

## LodeDB scale headline (GovReport 2M chunks; Trieve not run at 2M)

| Metric | LodeDB 1.2.0 @ 2M |
| --- | ---: |
| Ingest throughput | 5,386 chunks/s (2M in ~6 min) |
| Footprint | 1.08 GB (4-bit) |
| Index recall@10 (4-bit vs exact) | 0.944 |
| Single-query p50 (exact scan) | 45.5 ms |
| Batched-256 | 0.69 ms/query (1,448 qps, CPU) |

## Approximate search: the exact-scan crossover (opt-in ANN)

The exact scan is the authority (full recall, exact scores), but its single-query latency grows
with the corpus. 1.2.0 adds opt-in cluster-prune ANN (`ann="cluster"`): the query scores cluster
centroids, scans only the nearest clusters, and the exact TurboVec scan re-scores those candidates,
so scores stay exact and only the result set is approximate (a true neighbor in an unprobed cluster
can be missed, hence recall below the exact scan). Each row is a separate store built over the same
vectors (ANN tuning is create-time); recall is measured the same way as the exact row (vs fp32
brute force), so the two are directly comparable.

| Corpus | Config | Cluster build | Single-query p50 | Index recall@10 |
|---|---|---:|---:|---:|
| 200k | exact scan | - | 4.9 ms | 0.958 |
| 200k | ANN cluster (default, ~447 clusters) | 4.4 min | 6.0 ms | 0.944 |
| 200k | ANN cluster (nprobe 64) | 4.4 min | 12.7 ms | 0.955 |
| 2M | exact scan | - | 45.5 ms | 0.944 |
| 2M | ANN cluster (32 clusters, nprobe 8) | 3.0 min | 350 ms | 0.937 |

Two things stand out, and they are the honest result:

1. **At 200k the exact scan wins outright** (4.9 ms, recall 0.958): the corpus is small enough that
   a full SIMD scan beats the cluster-scoring overhead, so ANN is not worth enabling there.
2. **At 2M, ANN does not beat the exact scan under any practical configuration.** This is the
   build-vs-prune tension. Good pruning needs many clusters, but the k-means build is
   single-threaded and grows with `n * clusters`: the default `sqrt(n)` (~1414 clusters) projects
   to ~2.3 h at 2M, and even 256 clusters measured ~78 min (both exceeded a 2 h function timeout).
   A count small enough to build quickly (32 clusters, ~3 min) prunes so coarsely that each query
   scans ~1/4 of the corpus through the cluster postings, and with the gather/indirection overhead
   that path is **~7.7x slower than the exact scan (350 ms vs 45.5 ms) at slightly lower recall**
   (0.937 vs 0.944). Beating the exact scan would need enough clusters (~1024+) to scan well under
   ~65k candidates per query, but that build is in the impractical regime. Until the cluster build
   scales (issue #71) and the probed-candidate path tightens (issue #60), the exact scan (now 45.5
   ms at 2M, ~3x faster than 1.1.0's 131 ms) is the path at this scale; ANN cluster-prune helps
   only where the exact scan is far slower than LodeDB's is here.

## Reading the latency numbers

Trieve's `Server-Timing` shows a semantic query as roughly 47 ms query embedding plus ~1 ms Qdrant
search plus ~1 ms Postgres. Two consequences:

- Trieve's actual vector search (Qdrant ANN) is about 1 ms and stays flat as the corpus grows,
  faster than LodeDB's exact scan on the pure search phase. That is the regime LodeDB's opt-in ANN
  is meant for, with the build-cost caveat above.
- Trieve's end-to-end latency is dominated by the query-embedding round trip, which is where
  LodeDB's in-process design wins end to end (no network hop). That embed number is partly a
  harness artifact (a single-query HTTP call to an unoptimized model server); a production Trieve
  with a batched embedding server would be faster.

Note on the vector scan device: LodeDB's GPU-resident scan JIT-compiles its kernel through NVRTC,
which the stock `pytorch-cuda-runtime` image does not ship, so the scan runs on the CPU SIMD kernel
here (the result is byte-identical; only throughput differs). The batched-throughput figures above
are therefore CPU, not GPU, and this is unchanged from the 1.1.0 run (the GPU scan code is identical
across the two). A GPU-scan number needs an image that provides libnvrtc.

Net: LodeDB wins end to end at small-to-mid scale, on ingest, footprint, and batch throughput, and
its hybrid is now within a few milliseconds of its vector latency. Its exact scan got ~3x faster at
2M (45.5 ms), which moves the crossover with a dedicated ANN store further out. A scale-flat ANN
store (Qdrant) still wins the pure search phase past a few million vectors; LodeDB's own opt-in ANN
does not yet reach that regime (it loses to its own exact scan at 2M for the reasons above), so the
honest boundary is unchanged from 1.1.0: LodeDB is the small-to-mid, in-process, own-your-data
choice, not a billion-scale ANN engine.

## Caveats

- One matched scale (200k) and one quality dataset (MLDR). MLDR is lexically friendly (BM25 alone
  reaches recall@10 0.951), which flatters RRF hybrids; a semantic-heavy corpus or a stronger
  reranker could favor Trieve.
- Trieve ran with `bge-reranker-base` and MiniLM, the lighter end of its stack.
- The 2M row is LodeDB only (a 2M Trieve ingest at ~85 chunks/s was not run to completion).
- LodeDB's vector scan ran on the CPU here (see the latency note); GPU-scan throughput is not
  measured in this run.
- ANN recall is below the exact scan by construction, and its cluster build does not yet scale to
  the multi-million-vector regime (#71); the exact scan remains the default and the authority.

## Layout

- `lodedb_bench.py`, `modal_bench.py`: LodeDB side.
- `trieve_bench.py`, `trieve_modal.py`, `trieve_stack/`: Trieve side (model server, orchestrator,
  self-tests).
- `results/`: metrics JSON.
