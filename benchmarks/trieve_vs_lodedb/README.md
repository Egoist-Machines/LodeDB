# trieve_vs_lodedb benchmark

Head-to-head retrieval benchmark: LodeDB (embedded, exact-scan vector store) vs a
self-hosted build of Trieve at its last MIT-licensed commit (`a99b21e2`), both on Modal,
with the same dense embedding model. Metrics only (latency, throughput, recall, nDCG,
footprint, counts); no raw documents, queries, or embeddings are written, matching the
repo's benchmark provenance rules (`../README.md`).

The question: for a corpus that fits on one machine, how does LodeDB compare with the
Trieve stack (Qdrant ANN + SPLADE + cross-encoder rerank) a former Trieve user would be
leaving behind?

## What is compared

Both systems index the SAME pre-chunked corpus and use the SAME dense model
(`all-MiniLM-L6-v2`, 384-d, cosine), so the vector path is apples to apples. Trieve keeps
its full retrieval stack for the quality axis (dense plus SPLADE learned-sparse plus a
`bge-reranker-base` cross-encoder). LodeDB uses its shipping hybrid (BM25 fused with the
vector scan via Reciprocal Rank Fusion, no reranker) and its default 4-bit quantized codes.

| | LodeDB | Trieve |
|---|---|---|
| Vector search | exact SIMD scan (2/4-bit codes), optional GPU batch | Qdrant HNSW (ANN) |
| Lexical | BM25 | SPLADE learned-sparse (and BM25) |
| Fusion | Reciprocal Rank Fusion | dense+sparse union, cross-encoder rerank |
| Form | in-process library, on-disk | Qdrant + Postgres + Redis + workers + model servers |

## Axes

- Axis A (scale, latency, footprint): GovReport (`ccdv/govreport-summarization`) chunked at
  360 characters. Reports ingest throughput, single-query and batched latency, on-disk
  footprint, peak RSS, and index-fidelity recall (LodeDB 4-bit vs its own fp32 brute force).
- Axis B (retrieval quality): MLDR English (`Shitao/MLDR`, `corpus-en` plus `en/test`
  qrels). Reports doc-level recall@{10,100} and nDCG@10 for vector and hybrid on both.

## How to run (Modal)

LodeDB side:

    modal run benchmarks/trieve_vs_lodedb/modal_bench.py::smoke   # small validation
    modal run benchmarks/trieve_vs_lodedb/modal_bench.py::main    # 2M GovReport + full MLDR-en

Trieve side (builds `trieve-server` and `ingestion-worker` from source with no libtorch,
then runs the whole stack co-located in one GPU container with a Python model server for
dense/SPLADE/rerank plus a static OIDC discovery stub that `trieve-server` requires at
boot):

    modal run benchmarks/trieve_vs_lodedb/trieve_modal.py::smoke
    modal run benchmarks/trieve_vs_lodedb/trieve_modal.py::main

Results land in `results/*.json`.

## Fairness controls

- Identical pre-chunked corpus into both (same `chunk_text`, 360 characters).
- Same dense model both sides (Trieve serves it via a local text-embeddings server; LodeDB
  embeds with the same weights).
- Like-for-like modes: LodeDB `vector` vs Trieve `semantic`; LodeDB `hybrid` vs Trieve
  `hybrid`.
- LodeDB at its shipping 4-bit default; Qdrant at f32, no quantization.
- Latency is read from Trieve's per-phase `Server-Timing` header so the vector-search phase
  is separated from query embedding.

## Results (matched run: 200k GovReport chunks, ~1.5k MLDR docs, one L40S each)

| Metric | LodeDB | Trieve |
|---|---|---|
| Ingest throughput | 5,003 chunks/s | 85.7 chunks/s |
| Single query, vector-search phase | 6.3 ms | ~1 ms (Qdrant ANN) |
| Single query, end to end | ~6-13 ms | ~49 ms |
| Batched-256 throughput | ~7,000 qps (GPU) | ~25 qps |
| MLDR vector recall@10 | 0.916 | 0.919 |
| MLDR hybrid recall@10 / nDCG@10 | 0.965 / 0.918 | 0.955 / 0.861 |
| MLDR hybrid latency | 114 ms | 648 ms |
| Footprint (200k) | 106 MB | not measured |

LodeDB-only 2M GovReport: single query 131 ms (exact scan), footprint 1.08 GB, ingest
sustained above 4k chunks/s.

## Reading the latency numbers

Trieve's `Server-Timing` shows a semantic query as roughly 47 ms query embedding plus 1 ms
Qdrant search plus 1 ms Postgres. Two consequences:

- Trieve's actual vector search (Qdrant ANN) is about 1 ms and stays flat as the corpus
  grows, faster than LodeDB's exact scan (6.3 ms at 200k, 131 ms at 2M). On the pure search
  phase, ANN wins, more so at scale.
- Trieve's end-to-end latency is dominated by the query-embedding round trip, which is where
  LodeDB's in-process design wins end to end (no network hop). That embed number is partly a
  harness artifact (a single-query HTTP call to an unoptimized model server); a production
  Trieve with a batched embedding server would be faster.

Net: LodeDB wins end to end at small-to-mid scale, and on ingest, footprint, and batch
throughput; the exact-vs-ANN crossover favors an ANN store past a few million vectors.

## Caveats

- One matched scale (200k) and one quality dataset (MLDR). MLDR is lexically friendly (BM25
  alone reaches recall@10 0.951), which flatters RRF hybrids; a semantic-heavy corpus or a
  stronger reranker could favor Trieve.
- Trieve ran with `bge-reranker-base` and MiniLM, the lighter end of its stack.
- The 2M row is LodeDB only (a 2M Trieve ingest at ~85 chunks/s was not run to completion).

## Layout

- `lodedb_bench.py`, `modal_bench.py`: LodeDB side.
- `trieve_bench.py`, `trieve_modal.py`, `trieve_stack/`: Trieve side (model server,
  orchestrator, self-tests).
- `results/`: metrics JSON.
