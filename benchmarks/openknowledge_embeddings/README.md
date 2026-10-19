# openknowledge_embeddings benchmark

Compares OpenKnowledge's default OpenAI embedding provider with LodeDB's local
OpenAI-compatible `/v1/embeddings` endpoint. OpenKnowledge (Inkeep's local-first
Markdown wiki) accepts any OpenAI-compatible embeddings endpoint for semantic search;
its default provider is OpenAI `text-embedding-3-small`. `lodedb serve` exposes the
same wire contract, so an OpenKnowledge user can point `baseUrl` at a local LodeDB server.

The question: *what does an OpenKnowledge user trade by pointing `baseUrl` at LodeDB
instead of OpenAI?*

Metrics only (counts, bytes, latency, provider labels, and scores). No raw documents,
queries, embeddings, or credentials are written.

**Overall:** LodeDB passes OpenKnowledge's pre-registered quality gate unmodified. On
the full wiki, OpenAI indexes 1.4-2.9x faster for bulk indexing on an 8-vCPU quota
(median 1.8x across four pinned LodeDB runs), while LodeDB has roughly 38x lower p50
query latency, zero workload cost, and zero external content egress.

## Axis 1: retrieval quality

This uses OpenKnowledge's own `packages/server/src/embeddings/eval/semantic-eval.ts`
unmodified, with its committed `eval-set.json`: 22 documents, 28 query pairs, 15 tune
pairs, and 13 held-out pairs. The OpenAI and minilm providers were repeated twice, with
identical numbers across repeats.

### Reproduce

From an `open-knowledge` checkout with Bun installed, start LodeDB in another shell:

```bash
lodedb serve --model minilm --port 8093
```

Run the OpenAI provider:

```bash
OK_EMBED_SMOKE=1 OK_EMBEDDINGS_API_KEY=sk-... bun run --conditions=development packages/server/src/embeddings/eval/semantic-eval.ts
```

Run LodeDB through the same harness:

```bash
OK_EMBED_SMOKE=1 OK_EMBEDDINGS_API_KEY=placeholder OK_EMBEDDINGS_BASE_URL=http://127.0.0.1:8093/v1 OK_EMBEDDINGS_MODEL=minilm OK_EMBEDDINGS_DIMENSIONS=384 bun run --conditions=development packages/server/src/embeddings/eval/semantic-eval.ts
```

The BGE row uses `OK_EMBEDDINGS_MODEL=bge OK_EMBEDDINGS_DIMENSIONS=768` against
`lodedb serve --model bge`.

### Results

Held-out split, `n=13`. The shared lexical baseline is MRR `0.4997`, recall@1
`0.3077`, and recall@5 `0.7692`.

| provider | MRR@10 | recall@1 | recall@5 | MRR gain vs lexical (min 0.05) | recall@5 gain (min 0.08) | lexical-strong regression (max 0.03) | their FR2 gate |
|---|---:|---:|---:|---:|---:|---:|---|
| openai text-embedding-3-small (1536d) | 0.7974 | 0.6923 | 0.9231 | +0.2978 | +0.1538 | -0.2500 | PASS |
| lodedb minilm (384d, local) | 0.7526 | 0.6154 | 1.0000 | +0.2529 | +0.2308 | -0.1500 | PASS |
| lodedb bge (768d, local) | 0.7333 | 0.5385 | 1.0000 | +0.2337 | +0.2308 | -0.1167 | PASS |

LodeDB's bge preset (768d) also PASSES the FR2 gate, but scores below minilm through
this API because BGE is an asymmetric model whose query prefix cannot be expressed over
the OpenAI wire. This is why minilm is the documented recommendation.

LodeDB passes their pre-registered quality gate unmodified. The held-out sample has
only 13 pairs, so the per-metric differences are directional, not significant. In
particular, MRR is `0.7526` vs `0.7974`, while recall@5 is `1.0000` vs `0.9231`.
The tune split selected RRF `k=10` for both providers.

## Axis 2: throughput, latency, cost, and egress

### Methodology

`modal_bench.py` clones `kubernetes/website` at corpus commit
`71d23f81e3479361befc94564e2b955860c03164` and runs against Markdown under `content/en`,
sorted by path. It uses faithful ports of OpenKnowledge's chunker and embedder behavior:

- Chunking targets 8,000 characters, overlaps 400 characters, and caps a document at
  80 chunks. The port is parity-tested against the OpenKnowledge `chunking.test.ts`
  cases in `test_okchunk.py`.
- Documents use greedy sequential batches capped at 96 inputs and 96,000 characters.
  Batches are sent sequentially. Queries are sent as one sequential request each.
- Both providers use the same Python HTTP client, request shape, response checks,
  Float32 conversion, and normalization. Every request sends a Bearer header.
- The client uses persistent pooled HTTP connections. The `reconnects` counter is 0
  everywhere in the five committed JSONs. An unpooled pilot measured remote query p50
  at 171.5 ms versus 173.3 ms with pooling, so pooling did not materially change remote
  latency and the comparison is robust to HTTP client behavior.
- The chunker preserves OpenKnowledge's UTF-16 behavior. Lone surrogates are sanitized
  only at the HTTP boundary, immediately before JSON encoding, so UTF-16 parity is
  preserved.
- Document requests have 30 second timeouts and query requests have 8 second
  timeouts. Retryable statuses are 408, 409, 429, 500, 502, 503, and 504, with up to
  4 retries and a 500 ms exponential backoff.
- The first document batch is a warmup and is excluded from workload timing for both
  providers. The LodeDB server start and first batch are reported separately.

Provider-native dimensions are retained: LodeDB uses 384 dimensions and OpenAI uses
1,536 dimensions. No dimension matching is added for the benchmark.

### Hardware and setup

| item | value |
|---|---|
| container | Modal, Ubuntu 22.04, Python 3.11 |
| CPU and memory | CPU=8.0 quota, 16 GB; `os.cpu_count()` reports 24 host cores, but the quota is what matters |
| LodeDB | `lodedb serve --model minilm`, CPU ONNX, loopback, same container |
| OpenAI | measured from a Modal datacenter, a favorable network for the remote provider |

### Both sides measured identically

- The same sorted corpus, chunking, batches, timeouts, retry policy, HTTP client, and
  sequential query loop are used for both providers.
- Workload timing starts after each provider's warmup batch. Retries and request and
  response body bytes are counted for the measured workload.
- Document throughput is measured across all chunks. Query latency is measured across
  100 sequential queries in the full-corpus run and 50 sequential queries in the
  300-document run.

### Reproduce

From the repository root:

```bash
modal run benchmarks/openknowledge_embeddings/modal_bench.py::bench --docs 300 --queries 50
modal run benchmarks/openknowledge_embeddings/modal_bench.py::bench --docs 0 --queries 100
```

The run needs a Modal secret named `openai-embeddings-bench` containing
`OPENAI_API_KEY`. The metrics-only outputs are [`modal-300docs.json`](results/modal-300docs.json),
[`modal-full-corpus.json`](results/modal-full-corpus.json), and the three pinned LodeDB
repeats: [`modal-full-corpus-lodedb-repeat-1.json`](results/modal-full-corpus-lodedb-repeat-1.json),
[`modal-full-corpus-lodedb-repeat-2.json`](results/modal-full-corpus-lodedb-repeat-2.json),
and [`modal-full-corpus-lodedb-repeat-3.json`](results/modal-full-corpus-lodedb-repeat-3.json).

### Full corpus results

The full run contains 2,441 documents, 4,485 chunks, and 22,746,024 characters.
Chunk lengths are p50 6,791, p95 7,999, and max 8,000 characters. 836 documents
produced more than one chunk. Each provider used 257 document batches, with 0 retries
and 0 reconnects on both sides.

| metric | lodedb (minilm, 384d, loopback CPU) | openai (text-embedding-3-small, 1536d) |
|---|---:|---:|
| document embed wall time | 229.4 s median (main 275.2; range 165.8-355.7) | 121.2 s |
| chunks per second | 20.4 median (main 16.3; range 12.6-27.0) | 37.0 |
| batch latency p50 / p95 / max (ms) | 1,047.5 / 1,205.3 / 1,384.0 (main run) | 391.6 / 741.7 / 2,693.7 |
| query latency p50 / p95 / max (ms), 100 sequential queries | 4.5 / 6.5 / 7.7 (main run; p50 range 4.0-4.8 across four runs) | 173.3 / 346.1 / 2,804.8 |
| provider-reported workload tokens | 5,890,209 (local server estimate) | 5,805,063 (provider-metered) |
| cost per full index | $0 | $0.1161 (at $0.020 per 1M tokens) |
| external egress (request + response body bytes) | 0 | 159,504,761 |
| one-time setup | 3.8 s server start + 3.7 s first batch (ONNX session init + model fetch) | none |

### Run-to-run variance

Document embedding throughput on Modal CPU varied about 2x across identical full-corpus
LodeDB runs: 16.3, 27.0, 12.6, and 24.4 chunks/s (median 20.4, range 12.6-27.0).
Wall times were 275.2, 165.8, 355.7, and 183.5 s (median 229.4, range 165.8-355.7).
Query latency did not vary materially: p50 was 4.0-4.8 ms across the four runs. Raw
JSONs for every run are committed.

### 300-document results

This run contains 300 documents, 490 chunks, and 30 document batches.

| metric | lodedb (minilm, 384d, loopback CPU) | openai (text-embedding-3-small, 1536d) |
|---|---:|---:|
| document embed wall time | 27.0 s | 11.1 s |
| chunks per second | 18.1 | 44.1 |
| query latency p50 / p95 / max (ms), 50 sequential queries | 5.6 / 8.7 / 10.3 | 138.9 / 259.9 / 399.1 |
| cost | $0 | $0.0127 |
| external egress (request + response body bytes) | 0 | 18,736,627 |

### Reading the results

- **Bulk indexing.** OpenAI is 1.4-2.9x faster on an 8-vCPU quota than the local CPU
  container (median 1.8x from 37.0 vs 20.4 chunks per second). LodeDB's median full
  corpus wall time was 229.4 s (range 165.8-355.7) versus 121.2 s for OpenAI. Both
  index a 2,441-document wiki in minutes. Indexing is a one-time background operation
  that scales with local cores.
- **Interactive search latency.** For the per-keystroke user-facing operation, LodeDB
  p50 is roughly 38x lower in the main run: 4.5 ms vs 173.3 ms. Its max was 7.7 ms
  versus 2,804.8 ms for OpenAI, including one multi-second outlier. Across the four
  LodeDB runs, p50 remained 4.0-4.8 ms.
- **Cost and privacy.** LodeDB costs $0 and has 0 external request and response body
  bytes. The metered remote API receives the full wiki text and every search query;
  total external request and response body traffic is 159,504,761 bytes, or 159.5 MB,
  for this corpus.
- **Quality.** LodeDB passes OpenKnowledge's pre-registered quality gate. The held-out
  quality sample is small, so the MRR and recall deltas should be treated as
  directional rather than significant.

### Cost and token accounting

The price constant is pinned to [OpenAI API pricing](https://openai.com/api/pricing),
accessed 2026-07-09: `$0.020` per 1M tokens for `text-embedding-3-small`. The cost
calculation uses only the provider-metered OpenAI workload token count.

LodeDB's 5,890,209 workload tokens are a local server estimate;
OpenAI's 5,805,063 are provider-metered. These token columns are not directly
comparable. Warmup usage is excluded from both workload totals and costs.

## Caveats

- The quality result is a pass against the committed OpenKnowledge harness and its
  pre-registered thresholds, not a large evaluation. The held-out split has 13 query
  pairs.
- The OpenAI run was measured from a Modal datacenter, which is favorable to the remote
  provider's network latency. LodeDB's endpoint is loopback in the same container.
- The full-corpus document timing excludes the one-time LodeDB server start, ONNX
  session initialization, and model fetch; those are shown separately in the table.
