# Hybrid search recall benchmark

Measures the exact-token recall win that `mode="hybrid"` exists to deliver. Each probe is an
exact token (an error code, a hyphenated serial, an ISO date) planted in one document's body
among distractors that share no token with it. A content-blind embedding cannot rank the
carrier, so pure vector search misses it; the lexical BM25 ranker isolates it and Reciprocal
Rank Fusion lifts it into the top-k.

The deterministic hash embedding backend is used on purpose: it makes the "embedding cannot see
the literal token" failure mode reproducible without downloading a model, isolating the lexical
contribution. Output is raw-payload-free (recall@k, mean reciprocal rank, and per-query latency
per mode; never tokens or terms).

## Run

```bash
python benchmarks/hybrid/hybrid_recall.py
```

It prints a JSON summary with `recall_at_k`, `mrr`, and `mean_latency_ms` for `vector`,
`hybrid`, and `lexical`. Expect vector recall near zero on the planted tokens and hybrid recall
at 1.0 with the carrier ranked first, quantifying the difference between a usable and a useless
answer for local RAG.
