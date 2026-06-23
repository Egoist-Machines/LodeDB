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

# Persistence and commit-overhead benchmark

`persist_bench.py` measures the cost of the opt-in durable lexical index (`index_text=True`)
against the default vector-only flow (`index_text=False`). It answers three questions on one
corpus: does enabling it slow the write path, does it change query latency, and does hybrid
survive a reopen with no retained raw text. It records per-incremental-commit latency, query
latency per mode, reopen and load time, on-disk bytes (including the `.tvlex` sidecar), and
exact-token recall. Output is metrics-only: counts, bytes, and latencies, never tokens or text.

## Run locally

```bash
python -c "from benchmarks.hybrid.persist_bench import run_persist_bench; import json; print(json.dumps(run_persist_bench(scale=2000), indent=2))"
```

## Run on a Modal A10

The image builds LodeDB from the local `src/` tree, so run it from a checkout that has the
`index_text` change. The GPU-resident scan serves the batched queries automatically when CuPy
is present, so the vector path runs on real CUDA hardware while the lexical pass and commit
journaling run on CPU.

```bash
modal run benchmarks/hybrid/modal_bench.py::smoke   # tiny validation
modal run benchmarks/hybrid/modal_bench.py::a10     # full 20k-document run
```

Results land in `results/persist_a10.json`. The default config writes zero `.tvlex` bytes (the
on-disk layout is unchanged when the flag is off), the `index_text` config reopens straight from
the persisted tokens with hybrid recall intact and no raw text retained, and the per-commit
overhead from journaling the postings stays small relative to the vector write it rides
alongside.
