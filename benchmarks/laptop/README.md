# Laptop benchmark — embedding throughput + CPU scan latency

A metrics-only benchmark of the two stages LodeDB does on a laptop: **embedding** (the
accelerated stage on Apple Silicon — the Apple GPU via MPS) and the **CPU TurboVec scan**. It
runs the same measurement as `lodedb benchmark`, once per available embedding device, and
writes one combined results JSON. No document or query text is ever logged — only counts,
bytes, and latency.

There is no GPU vector search on Mac; the scan always runs on the CPU SIMD kernel (NEON).
For the CUDA GPU scan, see [`../gpu_vanilla_vs_augmented/`](../gpu_vanilla_vs_augmented).

## Run

```bash
# from the repo root, with the venv synced (uv sync --extra dev --extra embeddings --extra torch)
python benchmarks/laptop/run.py --docs 20000 --queries 200      # writes results/laptop_m1.json
python benchmarks/laptop/diagrams.py                            # renders docs/*.png + *.svg
```

`run.py` benchmarks every embedding device it finds (MPS → CPU). `diagrams.py` needs
`matplotlib`, which is a dev-only tool, not a LodeDB runtime dependency — install it
separately (`uv pip install matplotlib`).

## What it measures

- **Embedding throughput** (`docs/second`) — `add_many` throughput at embed batch 64, measured
  with the embedding model pre-loaded before timing (`run_local_benchmark` warms the backend
  first). This is warm steady state, not the one-time cold model load.
- **CPU scan latency** (`search_only_ms` p50/p95) — the TurboVec scan plus result
  materialization, with embedding excluded. Device-independent: it is always the CPU kernel.
- **End-to-end query latency** (`end_to_end_query_ms` p50/p95) — embed one query, then scan.
  This is what matters for interactive, one-query-at-a-time serving.

## Results

Measured numbers, the machine block, and the charts live in
[`../../docs/benchmarks.md`](../../docs/benchmarks.md). Re-run on your own hardware for
variance; these are single-run numbers on one machine.
