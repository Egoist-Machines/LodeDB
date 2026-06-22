# Filtered batch-search benchmark

Quantifies the filtering asymmetry in the multi-query path and the
allowlist-pushdown fix for it.

Before the fix, an unfiltered `search_many` rode the GPU-resident scan, but a
*filtered* one widened the effective `top_k` to the corpus size and
post-filtered — which tripped the resident `top_k` cap
(`GPU_DIRECT_TURBOVEC_MAX_TOP_K = 4096`) and silently bypassed the GPU to the
CPU kernel, scaling O(corpus) per query. After the fix, the filter is pushed
into the scan as a shared allowlist (in-kernel on CPU, an `-inf` score mask on
GPU/MPS), so `top_k` stays `k` and filtered batches stay on the fast path.

The harness sweeps `(gpu_policy × batch_size × {unfiltered, selective,
non-selective})`, capturing latency and the redacted `query_batch_completed`
telemetry — `gpu_stage_one_status` / `gpu_fallback_reason` — so the cliff (and
its closure) is *proven*, not just timed. It also records the host CPU ISA
(AVX2 vs AVX-512, which Modal varies per run) and the kernel's own backend
label.

## Local (CPU only)

```bash
python benchmarks/filtered_batch/filtered_batch.py
```

## Modal (real CUDA)

`modal` lives in the efficient-embeddings venv; the image is built from the
local `src/` tree, so run it from the checkout you want to measure (before vs
after the fix):

```bash
~/git/efficient-embeddings/.venv/bin/modal run benchmarks/filtered_batch/modal_bench.py::smoke   # quick A10
~/git/efficient-embeddings/.venv/bin/modal run benchmarks/filtered_batch/modal_bench.py::a10     # full A10
~/git/efficient-embeddings/.venv/bin/modal run benchmarks/filtered_batch/modal_bench.py::l40s    # full L40S
```

Read the `gpu_stage_one_status` column: unfiltered rows report `used`; before
the fix, filtered rows report `bypassed` / `*_top_k_exceeds_limit`; after the
fix they report `used` (resident mask) or, on a CPU-only host, the explicit
`filtered_batch_cpu_allowlist` reason rather than the corpus-wide widen.
