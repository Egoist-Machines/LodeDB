# Direct CUDA GPU Sweep

Launch-proof benchmark for LodeDB's optional GPU-resident direct TurboVec path.

It builds one local LodeDB index, then runs paired CPU/GPU query batches over the same
documents. Batch 1 should bypass GPU; batches >= 2 should use `gpu_cupy_exact_direct`
under the required GPU policy. The benchmark also checks visible `auto` memory fallback,
required-policy fail-closed behavior, CPU/GPU recall parity, and raw-payload-free
persistence.

## Run On Modal

```bash
modal run benchmarks/direct_gpu_sweep/modal_bench.py::smoke
modal run benchmarks/direct_gpu_sweep/modal_bench.py::smoke_a10
modal run benchmarks/direct_gpu_sweep/modal_bench.py::main \
  --out benchmarks/direct_gpu_sweep/results/results_l40s.json
modal run benchmarks/direct_gpu_sweep/modal_bench.py::main_a10 \
  --out benchmarks/direct_gpu_sweep/results/results_a10.json
```

The full runs use GovReport5K-shaped real documents, MiniLM embeddings, A10/L40S CUDA
hosts, and batch sizes `1,2,4,8,16,32,64,128,256,512,1024`.

Measured full-run artifacts are checked in as `results/results_a10.json` and
`results/results_l40s.json`. Both passed the raw-payload audit, batch >= 2 GPU-use
assertion, CPU/GPU recall parity, visible auto memory fallback, and required-policy
fail-closed checks.

## Render Charts

```bash
python benchmarks/direct_gpu_sweep/diagrams.py --out benchmarks/direct_gpu_sweep/docs
```

This writes PNG/SVG charts for GPU scan time, per-query search latency, end-to-end batch
latency, recall parity, and memory/copy accounting.

## Run On A CUDA Host

```bash
python benchmarks/direct_gpu_sweep/direct_gpu_sweep.py \
  --out benchmarks/direct_gpu_sweep/results/results_cuda.json \
  --dataset GovReport5K \
  --query-count 1024 \
  --batch-sizes 1,2,4,8,16,32,64,128,256,512,1024
```

Artifacts contain counts, ids, timings, backend labels, and byte accounting only. Raw
documents and raw queries are not written.
