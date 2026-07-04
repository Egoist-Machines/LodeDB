# Embedding megakernel spike (issue #67)

A metrics-only spike that runs the gate from
[issue #67](https://github.com/Egoist-Machines/LodeDB/issues/67) ("fused single-query MiniLM
embedding megakernel") on real hardware, so the moonshot can be accepted or rejected on data
rather than intuition. The spike itself only measures (no storage-format change); acting on its
result, it also ships the low-maintenance lever it surfaced as an opt-in embedding runtime (see
"Shipping the compiler path" below), rather than the hand kernel the issue gated.

The idea in the issue: for a single query, embedding dominates end-to-end latency while the
TurboVec scan is already sub-millisecond, so the embedding forward pass is the remaining
lever. The moonshot is to fuse the entire all-MiniLM-L6-v2 forward pass into one GPU kernel
for batch 1, keeping activations on-chip across all six encoder layers. The issue is explicit
that this is off-mission (the embedding runtime is deliberately delegated to ONNX Runtime),
carries a large maintenance surface (one kernel per architecture, size, precision, and
backend), and must clear a correctness/recall bar. So it filed a **gate to run before writing
any kernel**, which is what this benchmark implements.

## The gate, and the decision rule

The issue's gate, and how this spike answers each step:

1. **Attribute the single-query latency.** How much is the model forward pass vs
   tokenization, runtime dispatch, and pooling/normalization? If the forward pass is not the
   majority, the ceiling is low.
2. **Check whether an existing compiler already captures the win** with zero custom-kernel
   maintenance (ONNX Runtime graph fusion, CUDA Graphs, TensorRT, `torch.compile`). A
   hand-owned megakernel is justified only if it beats the best of these by a multiplicative
   margin.
3. **Bound the ceiling.** The memory-bandwidth roofline for a batch-1 forward pass is the
   floor any kernel, fused or not, must live above. It bounds the best achievable speedup.

**Decision rule (stated before the results):** recommend building the hand-written megakernel
only if the evidence shows a durable, multiplicative (say >=2x) win over the best
low-maintenance compiler path, *and* that win survives at the end-to-end `embed_query` level
(tokenization, pooling, and the device-to-host copy are not fused away). Anything less does
not justify owning one kernel per architecture, precision, and backend against the recall bar.

## What it measures

**Named baselines.** The issue cites two M1 numbers, 5.73 ms ("CPU embed") and 8.42 ms ("MPS
embed"). In this repo an ONNX `device="mps"` request routes to the CPU provider (Core ML is
opt-in because it measured slower on the dynamic-shape preset graph), so the 8.42 ms MPS
number is the *torch* path, not ONNX. The baselines are named so each number lands on the
right stack:

| baseline | runtime | device / provider |
|---|---|---|
| `onnx-cpu` | ONNX Runtime | CPUExecutionProvider (the default single-query path) |
| `onnx-coreml` | ONNX Runtime | CoreMLExecutionProvider (opt-in on Apple) |
| `torch-cpu` | sentence-transformers | CPU |
| `torch-mps` | sentence-transformers | Apple GPU (Metal), the 8.42 ms path |
| `onnx-cuda` / `onnx-tensorrt` / `torch-cuda` | ONNX Runtime / torch | NVIDIA GPU (Modal) |

Every baseline is parity-checked: its embedding is cosine-compared to the `onnx-cpu`
reference (all 1.0 on MiniLM), which is the correctness bar a megakernel would also have to
meet.

- **Attribution** (`spike_measure.attribute_*`): warm single-query latency split into
  tokenize / forward (`session.run` or torch `model()`) / pool / normalize / tuple, per
  baseline. ONNX uses the real `ONNXRuntimeEmbeddingBackend` seams; torch uses a manual HF
  pipeline (AutoModel + mean-pool + L2-norm) that is parity-checked against the SDK backend
  before its stage timings are trusted.
- **Compiler comparison** (`spike_measure.ort_optimization_sweep`, `spike_cuda`): the ONNX
  graph at each ORT optimization level (forward latency + optimized node/fused-op counts),
  plus, on CUDA, ORT CUDA Graph replay and `torch.compile(reduce-overhead)`. These are the
  best no-custom-kernel fusion paths, i.e. the bar.
- **Roofline** (`spike_roofline`): gather-aware weight-byte accounting read from the actual
  ONNX initializers (streamed weights vs the word/position/token-type tables that a batch-1
  pass only partially reads), turned into a per-device latency floor.

### Honest-measurement caveats

- The ORT profiler's summed node time is *profiled operator time, not device kernel time*,
  and it is inflated by profiling overhead; `model_run - summed_nodes` is unprofiled span
  (scheduler, allocation, sync), not clean launch overhead. Node counts are *graph nodes
  executed, not GPU kernel launches*. These are labeled as such in the artifacts.
- The roofline is *derived, not measured* (tagged `derived`), and uses vendor peak bandwidth,
  so it is an optimistic floor. That is deliberate: an optimistic floor is a valid *no-go*
  filter (if even the floor leaves little room, stop), not a *go* signal.
- CUDA Graphs and `torch.compile` are not genuine single-kernel fusion (KernelBench-Mega
  flags exactly this). They are the low-maintenance bar the megakernel must beat, not the
  megakernel.
- Artifacts are metrics-only: latencies, counts, cosine similarities, versions. No text,
  token ids, embeddings, or raw profiler traces are written (traces are parsed then deleted).

## Run it

```bash
# Local (Apple Silicon / any CPU). Needs the embedding extras + onnx for the roofline:
#   uv sync --extra dev --extra embeddings --extra torch && uv pip install onnx
python benchmarks/embedding_megakernel_spike/run.py           # -> results/spike_<machine>.json
python benchmarks/embedding_megakernel_spike/run.py --iters 200     # more samples

# CUDA, on Modal (A10 / L40S):
modal run benchmarks/embedding_megakernel_spike/modal_bench.py::smoke_a10
modal run benchmarks/embedding_megakernel_spike/modal_bench.py::a10
modal run benchmarks/embedding_megakernel_spike/modal_bench.py::l40s
```

`onnx` is an analysis-only tool for this spike (weight-byte accounting and optimized-graph
node counts), not a LodeDB dependency, like `matplotlib` for the other benchmarks' diagrams.

## Results

All numbers are `measured` p50 of warm single-query embedding, MiniLM (all-MiniLM-L6-v2,
384-dim). Single-run, single-machine; re-measure on your own hardware.

### Laptop (Apple M1, macOS arm64), the regime the issue cites

| baseline | `embed_query` p50 / p95 (ms) | forward-pass p50 (ms) | parity |
|---|---:|---:|---:|
| `onnx-cpu` (default) | 3.01 / 4.30 | 2.37 | 1.0 |
| `torch-cpu` | 8.12 / 11.77 | 6.13 | 1.0 |
| `torch-mps` (the 8.42 ms path) | 8.37 / 9.57 | 7.53 | 1.0 |
| `onnx-coreml` | 26.43 / 41.79 | 23.79 | 1.0 |

The forward pass is the large majority of `embed_query` on every baseline (roughly 70% of the
fast `onnx-cpu` path up to ~97% of the slower ones); tokenization is ~1-2% and pooling +
normalization + tuple conversion well under 1%. ORT graph optimization takes the CPU forward
graph from 562 nodes (`ORT_DISABLE_ALL`, 3.53 ms) to 284 nodes with fused
`LayerNormalization`x13 and `BiasGelu`x6 (`ORT_ENABLE_ALL`, 2.58 ms), a ~1.37x win for free.
The Apple GPU (`torch-mps`) forward is ~3x *slower* than the `onnx-cpu` forward and
dispatch-bound.

### Server GPU (Modal), the regime a megakernel would target

`embed_query` p50 through the SDK (`torch-compile-cuda` is the shipped opt-in path, pad 32):

| GPU | `onnx-cuda` | `onnx-tensorrt` | `torch-cuda` | `torch-compile-cuda` | `torch.compile` forward (eager->compiled) | fp16 roofline floor |
|---|---:|---:|---:|---:|---:|---:|
| A10 | 1.384 ms | 1.314 ms | 4.134 ms | **1.097 ms** | 2.70 -> 0.70 ms (3.85x) | 0.036 ms |
| L40S | 1.172 ms | 1.195 ms | 2.889 ms | **0.812 ms** | 1.67 -> 0.43 ms (3.91x) | 0.025 ms |

All CUDA baselines are parity 1.0 against the `onnx-cpu` reference. ORT graph optimization alone
gives ~1.4x on the A10 and ~1.6x on the L40S (`ORT_DISABLE_ALL` -> `ORT_ENABLE_ALL`). ORT CUDA
Graph capture is **blocked** on this graph: the dynamic-shape export inserts `Memcpy` nodes that
are not partitioned to the CUDA provider, so `torch.compile` and TensorRT are the viable
low-maintenance fusion paths.

The forward-only `torch.compile` gain is ~4x, but the shipped `embed_query` path wins only
**1.26x (A10) / 1.44x (L40S)** end to end: the ~4x is the forward pass alone, while `onnx-cuda`
already uses dynamic padding + ORT fusion, and the compiled path's fixed pad plus tokenization
and the device->host copy absorb most of the forward gain. The end-to-end win grows as queries
lengthen toward the pad; for short queries it is modest.

### Roofline

MiniLM is 90.3 MB of fp32 weights, but at batch 1 only 42.6 MB is *streamed* every forward
pass; the 47.7 MB of word/position/token-type tables are gathered (a few rows read, not the
whole table). Dividing the streamed bytes by peak bandwidth gives the fp16 floor a perfect
memory-bound kernel would approach: M1 GPU 0.31 ms, A10 0.036 ms, L40S 0.025 ms, H100
0.006 ms.

## Findings and recommendation

**Gate step 1: the forward pass is the lever.** Across every runtime and device, the model
forward pass is the large majority of warm single-query latency; tokenization is low
single-digit percent and pooling + normalization + tuple conversion are a fraction of a
percent. Folding pooling/normalization into the ONNX graph (a possible micro-optimization) is
therefore not worth it: it targets under ~2% of the total. So the forward pass is a real
lever; gate step 1 passes.

**Gate step 2: a low-maintenance compiler already captures the win.** On CUDA,
`torch.compile(reduce-overhead)` (Inductor fusion plus CUDA graphs, no custom kernel) cuts the
forward pass ~4x on the A10 and ~4.5x on the L40S on its own, and the TensorRT execution
provider (`onnx-tensorrt`) runs the same graph through a tuned compiler at the fastest measured
ONNX forward. Even plain ORT graph optimization, which fuses LayerNorm/Gelu and roughly halves
the node count, gives ~1.4x on the CPU and ~1.4-1.6x on CUDA for free. The point of the gate
holds: the multiplicative speedup a megakernel would chase is *already available* from an
existing compiler with zero custom-kernel maintenance.

**Gate step 3: the ceiling is real but not a green light.** The batch-1 memory floor sits well
below the best measured compiler path (e.g. A10: 0.036 ms floor vs 0.67 ms compiled forward),
so on paper there is headroom. But at batch 1 the encoder's GEMMs are tiny (a handful of rows),
so the pass is latency/launch-bound, not bandwidth-bound; that is the regime where keeping
activations on-chip (the megakernel's actual mechanism) saves little, because activation
traffic is already dwarfed by the irreducible weight streaming that no fusion can avoid. And a
megakernel only touches the forward: tokenization and the device-to-host copy remain. So the
optimistic floor is a no-go filter, not evidence a hand kernel can reach it.

**Recommendation: do not build the hand-written megakernel.** The spike does not clear the
issue's own bar, a durable multiplicative win over the best low-maintenance compiler path that
survives end-to-end. Concretely:

- On the laptop (the regime the issue cites), the fastest single-query path is `onnx-cpu`. A
  GPU megakernel cannot help CPU inference at all, and the Apple GPU (`torch-mps`) path is
  already slower than the CPU here and dispatch-bound, so there is no laptop win to chase.
- On a server GPU, if single-query embedding latency ever becomes a priority, the lever is
  already available and cheap: `torch.compile` or TensorRT over a fixed/bucketed input shape,
  with no custom kernel to own per architecture, precision, and backend.
- Owning a fused kernel would buy an uncertain, bounded increment over that free path, for one
  model, against the recall bar the issue itself flags. Off-mission for a vector store.

Because the low-maintenance paths already capture the multiplicative win, the gate's step-3
kernel prototype is not warranted; the go/no-go can be settled without one. If MiniLM
single-query GPU latency is ever prioritized, adopt a compiler path first and re-measure.

## Shipping the compiler path: `embedding_runtime="torch-compile"`

Acting on the spike, LodeDB now ships the low-maintenance lever as an opt-in embedding runtime
rather than a hand kernel:

```python
db = LodeDB(path="./data", model="minilm", device="cuda",
            embedding_runtime="torch-compile", max_seq_length=32)
```

It loads the raw HF encoder and runs it through `torch.compile` (CUDA-graph replay on CUDA),
producing embeddings identical to the ONNX/torch runtimes (parity 1.0 on MiniLM, verified in
`tests/test_torch_compile_runtime.py`). Measured `embed_query` speedup vs the default ONNX-CUDA
path: **1.26x on the A10, 1.44x on the L40S** for short queries (the forward pass alone is ~4x;
tokenization and the copy do not compile away). Scope and caveats, all honest:

- **CUDA is where it pays.** Off CUDA it falls back to inductor-only compilation with a smaller
  gain; the laptop default stays `onnx-cpu`, which a compiler-on-GPU does not touch.
- **Keep the pad small.** CUDA graphs need a static shape, so every input is padded to
  `max_seq_length`; a large value (the SDK default is 256) makes short queries slower than
  ONNX's dynamic padding. Set 32-64 for the query fast path (a warning fires above 128).
- **No new dependency:** it rides the existing `lodedb[torch]` tier and imports lazily.

This is the spike's actual conclusion in shipped form: a stock compiler over a fixed shape, not
a per-architecture hand kernel.
