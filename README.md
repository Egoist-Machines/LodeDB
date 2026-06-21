# LodeDB

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](pyproject.toml)

**A fast, exact embedded vector database for local RAG: in-process, on-disk, no server.**

*Built by [Egoist Machines, Inc.](https://egoistmachines.com) - efficient full-stack infrastructure
for reliable AI systems.*

Most embedded vector databases stop at the CPU. LodeDB runs the same on-disk index on the
GPU when you have one: batched search hits **24k queries/sec on an A10 and 50k qps on an L40S**,
2.8× to 4.8× the all-CPU ceiling, with recall unchanged. It also persists changed rows
incrementally, so a commit stays **sub-millisecond even at 1M vectors**.

Fast on a laptop. Faster on a GPU. Exact every time. Never phones home.

- **GPU-resident batch search**: an fp16 copy of the index lives on the GPU, scored with a
  tiled GEMM plus a streaming top-k (`[gpu]`, Linux/CUDA). [How it works](#gpu-resident-index).
- **O(changed) persistence**: commits only the rows that changed, 173× to 1,308× faster
  than a full rewrite. [How it works](#delta-persistence).
- **Compact storage**: the MIT [TurboVec](#turbovec) core packs vectors into 2/4-bit codes
  and scans them with SIMD CPU kernels.
- **In-process, on-disk** (`.tvim`/`.tvd`/`.jsd`): no daemon, no account, no API key.
- **Private by default**: text, ids, and vectors stay local; telemetry is metrics-only
  (counts, bytes, latency), never raw payloads.
- **Local embeddings**: `sentence-transformers` on CUDA, MPS, or CPU.
- **Batteries included**: a `lodedb` CLI, a loopback dev server, an MCP server, and a
  LangChain `VectorStore` adapter.

> 🏢 **Enterprise** The LodeDB core is Apache-2.0 and free to use. Enterprise licensing is
> available for commercial support, managed and at-scale serving, and on-prem / BYOC
> deployment. Contact [sales@egoistmachines.com](mailto:sales@egoistmachines.com).

## Install

```bash
pip install lodedb
```

That's it. Prebuilt wheels cover Linux, macOS (Apple Silicon and Intel), and Windows on
Python 3.11+, and bundle the TurboVec (Rust) core, so there's nothing to compile. Confirm
the install with `lodedb doctor`. Optional extras:

```bash
pip install "lodedb[gpu]"            # GPU-resident scan (Linux/CUDA)
pip install "lodedb[mcp,langchain]"  # MCP server + LangChain adapter
```

<details>
<summary><b>Build from source</b> (contributors, or a platform without a wheel)</summary>

Needs a Rust toolchain and a CBLAS provider (Accelerate on macOS, `libopenblas-dev` on
Linux). [uv](https://docs.astral.sh/uv/) builds and bundles the core for you:

```bash
git clone https://github.com/Egoist-Machines/LodeDB && cd LodeDB
uv sync                                 # builds + bundles the TurboVec core via maturin
uv sync --extra mcp --extra langchain   # + MCP server, LangChain adapter
uv sync --extra gpu                     # + GPU-resident scan (Linux/CUDA)
```

Run with `uv run` (e.g. `uv run lodedb doctor`).

</details>

## Quickstart

```python
from lodedb import LodeDB

db = LodeDB(path="./data", model="minilm")   # "minilm" (fast) | "bge" (quality)

fox = db.add("the quick brown fox jumps", metadata={"topic": "animals"})
db.add("a lazy dog sleeps all day", metadata={"topic": "animals"})

for score, doc_id, meta in db.search("fox", k=5):
    print(score, doc_id, meta)

for hits in db.search_many(["fox", "dog"], k=5):   # batched; the GPU can serve this
    print([(h.score, h.id, h.metadata) for h in hits])

db.get(fox)     # -> "the quick brown fox jumps"  (text retained by default)
db.persist()    # durable .tvim/.tvd/.jsd snapshot; replays on reopen
```

Reopen with `LodeDB(path="./data")`; no migration step. Original text is kept in a
`.tvtext` sidecar for `db.get`; pass `store_text=False` to keep none. Presets are `minilm`
(384-dim) and `bge` (768-dim), with weights pulled from Hugging Face on first use. More in
[`examples/`](examples/).

Need to read a store another process is writing to? Open it read-only — it takes no writer
lock, so it never blocks on (or is blocked by) the writer:

```python
reader = LodeDB.open_readonly("./data")   # or LodeDB(path="./data", read_only=True)
reader.search("fox", k=5)                 # reads a committed snapshot
reader.add("nope")                        # raises ReadOnlyError
```

## GPU-resident index

With the `[gpu]` extra on a CUDA host, LodeDB reconstructs the compact index into an fp16
matrix resident on the GPU and scores batched `search_many` with a tiled GEMM plus a
streaming top-k. It is opt-in and lazy: single queries, non-CUDA hosts, and GPU-memory
rejection fall back to the CPU scan, which stays the source of truth.

GPU throughput climbs with batch size while the CPU scan is flat. Same 4-bit index
(d=1536, 100K), same host, only the scoring step differs. Crossover is around batch 50:

| query batch | A10 GPU | L40S GPU |
|---:|---:|---:|
| 1 | 261 q/s | 432 q/s |
| 16 | 3,531 | 5,562 |
| 64 | 11,463 | 18,175 |
| 256 | 19,998 | 39,449 |
| 1024 | **24,037** | **50,326** |

Vanilla TurboVec CPU (all threads) on the same boxes: 8,497 q/s (A10 host), 10,420 q/s
(L40S host). At batch 1024 the GPU is 2.8× / 4.8× that, and it scales with GPU class.

![GPU throughput vs batch size: A10 and L40S vs the vanilla CPU scan](benchmarks/gpu_vanilla_vs_augmented/docs/speed_batch.png)

Recall is unchanged: the GPU scores the exact 4-bit reconstruction, so R@1 tracks the CPU
scan across datasets and bit-widths, and edges ahead on GloVe-200 where quantization error
is largest.

![Recall: vanilla CPU scan vs GPU fp16 reconstruction](benchmarks/gpu_vanilla_vs_augmented/docs/recall.png)

Other in-process vector databases stay CPU-bound. Alibaba's
[zvec](https://github.com/alibaba/zvec) reports about 8.4k q/s (VectorDBBench, 16-vCPU CPU,
Cohere 768-dim): the same class as the TurboVec CPU scan, and a different regime from ours,
so read it as the CPU-class baseline. The GPU-resident path is what clears it.

**Scope.** GPU search is Linux/CUDA-only and opt-in (`[gpu]`). macOS scans on the CPU (the
MPS scan is experimental). See [docs/benchmarks.md](docs/benchmarks.md) and
[docs/architecture.md](docs/architecture.md).

## Delta persistence

Most embedded indexes rewrite the whole file on every change (O(N)). LodeDB writes only the
rows that changed (O(changed)), so a 1,000-row commit stays sub-millisecond at any size:

| corpus | full rewrite | delta export | speedup |
|---:|---:|---:|---:|
| 100K | 42.4 ms | 0.25 ms | 173× |
| 500K | 190.4 ms | 0.24 ms | 782× |
| 1M | 404.9 ms | 0.31 ms | **1,308×** |

![Persist time: full rewrite vs delta export](benchmarks/gpu_vanilla_vs_augmented/docs/update.png)

The GPU path makes reads fast; the delta makes writes cheap. The on-disk format stays a
plain snapshot that replays on reopen.

## Benchmarks

All artifacts are metrics-only (counts, bytes, latency), never payloads. Full methodology
and the complete figure set are in [docs/benchmarks.md](docs/benchmarks.md); each
[benchmarks/](benchmarks/) folder has a README and a one-line reproduction command.

Local is the common case. On an Apple M1 (MiniLM, 20K docs) the CPU scan is ~0.25 ms p50,
and end-to-end single-query latency is 5.7 ms p50.

![Single-query latency on a laptop](benchmarks/laptop/docs/query_latency.png)

## CLI

```bash
lodedb doctor      # capability report: embedding / GPU / TurboVec backend
lodedb index ...   # build / add to an on-disk index
lodedb query ...   # search
lodedb serve       # loopback dev server (127.0.0.1, no auth)
lodedb mcp         # stdio MCP server for agent memory
lodedb benchmark   # local, metrics-only benchmark
```

## Limitations

- **Exact scan, no ANN.** Built for small-to-mid corpora where exact recall matters, not
  billion-scale.
- **GPU is Linux/CUDA-only and opt-in** (`[gpu]`). macOS scans on the CPU; the MPS scan is
  experimental and was slower than NEON on the hardware tested.
- **Single queries run on the CPU**; the GPU serves batched `search_many`.
- **Single writer, many readers, per path.** One handle holds the path open for *writing* at
  a time (an exclusive OS advisory lock); a second writer waits for it to close, then fails
  fast (`ConcurrentWriterError`) after `LODEDB_PERSIST_LOCK_TIMEOUT` (default 30s).
  **Read-only** handles (`LodeDB.open_readonly(path)` or `read_only=True`; used by
  `lodedb query`/`get`) take *no* lock, so they read one consistent committed snapshot **while**
  a writer is open — they just don't auto-see the writer's in-flight changes (no live
  cross-process refresh). Within one process the engine serializes operations under an
  in-process lock, so the threaded `lodedb serve` safely shares one handle. Local filesystems
  only (advisory locks are unreliable on NFS/SMB).
- **Crash-atomic commits.** A commit spans several files, but it is sealed by atomically
  swapping one `<key>.commit.json` root pointer over generation-addressed artifacts, so a
  crash mid-commit rolls back to the last committed generation on reopen (never a torn,
  half-applied store) and readers always load one consistent generation.
- **Durability is `fast` by default.** Commits are *atomic* but not fsync'd. Pass
  `durability="fsync"` (or `--durability fsync` / `LODEDB_DURABILITY=fsync`) to fsync each
  file and its directory on commit for power-loss durability, at some commit-throughput cost.
- **Model weights download from Hugging Face** on first use, then cache locally.

## TurboVec

The compact core is the upstream **MIT** [TurboVec](https://github.com/RyanCodrai/turbovec)
project (© Ryan Codrai), vendored under [`third_party/turbovec/`](third_party/turbovec/)
with its license preserved. LodeDB's lifecycle patches (encoded-row export/import,
`upsert_with_ids`, calibration) are Apache-2.0. See [`NOTICE`](NOTICE).

## License

Apache-2.0 ([`LICENSE`](LICENSE)). The bundled TurboVec core is MIT ([`NOTICE`](NOTICE),
[`third_party/turbovec/LICENSE`](third_party/turbovec/LICENSE)). "LodeDB" and
"[Egoist Machines](https://egoistmachines.com)" are trademarks; Apache-2.0 grants no
trademark rights (§6).

Enterprise licensing and commercial support are available from
[Egoist Machines, Inc.](https://egoistmachines.com): contact
[sales@egoistmachines.com](mailto:sales@egoistmachines.com).

## Contributing & security

PRs welcome; see [`CONTRIBUTING.md`](CONTRIBUTING.md). Report security issues **privately**
per [`SECURITY.md`](SECURITY.md), not in public issues. Other bugs and requests go to the
[issue tracker](https://github.com/Egoist-Machines/LodeDB/issues).
