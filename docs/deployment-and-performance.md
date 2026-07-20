# Deployment and performance

This is the operational reference for running LodeDB in production: how to put embedding on the
GPU, the constructor knobs that move performance, which model each preset alias maps to, the
dependency ranges LodeDB is tested against, and the gotchas worth knowing before they bite. It is
written to be acted on directly: where a step is conditional it says "if X, install Y", and every
GPU claim has a command to verify it.

If you only read one thing: the default `onnxruntime` wheel is CPU-only, so on a CUDA machine
embedding runs on the CPU (roughly 10 to 50 times slower) until you install `onnxruntime-gpu`.
LodeDB now logs a warning when this happens, and `lodedb doctor` flags it, but it is the single
most common misconfiguration, so it leads this page.

## Two GPU paths, kept separate

LodeDB uses the GPU in two independent places. Do not conflate them:

- **Embedding on the GPU** (this page): turning text into vectors with ONNX Runtime or PyTorch on
  CUDA or Apple MPS. Relevant whenever LodeDB owns the embedder (`model="minilm"`, etc.).
- **The GPU-resident vector scan** (`lodedb[gpu]`, Linux/CUDA): an fp16 copy of the index scored on
  the GPU for high-throughput batched search. This is about search, not embedding, and is covered
  in the README under [GPU-resident index](../README.md#gpu-resident-index).

Bring-your-own-vector indexes (`open_vector_store` / `add_vectors`) do no embedding, so the
embedding GPU story below does not apply to them; only the vector-scan path does.

## Running embedding on the GPU

### NVIDIA (CUDA)

The default `onnxruntime` wheel is CPU-only. To embed on an NVIDIA GPU with the ONNX runtime
(the default), install the GPU build of ONNX Runtime and make sure the device resolves to CUDA:

```bash
pip install "lodedb[embeddings]"      # brings the CPU-only onnxruntime by default
pip install onnxruntime-gpu           # replace it with the CUDA build (not a declared dependency)
```

Then either let `device="auto"` detect the GPU or ask for it explicitly:

```python
from lodedb import LodeDB

db = LodeDB("./data", model="minilm", device="cuda")   # or device="auto"
```

Two things decide whether this actually uses the GPU:

1. **ONNX Runtime must expose `CUDAExecutionProvider`.** That is what `onnxruntime-gpu` adds. With
   the CPU-only wheel it is absent, so the provider list filters down to CPU and embedding runs on
   the CPU. LodeDB logs a warning at open in this case and reports `effective_device="cpu"` on
   `db.embedding_resolution`.
2. **`device="auto"` detects the GPU through PyTorch.** If the PyTorch tier is not installed,
   `auto` cannot see the GPU and resolves to CPU even when `onnxruntime-gpu` is present. Either pass
   `device="cuda"` explicitly, or install `lodedb[torch]` so `auto` can detect the card.

For the smoothest setup on Linux/CUDA, install both `onnxruntime-gpu` and the CUDA PyTorch build
(`lodedb[torch]`): when the CUDA provider is selected, LodeDB calls ONNX Runtime's
`preload_dlls(cuda=True, cudnn=True)` and warms `torch.cuda`, so ONNX Runtime finds the CUDA and
cuDNN libraries that the PyTorch wheel bundles without a separate system CUDA install.

To force the PyTorch runtime instead of ONNX, pass `embedding_runtime="torch"`.

### Apple Silicon (MPS)

Install the PyTorch tier and select the `torch` runtime so embedding runs on Metal:

```bash
pip install "lodedb[embeddings,torch]"
```

```python
db = LodeDB("./data", model="minilm", device="mps", embedding_runtime="torch")
```

`device="mps"` runs the PyTorch sentence-transformers backend on Metal, but you have to ask for the
`torch` runtime to get there. With the default `embedding_runtime="auto"` and the ONNX extra
installed, LodeDB prefers ONNX Runtime, whose Apple Core ML provider is **off by default**: on the
preset graphs it fragments into many Core ML/CPU partitions and measured slower than the plain CPU
provider for single-query embedding (about 16 ms vs 3 ms on an M-series CPU). So `auto` on Apple
Silicon embeds on the CPU. Pass `embedding_runtime="torch"` for Metal, or opt into Core ML for the
ONNX path with `LODEDB_ONNX_COREML=1`.

### Windows (NVIDIA)

On Windows, PyPI serves the CPU-only PyTorch build by default, and no package metadata can redirect
which wheel pip resolves. `lodedb doctor` detects a CPU-only PyTorch on Windows and prints the fix;
`lodedb doctor --fix` reinstalls the CUDA build. See the README
[Windows: NVIDIA GPU embeddings](../README.md#install) note for the exact commands.

### Confirm the embedder is actually on the GPU

Three independent checks, cheapest first:

```bash
lodedb doctor
```

Read the "Embedding" block. `CUDA available : True` with `onnx providers : CUDAExecutionProvider,
CPUExecutionProvider` means the GPU is wired up. If you instead see `CUDA available : True` with
`onnx providers : CPUExecutionProvider`, doctor prints a `!` warning line: the machine has a GPU
but ONNX Runtime is CPU-only, so install `onnxruntime-gpu`.

Directly, ONNX Runtime should list the CUDA provider:

```python
import onnxruntime
print(onnxruntime.get_available_providers())
# expect 'CUDAExecutionProvider' in the list; if it is not, you have the CPU-only wheel
```

From an open handle, `embedding_resolution` reports what was selected:

```python
db = LodeDB("./data", model="minilm", device="cuda")
print(db.embedding_resolution.to_dict())
# effective_device='cuda', fallback_used=False  -> on the GPU
# effective_device='cpu',  fallback_used=True   -> CUDA requested, fell back to CPU (no GPU wheel)
```

## Performance knobs

All of these are `LodeDB(...)` constructor arguments. Defaults are tuned for a correct,
low-latency single-process store; change them when the note applies.

### Native extension build profile

Benchmark and production numbers require the release-profile native extension. Published
`pip install lodedb` wheels and source builds through `uv sync` or `pip install -e .` use a
release build by default. Running `maturin develop` without `--release` produces a debug kernel
that is roughly 100x slower. Run `lodedb doctor` to see the profile of the extension currently
loaded by Python.

| Argument | Default | What it controls | Change it when |
| --- | --- | --- | --- |
| `device` | `"auto"` | Embedding device: `auto` / `cpu` / `mps` / `cuda`. Affects embedding only, not the vector scan. | You have a GPU (see above), or want to pin CPU for reproducibility. |
| `embedding_runtime` | `"auto"` | `auto` (prefer ONNX, fall back to PyTorch) / `onnx` / `torch` / `torch-compile`. | Force `onnx` for lowest single-query latency, `torch` for the PyTorch path / CLIP / MPS, or `torch-compile` for a `torch.compile`d encoder on an NVIDIA GPU (see below). |
| `embedding_dtype` | `"float32"` | `float32` / `float16` / `bfloat16`. Honored only by `torch-compile` (rejected otherwise). | Pass `float16` on an NVIDIA GPU to roughly halve the embedding forward's weight traffic (about 1.7x single-query, up to ~1.8x bulk vs ONNX-CUDA), within cosine 0.999 of fp32. |
| `batch_size` | `32` | Texts embedded per forward pass. | Raise it for throughput on a GPU or for large batched `search_many`; lower it under memory pressure. |
| `max_seq_length` | `256` | Token budget per document before truncation. | Raise for long documents whose tail carries meaning; lower to embed faster. |
| `chunk_character_limit` | `900` | Characters before a document is split into chunks. | Raise to keep long documents as one chunk (fewer duplicate-id hits, see gotchas); lower for finer-grained retrieval. |
| `ann` | `None` (exact scan) | `"cluster"` opts into IVF-style cluster pruning with exact re-score. Create-time only. | The corpus is large enough that the full exact scan is the query bottleneck. Small and mid-size corpora should stay exact. |
| `ann_nprobe` | corpus-derived | Clusters probed by ANN. Persisted at creation; supplied on reopen it is a session-only override. | For large corpora, measure 64 to 256 as a starting range. More probes cost scan work; probe-all is exact. |
| `rescore` | `None` | `"original"` captures source vectors for compact-scan candidates to be re-ranked by fp32 dots. Create-time only. | Close-result ordering is limited by 4-bit codes and the sidecar footprint is acceptable. `float16` costs about 2 bytes per dimension per vector. |
| `rescore_oversample` | `4` | First-stage candidate multiplier for a rescore-enabled store. Supplied on reopen it is session-only. | For large corpora, measure 2 to 8. More candidates can improve final ordering and add sidecar read work. |
| `commit_mode` | `"wal"` | `wal` (append per mutation, checkpoint periodically) / `generation` (publish an MVCC generation per commit). | Keep `wal` for low-latency single-process writes; use `generation` when lock-free readers must see every uncheckpointed write. |
| `durability` | `"fast"` | `fast` (atomic rename) / `fsync` (fsync each file and directory on commit). | Use `fsync` when you need power-loss durability and can trade commit throughput. |
| `compression` | `True` | zstd-compress the retained raw-text store. Create-time only. | Rarely; leave on unless you have measured a reason not to. |

`store_text` and `index_text` are capability switches, not performance knobs: they decide whether
`get`/`get_texts` and `mode="hybrid"`/`"lexical"` are available. See the constructor docstring for
their exact semantics. The docstring is authoritative for the full argument list.

Rescore does not make an ANN result globally exact: it makes the final scores and ordering exact
among the candidates that survived the compact first stage. A store created without `rescore` has
already discarded the original vectors at first ingest, so enabling it later requires rebuilding.
Mode and dtype must match on reopen; only the candidate multiplier is an engine-lifetime override.

## Model aliases

The `model=` preset is a LodeDB alias for a specific Hugging Face model and index shape. Weights are
pulled from Hugging Face on first use and cached.

| Alias | Hugging Face model | Dim | Pooling | Notes |
| --- | --- | --- | --- | --- |
| `minilm` (default) | `sentence-transformers/all-MiniLM-L6-v2` | 384 | mean | Fast general-purpose default. |
| `bge` | `BAAI/bge-base-en-v1.5` | 768 | cls | Higher quality, larger vectors; applies a query prefix internally. |
| `clip` | `sentence-transformers/clip-ViT-B-32` | 512 | n/a | Image and text in one shared space. Needs the `[image]` extra and the PyTorch tier. See [multimodal.md](multimodal.md). |

All presets store 4-bit TurboVec codes. For any other model or dimension, pass your own
`embedder=` (an `EngineEmbeddingBackend`) or open a vector-only index with `vector_dim=` and bring
precomputed vectors.

## Dependency compatibility

Built-in embedding is opt-in. `pip install lodedb` is a dependency-light vector store (numpy, typer,
pyyaml); the embedding runtimes come from extras. The heavy numeric and ML dependencies are capped
below the next untested major, so a future major cannot silently resolve into an install and change
behavior or memory use.

| Package | Extra | Declared range | Tested (uv.lock) |
| --- | --- | --- | --- |
| numpy | base | `>=2.0.0,<3` | 2.4.6 |
| onnxruntime | `embeddings` | `>=1.20.0,<2` | 1.27.0 |
| transformers | `embeddings` | `>=4.40.0,<5` | 4.53.3 |
| sentence-transformers | `torch` | `>=3.0.0,<5` | 4.1.0 |

`onnxruntime-gpu` (CUDA) and `cupy-cuda12x` (the `gpu` extra, for the vector scan) are not part of
the tested lock resolution; install them for your CUDA version as shown above. When a new major of a
capped package is validated, raise the cap in `pyproject.toml` and re-run `uv lock`.

The `sentence-transformers` cap is tighter than the others for a concrete reason. Embedding about
300 queries with the `bge` preset used roughly 67 GB of RSS on `transformers` 5.12.1 /
`sentence-transformers` 5.6.0 (an H100 host: GPU idle, about 27 cores pegged, no progress after 24
minutes), versus about 21 GB for the same `bge` workload pinned to the 4.x majors. There was no
error and no out-of-memory message, and the cause was never traced to a specific change in the 5.x
line (the comparison also spanned two machines, so treat the exact numbers as indicative). For a
regression this silent, a stated known-good range plus an install cap is the only practical guard,
so `sentence-transformers` and `transformers` are both held on 4.x here. This is separate from
running on the CPU (see [Running embedding on the GPU](#running-embedding-on-the-gpu)): that same
4.x `bge` run was still CPU-bound and slow until `onnxruntime-gpu` was installed. Two independent
failure modes, two independent fixes.

## Patterns

### One model, many per-tenant indexes

For multi-tenant isolation, give each tenant its own LodeDB path and open one handle per tenant:

```python
from lodedb import LodeDB

# One isolated store per tenant, same preset across them.
tenants = {name: LodeDB(f"./tenants/{name}", model="minilm") for name in ("acme", "globex")}
tenants["acme"].add("acme onboarding notes", metadata={"doc": "welcome"})
tenants["globex"].search("onboarding", k=5)
```

Each store is a separate path with its own writer lock, index, and text, so tenants never share
data. This loads the embedding model once per handle. To load the model once and reuse it across
all tenants, build one embedder and pass it as `embedder=` to each `LodeDB`; a custom `embedder=`
must expose `native_dim` and a non-secret `required_model_name`, which is pinned into each index
header and re-enforced on reopen.

### LodeDB as the store, or as an index over your own store

With `store_text=True` (the default) LodeDB keeps the original text, so a search result is
self-contained: recover the text with `db.get(hit.id)` and you need no second datastore.

```python
for hit in db.search(query, k=5):
    print(hit.id, hit.score, db.get(hit.id))   # LodeDB is the store of record
```

If you already have a canonical store (a relational database, object storage), open LodeDB with
`store_text=False`: it retains no raw text, search returns ids and metadata, and you hydrate the
content from your own store by id. `hit.metadata` is populated either way, so a small payload can
ride along on the hit without a second lookup.

## Operational gotchas

### A chunked document can appear multiple times in search results

A document longer than `chunk_character_limit` (default 900 characters) is split into chunks, and
every search mode scores chunks. So one long document can appear in the results more than once, each
hit carrying the **same** `id` (the document id) with a different per-chunk `score`. If you want one
row per document, dedupe by `hit.id`, keeping the first (best-scoring) occurrence:

```python
seen, unique = set(), []
for hit in db.search(query, k=50):
    if hit.id in seen:
        continue
    seen.add(hit.id)
    unique.append(hit)
```

`k` counts chunk hits, so request a larger `k` when long documents are expected. `db.get(id)` and
`db.get_texts(ids)` return the reassembled full document text regardless of how it was chunked.

### Reopen re-enforces the index identity

On reopen, the persisted index identity (embedding model, dimension, provider, task, storage
profile, bit width) is re-enforced, and the effective `store_text` / `index_text` flags must match
what the path was written with. Reopening a path with a different `model` (or
`embedder` / `vector_dim` / `bit_width`) raises rather than silently rescoring. There is no in-place
format conversion: changing any of these means a fresh path and a reindex. In particular, a
vector-only path cannot be reopened as a text-in preset index.

### One writer per path; readers take no lock

A writable handle holds a single-writer lock on the path (`<dir>/.lodedb.lock`) for its lifetime, so
a second writer on the same path blocks until the first closes. Read-only handles
(`LodeDB.open_readonly(path)` or `read_only=True`) take **no** lock, so they can query a path while a
writer holds it. Construct one writable handle per path and share it (it is safe to call from
multiple threads; calls are serialized internally). See the README
[Concurrency and durability](../README.md#concurrency--durability) section for the durability model.

Read-only handles are also cheaper to hold than writable ones: they skip the open-time
dequantization of the committed vectors and serve queries from the quantized index (and the
original-precision rescore sidecar, when configured), so a read-only open neither pays the
full-corpus f32 reconstruction nor keeps those copies resident. A serving tier that opens one
read-only handle per tenant should expect memory to scale with the quantized index, not with
`chunks x dim x 4` bytes.

### The index directory is LodeDB-owned

A LodeDB path is a directory of files LodeDB manages (`.tvim` / `.tvd` / `.tvtext` / `.tvlex` / WAL /
commit manifest). Point each store at its own directory and do not drop unrelated files into it or
share it with another store.
