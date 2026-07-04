# Changelog

All notable changes to LodeDB are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **The opt-in ANN index (`ann="cluster"`) build is parallelized.** The k-means partition that
  `ann="cluster"` builds was single-threaded, so the one-time build grew as `O(n^1.5)` and was
  impractical at the multi-million-vector scale ANN is meant for. The seeding and Lloyd assignment
  passes now run across all cores; the clustering is byte-identical to the previous serial build
  (order-preserving parallelism), so results, determinism, defaults, and recall are unchanged. On a
  32-vCPU host a 200k-vector build drops from ~275s to ~10s and a 2M-vector build from ~2.4h to
  ~4.8min at the default `sqrt(n)` cluster count; pass an explicit `ann_clusters` cap to trade a
  little pruning for a faster build at larger scale.
- **Hybrid search is now the default.** `LodeDB.search` / `search_many` (and the graph
  `semantic_nodes` / `semantic_edges` / `search_subgraph`) default to `mode="hybrid"` when a
  lexical source is available (`store_text=True` or `index_text=True`, both on by default) and
  fall back to `mode="vector"` when neither is present, so the fused BM25 + vector ranking that
  recovers exact tokens (error codes, serials, dates) is the default without ever raising on a
  vector-only store. A default query on a text-retaining store now returns Reciprocal Rank Fusion
  scores rather than raw cosine similarities; pass `mode="vector"` for the previous pure-vector
  behavior. Explicitly requesting `mode="hybrid"`/`"lexical"` with no lexical source still raises
  the same actionable error. The `lodedb` MCP server already defaulted to hybrid, so its behavior
  is unchanged.

## [1.2.0] - 2026-07-02

### Changed

- **Built-in text embedding is now opt-in.** The embedding runtimes (`onnxruntime` +
  `transformers`, and `sentence-transformers`/PyTorch) are no longer base dependencies. The base
  `pip install lodedb` is a dependency-light vector store: bring your own vectors
  (`open_vector_store` / `add_vectors` / `search_by_vector`) or pass `embedder=`. Install the
  built-in embedding stack with the new `lodedb[embeddings]` (ONNX, torch-free) and
  `lodedb[torch]` (PyTorch fallback, CLIP, `device="mps"`) extras. Requesting a preset text model
  without an embedding runtime now raises a clear install hint instead of a deep
  `ModuleNotFoundError` at encode time. Existing installs that relied on `pip install lodedb`
  embedding text out of the box should switch to `pip install lodedb[embeddings]`.
- **Heavy dependency majors are capped to the tested range.** The numeric and embedding
  dependencies now carry upper bounds (`numpy>=2.0.0,<3`, `onnxruntime>=1.20.0,<2`,
  `transformers>=4.40.0,<5`, `sentence-transformers>=3.0.0,<5`), so a future major release cannot
  silently resolve into an install and change behavior or memory use. `sentence-transformers` is
  held at `<5` because the 5.x line paired with `transformers` 5.x showed a large, silent embedding
  memory regression in the field (roughly 67 GB versus 21 GB for the same workload on the 4.x
  majors); see `docs/deployment-and-performance.md`.

### Added

- `lodedb[embeddings]` and `lodedb[torch]` extras for the built-in embedding runtimes, plus a
  `lodedb[all]` convenience extra (embeddings, torch, image, MCP, and the framework adapters).
  The `lodedb[image]` extra now pulls the PyTorch tier itself.
- **GPU embedding fallback is no longer silent.** When the resolved device is CUDA but ONNX Runtime
  exposes no `CUDAExecutionProvider` (the default `onnxruntime` wheel is CPU-only), LodeDB logs a
  warning at index open that embedding will run on the CPU and to install `onnxruntime-gpu`, and
  `lodedb doctor` flags the same mismatch on one line. `embedding_resolution` now reports the
  effective device as `cpu` (and `fallback_used=True`) in this case, so the CLI `index`/`query`
  output and `doctor` agree with what actually ran.
- **`docs/deployment-and-performance.md`** deployment and performance reference: running on the GPU
  (ONNX Runtime and PyTorch, with verification commands), the constructor performance knobs, the
  model-alias table, dependency compatibility ranges, and operational gotchas (chunked-document
  duplicate ids, reopen format identity, single-writer locking).
- README "Two ways to use LodeDB" capability matrix (text-in vs vector-in), plus `batch_size` /
  `max_seq_length` and reopen-identity documentation on the `LodeDB` constructor and a
  chunked-document duplicate-id note on `search`.

## [1.1.0] - 2026-06-30

### Added

- **Swift / iOS bindings.** A native Swift package (`LodeDBCore`) for macOS and iOS over the
  same Rust core, distributed as the
  [`swift-lodedb`](https://github.com/Egoist-Machines/swift-lodedb) SwiftPM package (a
  `LodeDBCoreFFI.xcframework` binary target). It covers durable open/persist, vector / text /
  hybrid search, the full metadata-filter grammar, batched search, late-interaction (MaxSim),
  on-device embedders (Apple `NLEmbedding` out of the box, or an ONNX parity path with a
  persisted model-identity guard), and a `LodeMemory` save/recall/forget agent-memory facade.
  The `.tvim` format is byte-compatible across platforms, so an index built on a server loads
  on a phone. See [swift/LodeDBCore/README.md](swift/LodeDBCore/README.md) and
  [docs/swift-agent-contract.md](docs/swift-agent-contract.md).

### Fixed

- **Atomic batch delete.** `delete_documents` now validates every id before mutating, so a
  blank id later in a batch no longer leaves earlier deletions applied behind a failed request.
- **WAL replay preserves multi-vector patches.** A crash and reopen before a checkpoint now
  restores the late-interaction patch matrix, not just the anchor vector.

## [1.0.0] - 2026-06-29

### Changed

- **The native Rust core (TurboVec) is now the sole engine.** The Python `LodeEngine` and
  `LodeIndex` are removed; every read and write — text and vector ingest, search
  (vector / hybrid / lexical), filters, batch, late-interaction, persistence, and the
  WAL/generation commit — runs through the in-process Rust `CoreEngine`, with no Python
  fallback. The public Python API is unchanged but for one additive change: `list_documents`
  gains `after`/`limit` keyset paging. MPS vector scanning was dropped (MPS embedding via
  PyTorch is unaffected).

### Added

- **Cross-platform single-writer lock.** The writer lock now holds on Windows (an exclusive
  share-mode open of the `.lodedb.lock` sentinel) as well as via the Unix BSD advisory lock,
  so concurrent processes serialize on every platform.
- **Compressed retained text (`compression=`).** The opt-in document-text store is stored
  zstd-compressed by default; pass `compression=False` at create time to keep it uncompressed.
  The choice persists with the store and wins on reopen.

## [0.4.0] - 2026-06-25

### Added

- **Multimodal (CLIP) and bring-your-own-vector search.** A `clip` preset embeds images and text
  into one shared space (`db.add_image()` / `db.add_images()` and cross-modal `search` /
  `search_by_image`) via a sentence-transformers CLIP model behind the optional `lodedb[image]`
  extra (Pillow only, lazy-imported). A public `embedder=` argument drives a text-capable index
  with any embedding backend at its own dimension (it must declare a non-secret
  `required_model_name`), and `LodeCollection` groups named vector spaces (sibling indexes) under
  one root, reopened from a manifest that records each space's kind, identity, bit width, and
  privacy flags (`store_text`/`index_text`) and re-applies them on reopen, so a `store_text=False`
  space never silently flips back to retaining raw text. The collection registry is crash-safe to
  the engine's standard: the manifest honors `durability="fsync"`, and a failed publish rolls the
  space back (closing it, releasing its lock) instead of leaving it open and unregistered. A
  preset, custom-embedder, or vector-only
  index pins its model identity in the on-disk header and re-enforces it on reopen. The raw image
  is never stored (keep its path in metadata), and `store_text=False` now keeps raw text off disk
  in WAL commit mode too: WAL records log chunk embeddings (and lexical tokens when
  `index_text=True`), never the raw document body or a vector caption. A vector or image upsert
  refreshes the document's lexical postings, so replacing a text document clears its stale terms
  and an image caption is searchable when `index_text=True`. Reopen validates the full persisted
  route identity (model, provider, task, dimension, storage profile, and bit width); image decoding
  is bounded by `LODEDB_MAX_IMAGE_PIXELS` (a decompression-bomb guard, ~64 MP default); and
  `stats()` reports per-handle image-embedding metrics split by phase (ingest vs query: count,
  encode time, failures).
- **PrivateGPT vector-store provider.** `lodedb.local.integrations.privategpt` lets
  [PrivateGPT](https://github.com/zylon-ai/private-gpt) use LodeDB as its local vector store.
  PrivateGPT's store layer is LlamaIndex's `BasePydanticVectorStore` selected by
  `vectorstore.database` in `settings.yaml`, which LodeDB already implements via the
  `lodedb[llama-index]` adapter, so this is a small provider shim rather than a new adapter: a
  `VectorStoreFactory` subclass that reads PrivateGPT settings and builds the text-path
  `LodeDBVectorStore` (one local LodeDB index per collection, with `path` / `model` / `device` /
  `store_text` / `index_text` configurable from an optional `lodedb:` settings block), plus
  `register_lodedb_provider()` which registers it under `"lodedb"` with PrivateGPT's
  `register_vector_store` factory registry. Selecting it inside PrivateGPT is one line to trigger
  registration (the registry is process-local, with no entry-point auto-discovery) plus
  `vectorstore.database: lodedb` (or `PGPT_VECTORSTORE=lodedb`); see
  `examples/privategpt_provider.py` and `docs/integrations.md`. No PrivateGPT fork is required.

### Changed

- **ONNX Runtime is now the default embedding runtime, with a PyTorch fallback.** A new
  `embedding_runtime` option on `LodeDB` (and `--runtime` on the CLI; `LODEDB_EMBEDDING_RUNTIME`
  for the MCP server) selects `"auto"` (default), `"onnx"`, or `"torch"`. Under `"auto"` LodeDB
  embeds through ONNX Runtime when `onnxruntime` is installed and the model's ONNX graph can be
  obtained, and otherwise falls back to PyTorch `sentence-transformers`. The `minilm` and `bge`
  presets ship a prebuilt ONNX graph on the Hub, so it is fetched and cached on first use (under
  `~/.cache/lodedb/onnx`, override with `LODEDB_ONNX_CACHE`) with no export step; exporting an ONNX
  graph for a model that does not ship one is the opt-in `lodedb[onnx-export]` extra (Optimum, run
  in a subprocess). ONNX produces vectors matching the sentence-transformers path for the same model
  (cosine > 0.99 on MiniLM), so existing indexes stay compatible. It is markedly faster for
  single-query and incremental-add latency (about 3x on a measured Apple Silicon CPU); large-batch
  cold-indexing throughput is hardware-dependent and can still favor torch on CPU, so pass
  `embedding_runtime="torch"` for batch-indexing-heavy workloads. `onnxruntime` and `transformers`
  join the base install; `sentence-transformers` (which pulls torch) remains for the fallback.
  `lodedb doctor` now reports the preferred runtime (with its torch fallback) and the active ONNX
  execution providers. On Apple Silicon the ONNX runtime uses the CPU execution provider by default;
  the Core ML provider is opt-in (`LODEDB_ONNX_COREML=1`) because on the preset graphs it fragments
  into many Core ML/CPU partitions and measured slower than CPU for single-query embedding.

## [0.3.0] - 2026-06-24

### Changed

- **The default commit mode is now `commit_mode="wal"`.** A durable single `add`/`remove`
  appends one length-prefixed, CRC32-framed record to a `<key>.wal` log (a single `write`, plus
  one `fsync` under `durability="fsync"`) and a full generation is checkpointed only periodically,
  instead of publishing a new generation on every mutation, removing the multi-file generation
  publish from the hot write path. The WAL records logical
  mutations and is replayed on open by re-driving the same engine verbs, so recovery is
  crash-atomic: a half-written trailing record is discarded, an interior-corrupt record fails
  closed, and every writable open folds any recovered WAL into a clean committed generation (so a
  handle always opens onto a consistent generation, regardless of mode). WAL is single-writer; a
  concurrent `open_readonly` reader loads a consistent committed generation, but the last
  checkpointed one, not the writer's in-flight WAL. Pass `commit_mode="generation"` (or
  `LODEDB_COMMIT_MODE=generation`) to keep the previous behavior: a crash-atomic, MVCC-readable
  generation published on every write, for deployments where many out-of-process readers must see
  each write the instant it commits.
- **Direct TurboVec index sync is now O(changed) per mutation.** Each `add`/`upsert`/`delete`
  previously re-derived the full corpus diff against the in-memory index on every call (scanning
  the whole id map and all of `state.chunks`, plus copying both id maps and rebuilding the reverse
  id map), making a single durable add O(corpus). The mutation verbs now hand the sync their exact
  chunk-level delta, truth-checked against the post-mutation `state.chunks` (with a full-corpus-diff
  fallback for any path that does not supply one), so per-add work is O(changed) and flat with
  corpus size. Combined with the WAL default this puts durable single-add at ~0.6 ms on an L40S at
  ~17.5k docs (down from ~6 ms), in the sqlite-vec/qdrant range; both commit modes benefit.

## [0.2.1] - 2026-06-23

### Added

- **One-command MCP install for coding assistants.** `lodedb mcp install --client <client>`
  registers the LodeDB MCP server with a coding assistant in one step, instead of hand-editing
  each host's config. It supports `claude-code`, `claude-desktop`, `cursor`, `lm-studio`, `codex`,
  and `all`, resolving the right launch command for the current environment so `command`/`args`
  are correct even when `lodedb` is not on `PATH` (it falls back to the `uv run --project ...`
  form, then an absolute path to the entry point), and resolving `--path` to an absolute path so
  the entry works wherever the client launches the server. The edit is idempotent (an existing
  `lodedb` entry is updated, never duplicated) and leaves other servers untouched; Claude Code is
  registered via `claude mcp add` and the others edit the JSON/TOML config directly. It passes
  through the `lodedb mcp` options (`--path`, `--model`, `--device`, `--exclude-text`,
  `--no-store-text`), prints the entry and the file it wrote, and supports `--dry-run`, a
  `--config <path>` override, and Cursor's project-level `--project <dir>`. `lodedb mcp uninstall
  --client <client>` removes the entry again.

- **MCP search returns document text and defaults to hybrid.** The `lodedb mcp` server's
  `lodedb_search` tool now returns each hit's stored text alongside the score, id, and
  metadata, so an agent can rank and answer in a single call instead of chaining a follow-up
  get-by-id. It also runs hybrid (BM25 lexical + vector) ranking by default when a lexical
  source is available, recovering exact tokens like error codes and serials, and falls back to
  a vector scan otherwise. Pass `--exclude-text` to redact text from the server (search returns
  metrics only and the get-by-id tool is withdrawn) while keeping it on disk for hybrid search,
  or `--no-store-text` to retain no text at all. The stats tool stays metrics-only and raw
  query text never leaves the process.

- **`lodedb doctor` detects CPU-only PyTorch on Windows.** PyPI serves the CPU-only torch
  wheel on Windows by default (and no package metadata can redirect torch to the CUDA index),
  so embeddings silently run on the CPU. `doctor` now flags this and prints the CUDA-index
  reinstall command, and `lodedb doctor --fix` runs the reinstall so embeddings use an NVIDIA
  GPU.

- **LlamaIndex `VectorStore` adapter** (`lodedb[llama-index]`). `LodeDBVectorStore` in
  `lodedb.local.integrations.llama_index` wraps the LodeDB SDK as a LlamaIndex
  `BasePydanticVectorStore`, joining the existing LangChain adapter. It is *text-path* —
  LodeDB embeds text internally (`is_embedding_query=False`), so LlamaIndex's own
  `embed_model` is not used. Query modes map onto the SDK's retrieval modes:
  `VectorStoreQueryMode.DEFAULT` to vector search, `HYBRID`/`SEMANTIC_HYBRID` to the BM25 + RRF
  hybrid, and `SPARSE`/`TEXT_SEARCH` to lexical (BM25) search. Metadata filters translate the
  LlamaIndex `FilterOperator` comparisons (`==` `!=` `>` `>=` `<` `<=` `in` `nin`, plus
  `is_empty`) and `FilterCondition` composition (`and` / `or` / `not`, nestable) into LodeDB's
  predicate grammar. `get_nodes` and filter-based `delete_nodes` are served by metadata
  enumeration, and `delete(ref_doc_id)` / `doc_ids` scoping resolve through durable metadata so
  they survive a reopen. `add` / `query` / `delete`, `node_ids` scoping, and async shims round
  out the surface; operations LodeDB cannot honor (full-precision vector reads, `MMR`/learned
  query modes, substring/list filter operators) raise clearly.
- **LlamaIndex `PropertyGraphStore` adapter** (`lodedb[llama-index]`). `LodeDBPropertyGraphStore`
  in `lodedb.local.integrations.llama_index_graph` wraps `lodedb.graph.KnowledgeGraph` as a
  LlamaIndex `PropertyGraphStore`, so `PropertyGraphIndex` can use LodeDB's hybrid graph layer
  (SQLite topology + LodeDB semantic index). `EntityNode`/`ChunkNode` map to typed graph nodes
  (node properties round-trip as JSON), and `Relation` to directed, typed edges; `get` /
  `get_triplets` / `get_rel_map` traverse the topology, while `vector_query` runs semantic node
  search. `supports_vector_queries` is true; `structured_query` (Cypher) is not supported.
  `KnowledgeGraph` gains `list_nodes` / `list_edges` complete-set enumeration (to back the
  adapter's topology reads) and a **vector-only mode** (`KnowledgeGraph(vector_dim=D)`) that
  indexes nodes and edges by caller-supplied embeddings at an arbitrary dimension with no
  internal embedder. The adapter detects that mode and stores LlamaIndex node embeddings /
  queries by `query.query_embedding`, so the high-level `PropertyGraphIndex` works with **any**
  `embed_model` (set `D` to the embedder's dimension). In the default text-path mode it embeds
  node text and `query.query_str` with the graph's own model, mapping
  `DEFAULT`/`HYBRID`/`SEMANTIC_HYBRID`/`SPARSE`/`TEXT_SEARCH` the same way the vector-store
  adapter does.
- **mem0 `VectorStore` adapter** (`lodedb[mem0]`). `LodeDBVectorStore` in
  `lodedb.local.integrations.mem0` implements mem0's `VectorStoreBase` against LodeDB's
  vector-in API, plus a `register_mem0_provider()` helper so `Memory.from_config({"vector_store":
  {"provider": "lodedb", ...}})` resolves it from mem0's runtime factory. mem0 owns the
  embeddings; LodeDB stores and searches the vectors locally and persists changed rows
  incrementally. The full mem0 payload JSON is retained in LodeDB's raw-text sidecar, so raw
  memory text and list-valued fields (such as `linked_memory_ids`) never leak into the redacted,
  scalar-only metadata used for `user_id`/`agent_id`/`run_id` filtering. mem0's filter grammar
  (`eq`/`ne`/`gt`/`gte`/`lt`/`lte`/`in`/`nin`, plus `AND`/`OR`/`NOT`) translates into LodeDB's
  predicate grammar; `keyword_search` runs BM25 over the retained payload text and `search_batch`
  uses the vector-in batch query path. cosine similarity only.

### Documentation

- **MCP server setup.** Added a "Use as an MCP server" README section covering the exposed
  tools and host wiring for Claude Code/Desktop, Cursor, and LM Studio.
- **Windows GPU embeddings.** README documents the CPU-only-torch default on Windows and the
  `lodedb doctor --fix` (or manual CUDA-index) reinstall.

## [0.2.0] - 2026-06-23

### Added

- **Knowledge-graph and memory backend.** New `lodedb.graph.KnowledgeGraph` layer: graph topology
  (nodes, typed edges, properties) in an embedded SQLite sidecar with the semantic index in LodeDB,
  `neighbors` / `k_hop` traversal, and `search_subgraph` (semantic seeds plus k-hop expansion). The
  SDK gained a vector-in path (`add_vectors` / `add_vectors_many` / `search_by_vector` /
  `search_many_by_vector`) for precomputed embeddings, a vector-only index
  (`LodeDB(vector_dim=...)` / `open_vector_store`, no embedding model), metadata enumeration
  (`list_documents` / `count`), a selectivity-ordered predicate filter planner, and inline metadata
  on search results. See `docs/graph.md`.

- **Hybrid lexical + vector search (BM25 + RRF).** `search`/`search_many` take a `mode`
  parameter: `"vector"` (default, unchanged), `"hybrid"`, and `"lexical"`. Hybrid runs an
  Okapi BM25 ranker alongside the vector scan and fuses the two ranked lists with Reciprocal
  Rank Fusion, so exact tokens the embedding misses (error codes like `E1234`, serials like
  `ABC-123`, dates like `2024-01-15`) are recovered when they appear in the document body. The
  tokenizer keeps code-like tokens whole. A `filter` constrains both rankers, so filtered
  hybrid returns the true top-k of the matching subset. The BM25 ranking and the fusion are a
  pure-Python CPU post-step and the vector kernel is untouched; the vector half of a hybrid
  query still rides the batched GPU/MPS scan that serves `search_many`. The serving BM25 index
  lives in memory, is maintained incrementally across mutations (a small change folds in only
  the changed chunks), and is never sent to telemetry. By default it is rebuilt
  from the retained raw text, so `mode="hybrid"`/`"lexical"` work whenever `store_text=True`
  (the default). The new `LodeDB(..., index_text=True)` flag instead persists the index: the
  per-chunk terms are captured at `add` time into a dedicated `.tvlex` base plus a `.lxd` delta
  journal (checksum-guarded, committed O(changed) per write, pinned by the same root commit
  manifest as the index), so hybrid and lexical search survive a reopen without rebuilding from
  raw text and without requiring `store_text=True`. The `.tvlex` sidecar holds payload-derived
  terms only and, like the raw-text sidecar, never reaches the redacted `.json`/`.jsd`/`.tvim`/
  `.tvd` artifacts or telemetry; `index_text` defaults to off, leaving the standard layout
  unchanged. A lexical query with neither flag set raises a clear error. The same `mode` is
  exposed on the knowledge-graph search API (`semantic_nodes` / `semantic_edges` /
  `search_subgraph`), so exact tokens in node labels and edge facts are recoverable there too.

## [0.1.2] - 2026-06-22

### Added

- **Metadata filter predicates.** `search`/`search_many` `filter=` now accepts comparison
  operators (`$eq` / `$ne` / `$gt` / `$gte` / `$lt` / `$lte` / `$in` / `$nin` / `$exists`) and
  boolean composition (`$and` / `$or` / `$not`) alongside exact match. A bare scalar stays
  exact-match, so existing filters are unchanged. Ordered comparisons (`$gt`/`$gte`/`$lt`/`$lte`)
  are numeric when both the stored value and the operand parse as numbers, otherwise
  lexicographic; equality and membership (`$eq`/`$ne`/`$in`/`$nin`) always compare as strings, so
  `{"n": {"$eq": 9.9}}` does not match a stored `9.90`. No storage-format change — the predicates
  evaluate at query time over the existing string metadata.
- **Single-writer concurrency safety.** A LodeDB handle now holds an exclusive OS advisory
  lock (`<dir>/.lodedb.lock`) for its lifetime, so concurrent processes can no longer corrupt
  the on-disk store. A second open of the same path waits for the first to close — then loads
  the accumulated state and composes — and fails fast with `ConcurrentWriterError` once
  `LODEDB_PERSIST_LOCK_TIMEOUT` (default 30s) elapses, the model SQLite uses with a busy
  timeout. The kernel releases the lock on process exit, so a crash never wedges the path.
  Local filesystems only (advisory locks are unreliable on NFS/SMB). Live cross-process
  refresh (a reader auto-seeing another live process's writes) remains out of scope.
- **Read-only handles (single writer, many readers).** `LodeDB.open_readonly(path)` (or
  `read_only=True`) opens a non-mutating snapshot that takes **no** writer lock, so it can
  read a path while a writer holds it — `lodedb query` and `lodedb get` now use it, so they
  work alongside a running `lodedb serve`/`mcp`. Mutating calls raise `ReadOnlyError`; the
  path must already exist. A read-only open loads the single consistent generation named by
  the atomic commit manifest (below), so it never observes a torn cross-file mix.
- **In-process operation lock.** The engine now serializes its public operations under a
  reentrant lock, so the threaded `lodedb serve` can safely share one handle across request
  threads (concurrent `add`/`search`/`remove` no longer race on shared state).
- **Configurable durability.** `durability="fsync"` (also `--durability fsync` on
  `serve`/`index`, or `LODEDB_DURABILITY=fsync`) fsyncs each persisted file and its directory
  on commit for power-loss durability. The default `"fast"` keeps the prior atomic-rename
  behavior (atomic, not power-loss durable) and commit throughput.
- **Crash-atomic multi-file commits (atomic root manifest).** A commit touches several files
  (JSON state base + `.jsd` journal, `.tvim` vector base + `.tvd` journal, and the opt-in
  `.tvtext` raw-text base + `.txd` journal); they are now written as generation-addressed
  artifacts under a per-index `<key>.gen/` directory and sealed by atomically swapping a single
  `<key>.commit.json` root pointer — that swap is the only commit point. Because every
  artifact (including raw text, on by default) is generation-addressed and pinned by the root,
  none is overwritten in place, so a crash (or `kill -9`) mid-commit leaves the previously
  committed generation fully intact and the next open rolls back to it (dropping the
  uncommitted artifacts) instead of failing closed and stuck. Lock-free readers load exactly
  the generation the root manifest names — consistent snapshot isolation (text included), no
  torn cross-file reads. Stores written by v0.1.x load via a legacy fallback and migrate to
  the new layout on their next write; superseded generations are garbage-collected (the most
  recent few are retained for in-flight readers).
- **First-class opt-in Apple-GPU (MPS) exact scan.** The Metal/MPS resident scan is now a
  selectable route at CUDA-level capability, off by default: in-place `patch()` on small
  mutations (O(changed) swap-remove + batched upsert), an `MpsDirectTurboVecPolicy`
  (`LODEDB_MPS_DIRECT_TURBOVEC`, default `off`), engine dispatch for batched `search_many`,
  an optional `LODEDB_MPS_MEMORY_BUDGET_BYTES` guard, and honest `lodedb doctor` reporting.
  MPS uses shared resident-scan helper code for deterministic top-k ordering and tile sizing;
  CUDA extraction onto those helpers remains a hardware-verified follow-up. **NEON remains the
  default on Apple Silicon** — the MPS scan was slower than NEON at every batch size on the
  measured M1, so it is opt-in and off by default; any future default flip is gated on per-chip
  `benchmarks/mps_vs_neon` crossover data, especially for newer Apple GPUs such as M5.

### Changed

- **Filtered multi-query search pushes a shared allowlist into the batch scan.** `search_many`
  (and `query_batch`) with a `filter=` no longer widens each query's effective `top_k` to the
  corpus and post-filters; it groups queries by identical filter and pushes one shared allowlist
  into a single batched native scan, so `top_k` stays `k`. A generation-keyed metadata posting
  index resolves a filter to chunk ids in O(matching docs) instead of scanning the corpus, rebuilt
  lazily on the first filtered query after a mutation (no write/commit overhead). The GPU/MPS
  resident scans honour the allowlist via a per-tile score mask, so filtered batches stay on the
  resident path above the 4096-row cap instead of falling back to the CPU kernel. Measured locally
  (20k docs, batch 32, 1%-selective filter), filtered `search_many` dropped from ~335 ms to ~4 ms,
  matching unfiltered latency.
- **Incremental commits are O(changed), not O(corpus).** A single-doc commit no longer does
  three pieces of whole-corpus work on the write path: it no longer eagerly rebuilds
  TurboVec's SIMD "blocked" layout (the next query rebuilds it lazily, once, amortizing a
  burst of commits), no longer runs a per-commit quantization-drift self-score search (that
  drift metric is now sampled opportunistically on the next query that warms the layout), and
  drops only the transient embeddings of the rows just added rather than re-walking every
  chunk. Measured single-doc `add()` latency at 20K docs dropped from ~58 ms to ~15 ms
  (~3.9×), and the gap widens with corpus size since the removed work was O(corpus). The
  deferred layout rebuild lands once on the first query after a write burst.
- **Raw-text persistence is O(changed) too.** With `store_text=True` (the default), an
  incremental commit now appends a small `.txd` text delta (the upserted texts + deleted ids
  of that batch) onto a `g<epoch>.tvtext` base, instead of rewriting the whole
  `document_id -> text` map every commit; a load replays the deltas onto the base, and the
  store remains checksum-guarded and fails closed. This removes the last whole-corpus write
  from the commit path: isolated, the per-commit text write drops from ~57 ms at 20K docs
  (~244 ms at 80K) to a flat ~0.7 ms regardless of corpus size. Raw text stays in the atomic
  commit set (base + journal pinned by the root manifest), so it still commits and rolls back
  with the generation; v0.1.x single-file `.tvtext` sidecars migrate into the journal on the
  next write.

## [0.1.1] - 2026-06-20

### Changed

- **GPU (`[gpu]`) resident copy now patches in place on small mutations** instead of
  rebuilding the whole dequantized array. Adds and removes apply in O(changed) rows
  (swap-remove + batched upsert) with a fail-closed rebuild fallback, so syncing a small
  delta into a large GPU-resident index is dramatically cheaper — e.g. ~560× faster at
  1,000 changed rows over a 1M-row corpus on an A10 — with identical top-k results.

### Fixed

- **GPU memory admission** now accounts for the 1.5× resident over-allocation, so it no
  longer under-counts the device memory an index will occupy.

## [0.1.0] - 2026-06-19

First public release.

### Added

- **Local-first, privacy-by-default vector database.** Embeds into your process; your
  data never leaves the machine. No accounts, no network calls on the core path.
- **Compact storage** via the bundled, patched **TurboVec** core (2/4-bit codes), shipped
  as the `lodedb._turbovec` extension. Wheels are `abi3`, so one wheel covers all supported
  Python versions on a platform and there is nothing to compile on install.
- **Local embeddings** through `sentence-transformers` on CPU, CUDA, or Apple MPS, with
  built-in model presets (`LOCAL_MODEL_PRESETS`).
- **Delta persistence**: on-disk index (`.tvim` / `.tvd`) plus a journal (`.jsd`) for fast
  incremental updates, and an optional `.tvtext` raw-text sidecar gated by `store_text`
  (default `True`; set `store_text=False` to keep no document text on disk).
- **Python API** — `LodeDB` with `add`, `search`, `search_many`, `get`, and `persist`;
  results returned as `LodeSearchHit`.
- **`lodedb` CLI** — `doctor`, `index`, `query`, `get`, `benchmark`, `serve`, `mcp`.
- **Local HTTP dev server** (`lodedb serve`) — loopback-only, no auth, metrics-only
  telemetry; exposes `/healthz`, `/stats`, `/search`, and `/get`.
- **Optional extras**:
  - `[gpu]` — opt-in CUDA-resident exact scan (cupy; Linux/CUDA only).
  - `[mcp]` — stdio MCP server so coding agents can use LodeDB as local memory.
  - `[langchain]` — LangChain `VectorStore` adapter.

### Notes

- Runtime PyPI dependencies are kept lean: `numpy`, `typer`, `sentence-transformers`,
  `pyyaml`. Heavier research dependencies are not imported on the core path; this is
  enforced by `tests/test_import_boundary.py`.
- LodeDB is licensed under Apache-2.0. The vendored TurboVec core
  (`third_party/turbovec/`) is MIT — see [`NOTICE`](NOTICE).

[Unreleased]: https://github.com/Egoist-Machines/LodeDB/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/Egoist-Machines/LodeDB/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Egoist-Machines/LodeDB/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Egoist-Machines/LodeDB/releases/tag/v0.1.0
