# Changelog

All notable changes to LodeDB are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Optional write-ahead-log commit mode (`commit_mode="wal"`).** A single-process,
  single-writer alternative to the default per-mutation generation publish, for write-heavy
  workloads. Each `add`/`remove` appends one length-prefixed, CRC32-framed record to a
  `<key>.wal` log (a single `write`, plus one `fsync` under `durability="fsync"`) and a full
  generation is checkpointed only periodically, dropping durable single-add latency by ~10x into
  the sqlite-vec/qdrant range. The WAL records logical mutations and is replayed on open by
  re-driving the same engine verbs, so recovery is crash-atomic: a half-written trailing record
  is discarded, an interior-corrupt record fails closed, and a clean `close()`/`persist()` folds
  the WAL into a generation. Opt in per handle (`LodeDB(path, commit_mode="wal")`) or via
  `LODEDB_COMMIT_MODE=wal`. The default `generation` mode, its on-disk layout, and its lock-free
  MVCC readers are unchanged; a concurrent `open_readonly` reader still sees the last committed
  generation (just not the writer's in-flight WAL).

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
