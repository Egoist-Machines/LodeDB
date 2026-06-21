# Changelog

All notable changes to LodeDB are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

### Changed

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

[Unreleased]: https://github.com/Egoist-Machines/LodeDB/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/Egoist-Machines/LodeDB/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Egoist-Machines/LodeDB/releases/tag/v0.1.0
