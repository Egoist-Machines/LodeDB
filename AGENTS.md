# AGENTS.md

Guidance for humans and AI coding agents working in the LodeDB repository.

## What this is

LodeDB is an open-source (Apache-2.0), local-first embedded vector database. The public
surface is `import lodedb` plus the `lodedb` CLI; everything lives under `src/lodedb/`.

```
src/lodedb/__init__.py   public API: LodeDB, LodeSearchHit, the CLI
src/lodedb/local/        local-first SDK: db, CLI, MPS/CPU embedding selector, dev
                         server, MCP server, LangChain/LlamaIndex/mem0 adapters,
                         and the `lodedb migrate` toolkit (local/migrate/)
src/lodedb/engine/       engine core: index / storage / embedding / route profiles
                         (LodeEngine in core.py, LodeIndex in index.py)
third_party/turbovec/    vendored MIT compact vector core (preserve LICENSE)
tests/                   local SDK suite + import-boundary guard
```

## Hard rules (non-negotiable)

- **Privacy / metrics-only:** telemetry and the redacted artifacts (`.json` snapshot, `.jsd`
  journal, `.tvim`/`.tvd` sidecars, audit log) never carry raw documents, queries, chunks,
  embeddings, or credentials — only counts / bytes / latency / ids / timestamps. The one
  exception is original document text, retained by default in a *separate* raw-text store
  (`g<epoch>.tvtext` base + `.txd` delta journal) for `db.get(id)`; pass `store_text=False` to
  keep no text on disk. No telemetry/redacted path reads that store.
- **Lean dependencies:** base runtime PyPI deps are `numpy`, `typer`, `pyyaml` — a
  dependency-light vector store (bring your own vectors via `open_vector_store` / `add_vectors`,
  or pass `embedder=`). Built-in text embedding is opt-in: `[embeddings]` (`onnxruntime` +
  `transformers`, torch-free) and `[torch]` (`sentence-transformers` — the PyTorch fallback, the
  `clip` preset, and `device="mps"`); other extras are `[image]`, `[onnx-export]`, `[mcp]`,
  `[langchain]`, `[llama-index]`, `[mem0]`, `[gpu]`, `[cloud]` (the first-party managed-cloud
  client `lodedb.cloud` + the `lodedb cloud` CLI, over the bundled native transfer core; the
  extra adds only `httpx` + `pynacl`, both reached lazily and never on a plain import — the
  extra is deliberately NOT part of `[all]`), `[cloud-sealed]` (the delegated-custody
  sealed-store client verbs; adds only `cryptography`, reached lazily and never on a plain
  import, and likewise NOT part of `[all]`), and `[all]`. The patched TurboVec core is
  vendored under `third_party/turbovec/` and bundled into the wheel as `lodedb._turbovec` — not a
  PyPI dependency. The embedding runtimes load lazily (only when a preset/CLIP backend is built),
  so a plain `import lodedb` must not import `onnxruntime` / `transformers` /
  `sentence-transformers` / `torch` even when they are installed; the preset path raises a clear
  `lodedb[embeddings]` install hint when none is.
  Importing LodeDB must **not** load `faiss` / `modal` / `mteb` / `datasets` / `matplotlib`
  / `sklearn` — `tests/test_import_boundary.py` fails closed (fresh subprocess) if it ever
  does. (`scikit-learn` may be *installed* transitively via `sentence-transformers`; the
  guard checks what *loads*, not what is installed.) The same discipline covers the optional
  framework adapters (`langchain` / `llama_index` / `mem0`) and the `lodedb migrate` source
  providers (`psycopg` / `qdrant_client` / `chromadb` / `lancedb`): `import lodedb` reaches the
  `migrate` CLI sub-app, so every source exporter (`local/migrate/sources/`) keeps its provider
  import function-local, and the same test guards them.
- **Licensing:** our code is Apache-2.0; preserve the upstream **MIT** notice on
  `third_party/turbovec/` (see `NOTICE`). No GPL/AGPL/LGPL dependencies.
- **Local path stays local:** the local SDK runs in a no-auth, loopback/private-network,
  CPU-scan profile. Binding `lodedb serve` to an RFC1918/private address is an intentional
  trusted-LAN mode; never allow public/untrusted binds without adding an explicit auth story.
  Don't add auth, external network, or server requirements to the embedded SDK. The CUDA scan
  (`[gpu]`) is an opt-in extra and must stay lazy — importing LodeDB never requires CuPy.
- **Concurrency invariants:** a writable handle holds the cross-process single-writer lock
  (`engine/_filelock.py`) for its lifetime; a `read_only=True` handle takes **no** lock and
  must never persist (`_persist_state` enforces this). Public `LodeEngine` operations run
  under one reentrant in-process lock (`@_synchronized`) so the threaded `serve` can share a
  handle. Persist every file via `durable_replace` (temp + `os.replace`, with the fsync gated
  on `durability="fsync"`) — never write a persisted artifact in place. Two carve-outs hold the
  lock on the file they rewrite, so a rename would drop it: journal appends (`.txd`, `<key>.wal`)
  and the fixed-width `<key>.lsn` counter, which is CRC-guarded and reseeds itself from the store
  when torn.
- **Atomic commits (`engine/_commit_manifest.py`):** an index commits by writing
  generation-addressed artifacts under `<key>.gen/` and then atomically swapping the
  `<key>.commit.json` root pointer — that swap is the *only* commit point, so never overwrite a
  base in place (write a new epoch) and never publish state outside a root-manifest swap. Loads
  go through the committed root (embedded per-store manifests); writer-open recovery
  (`_recover_to_commit`) rolls a torn commit back to the last good generation and GCs
  superseded epochs. A top-level `<key>.json` with no commit manifest is a legacy (v0.1.x)
  store: load it via the fallback and migrate on next write. Keep this back-compat path.
- **Keep the commit path O(changed).** A mutation invalidates TurboVec's derived SIMD
  "blocked" layout; do not rebuild it (`prepare()`) or `search()` it on the commit path — the
  next query rebuilds it lazily (one repack amortizes a burst of commits). The
  quantization-drift metric is therefore buffered (`_buffer_pending_drift`) and sampled on the
  next warm query (`_sample_pending_drift`), never per commit. Drop only the rows just synced
  (`_discard_direct_turbovec_transient_embeddings(state, synced_ids)`), not every chunk. The
  exception is cold builds/compaction in `build_turbovec_serving_index`, which prepare once.
  This includes the raw-text store (enabled by default; opt out with `store_text=False`): a
  delta commit appends a `.txd` text delta (upserted texts + deleted ids) via `_journal_text`,
  never rewriting the full `document_id -> text` map — only a base rewrite (cold
  build/compaction) writes the full `g<epoch>.tvtext` base.

## Develop

```bash
uv sync --extra dev --extra embeddings --extra torch --extra mcp --extra langchain --extra llama-index --extra mem0 --extra cloud --extra cloud-sealed  # build venv (compiles TurboVec)
uv run pytest -q                                                # run the suite
uv run ruff check .                                             # lint (line-length 100)
```

The vendored TurboVec crate is built from source by `uv sync` (maturin / pyo3). It links a
CBLAS provider: the Accelerate framework on macOS, OpenBLAS on Linux (`apt-get install
libopenblas-dev`).

## Releasing

Cut a release `vX.Y.Z` by setting the **same** version in every place below, then tagging.
The `version-check` CI job asserts that pyproject, `__version__`, and the tag all agree,
and the `smoke` job asserts `uv lock --check`, so a missed file or stale lockfile fails CI.

1. `pyproject.toml` `version`.
2. `src/lodedb/__init__.py` `__version__` (must equal the pyproject version and the tag).
3. `Cargo.toml` `[workspace.package] version` (the `lodedb-core` / `lodedb-ffi` / `lodedb-gpu`
   crates inherit it via `version.workspace = true`).
4. Regenerate `uv.lock` with `uv lock`.
5. Regenerate `Cargo.lock` with `cargo build` (or `cargo update -w`).
6. Add a `CHANGELOG.md` entry.
7. Commit `release: vX.Y.Z` on `main`, then `git tag -a vX.Y.Z && git push --tags`.

The tag push triggers `release.yml`: it builds the wheels + sdist and the
`LodeDBCoreFFI.xcframework`, publishes to PyPI (Trusted Publishing), creates the GitHub
Release with the xcframework attached, and (via the `swift-package-publish` job) pushes the
matching `vX.Y.Z` to the `Egoist-Machines/swift-lodedb` SwiftPM package. The Swift version is
derived from the release tag, so there is no separate Swift version to bump.

Do **not** bump on a routine release: `NATIVE_CORE_ABI_VERSION`
(`crates/lodedb-core/src/version.rs`), which tracks the C ABI and changes only when the ABI
does; and dependency version constraints (e.g. `mcp>=1.0.0`), which are not the package version.
