# Native Core Migration Notes

This branch moves LodeDB toward a shared Rust engine while keeping the Python public API and
existing stores stable.

## Current Runtime State

LodeDB runs entirely on the native Rust `CoreEngine`; the Python engine and its `LodeIndex`
facade have been removed. Each handle opens one native engine on a dedicated worker thread
(the engine is thread-confined) and routes every read and write through it, so a handle shared
across threads serializes onto that worker. Embedding stays in Python: the SDK embeds, the
native core stores and scores.

- The native engine is the sole reader and writer. A writable handle takes the cross-process
  single-writer lock on `<dir>/.lodedb.lock`; a second writer surfaces as `ConcurrentWriterError`.
  Read-only handles open lock-free.
- Vector, batch vector, text, lexical (BM25), hybrid (BM25 + RRF), filtered, payload, and
  enumeration reads (`get`/`get_texts`/`get_document`/`list_documents`, with `after`/`limit`
  keyset paging) all resolve natively. `search_many` shares one native batched scan.
- Reopen enforces the persisted index identity (model, provider, task, dimension, storage
  profile, bit width); a mismatch fails closed.
- WAL and generation commit modes are native. A native-authored leftover WAL is replayed on the
  next open; a generation-mode reopen folds a leftover WAL via a transient WAL-mode open.
- Rust storage loads committed `.tvim` bases plus committed `.tvd` delta segments through the
  TurboVec encoded-row replay strategy.
- The `LODEDB_NATIVE_CORE` off/shadow modes, `LODEDB_NATIVE_CORE_WRITE`,
  `LODEDB_NATIVE_CORE_STRICT_PARITY`, and the Python-side MPS / GPU-direct resident-scan knobs are
  removed. Acceleration is native NEON plus the optional native CUDA scan (`crates/lodedb-gpu`).

## Swift / iOS Binding State

- `swift/LodeDBCore` is a source Swift package with no Python dependency.
- Local development loads the shared Rust C ABI dynamically through `LODEDB_FFI_DYLIB`.
- When that dylib is configured, Swift text ingestion calls native `prepare_text_upsert` and
  `apply_text_upsert`; Swift embedders still provide embeddings outside the core.
- Text searches over native-ingested text call the same native query-plan/search protocol for
  vector, lexical, and hybrid modes while the handle remains covered by native state. Locally
  diverged handles fall back to the Swift mirror.
- `scripts/package_xcframework.sh` builds `lodedb-ffi` static libraries for installed Rust
  Apple targets and assembles `LodeDBCoreFFI.xcframework`. It defaults to the host target for
  local verification; set `LODEDB_XCFRAMEWORK_TARGETS` to include iOS device/simulator targets
  after installing them with `rustup target add`.
- The `release` workflow builds the Swift/iOS artifact for
  `aarch64-apple-darwin`, `aarch64-apple-ios`, and `aarch64-apple-ios-sim`, uploads
  `LodeDBCoreFFI.xcframework.zip`, and attaches it to tagged GitHub Releases alongside the
  Python wheels and sdist.

## Python Engine Removed

The Python engine oracle and the `LodeIndex` facade have been removed; the native Rust
`CoreEngine` is the sole engine (see "Current Runtime State"). The gate's parity,
text-orchestration, query-assembly, and benchmark conditions were met: the full suite passes
under `LODEDB_NATIVE_CORE_STRICT_PARITY=1` (both engines cross-checked) with the native result
matching on every functional test before the Python engine was removed.

Two pre-native store states are not covered native-only and form a migration boundary; open
such a store with a pre-removal release to re-checkpoint it into the native layout before
upgrading:

- Pre-commit-manifest v0.4 stores (the legacy top-level `<key>.json` layout): the native loader
  reads `<key>.commit.json` generation manifests only.
- A leftover pre-native text-ingest WAL (`upsert_documents` records): the native core cannot
  re-embed during recovery, so it fails closed with the WAL left intact (no data loss) rather
  than replaying it.

## Current Performance Gate Snapshot

The metrics-only Rust-vs-Python closure benchmark is recorded at
`benchmarks/native_migration/results/rust_vs_python_local.json`. The latest local run uses
2,000 deterministic documents, 200 deterministic queries, `dim=64`, and `k=8`.

The artifact reports `pass_fail_summary.passed=true`. Rust/Python elapsed ratios:

| Path | Ratio |
| --- | ---: |
| Vector upsert | 0.259 |
| Unfiltered vector search | 0.914 |
| Filtered vector search | 0.647 |
| Batch vector search | 0.278 |
| Text prepare/apply with `HashEmbeddingBackend` | 0.368 |
| Lexical search | 0.248 |
| Hybrid search | 0.481 |
| Persisted reopen/query | 0.604 |

These numbers prove the current deterministic benchmark gates, not removal of the Python oracle.
The oracle remains in the runtime until broader CI publication, compatibility fixtures, and the
default-native release cycle are complete.

The `ci` workflow runs the same Rust-vs-Python benchmark on Linux and uploads the metrics-only
`native-core-rust-vs-python-benchmark` artifact. That job fails closed when
`pass_fail_summary.passed` is false.

## Usability Declaration

The native Rust core is usable for the covered default-on paths: fresh vector-only Python
handles, maintained metadata-filter and BM25 query indexes, batched vector search through the
Rust TurboVec adapter, Swift vector search, and Swift text prepare/apply plus vector, lexical,
and hybrid search while the native handle owns the current state.

The core is not yet the only runtime authority. Existing persisted Python stores still open
without migration through the Python oracle, and Python remains responsible for durable writes,
embedding runtimes, CLI/server ergonomics, and integration adapters until the release-cycle
removal gate above is met.

## Verification Commands

Focused checks used during the default-on cutover:

```bash
PYTHONPATH=.:src LODEDB_ALLOW_MOCK_TURBOVEC=1 uv run pytest -q \
  tests/test_native_core_flags.py \
  tests/test_native_core_shadow_vector_store.py \
  tests/test_vector_only_index.py \
  tests/test_local_vector_in.py \
  tests/test_import_boundary.py

LODEDB_NATIVE_CORE_EXTENSION_PATH=third_party/turbovec/target/debug/lib_turbovec.dylib \
PYTHONPATH=.:src LODEDB_ALLOW_MOCK_TURBOVEC=1 \
  uv run pytest -q tests/test_native_core_extension.py

cargo build -p lodedb-ffi
LODEDB_FFI_SANITIZERS=1 cargo test -p lodedb-ffi --test abi_smoke -- --nocapture
LODEDB_FFI_DYLIB="$(pwd)/target/debug/liblodedb_ffi.dylib" \
  swift test --package-path swift/LodeDBCore
env -u LODEDB_FFI_DYLIB swift test --package-path swift/LodeDBCore
LODEDB_XCFRAMEWORK_TARGETS="aarch64-apple-ios aarch64-apple-darwin" \
  swift/LodeDBCore/scripts/package_xcframework.sh
```
