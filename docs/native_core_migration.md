# Native Core Migration Notes

This branch moves LodeDB toward a shared Rust engine while keeping the Python public API and
existing stores stable.

## Current Runtime State

- `LODEDB_NATIVE_CORE=on` is the default.
- Fresh vector-only handles (`LodeDB.open_vector_store(...)`) execute vector queries through the
  native `CoreEngine` when the bundled extension is available and the native handle covers all
  in-memory mutations for that handle.
- Python remains the durable oracle. Vector mutations still commit through the existing Python
  engine, and persisted stores open without migration.
- Existing non-empty stores fall back to Python until native storage can reload exact persisted
  vectors. This avoids scoring against incomplete native state.
- `LODEDB_NATIVE_CORE=off` remains available for one deprecation cycle.
- `LODEDB_NATIVE_CORE=shadow` keeps Python authoritative while checking native parity on covered
  vector-only handles.
- `LODEDB_NATIVE_CORE_WRITE=on` is available for explicit fresh vector-only and text stores
  opened with `commit_mode="generation"`; Rust writes the same generation artifacts that Python
  can reopen. WAL-mode write-through and existing non-empty store rewrites remain on the Python
  oracle until the storage cutover is complete.

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

## Removal Gate For Python Runtime Paths

Do not remove the Python engine oracle from the runtime package until all of these are true:

- Native storage can load existing generation, WAL, text, lexical, and vector sidecars with exact
  query parity.
- Native text prepare/apply can replace Python text mutation orchestration while bindings keep
  embeddings outside the core.
- Native query assembly covers vector, batch vector, lexical, hybrid, filters, and metadata
  inclusion for existing stores.
- Golden fixtures and differential tests cover v0.4 generation, WAL, raw-text, lexical-text, and
  legacy top-level JSON stores.
- Benchmarks show no unacceptable regression on the performance gates in `GOAL.md`.

Until then, Python engine paths are intentionally retained as compatibility and persistence
fallbacks rather than archived.

## Current Performance Gate Snapshot

The metrics-only Rust-vs-Python closure benchmark is recorded at
`benchmarks/native_migration/results/rust_vs_python_local.json`. The latest local run uses
2,000 deterministic documents, 200 deterministic queries, `dim=64`, and `k=8`.

The artifact reports `pass_fail_summary.passed=true`. Rust/Python elapsed ratios:

| Path | Ratio |
| --- | ---: |
| Vector upsert | 0.242 |
| Unfiltered vector search | 0.920 |
| Filtered vector search | 0.652 |
| Batch vector search | 0.303 |
| Text prepare/apply with `HashEmbeddingBackend` | 0.529 |
| Lexical search | 0.331 |
| Hybrid search | 0.996 |
| Persisted reopen/query | 0.598 |

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
