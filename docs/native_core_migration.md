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

## Swift / iOS Binding State

- `swift/LodeDBCore` is a source Swift package with no Python dependency.
- Local development loads the shared Rust C ABI dynamically through `LODEDB_FFI_DYLIB`.
- When that dylib is configured, Swift text ingestion calls native `prepare_text_upsert` and
  `apply_text_upsert`; Swift embedders still provide embeddings outside the core.
- Unfiltered vector searches over native-ingested text can query the same native handle through
  the C ABI. Filtered searches and locally diverged handles fall back to the Swift mirror.
- `scripts/package_xcframework.sh` builds `lodedb-ffi` static libraries for installed Rust
  Apple targets and assembles `LodeDBCoreFFI.xcframework`. It defaults to the host target for
  local verification; set `LODEDB_XCFRAMEWORK_TARGETS` to include iOS device/simulator targets
  after installing them with `rustup target add`.

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
LODEDB_FFI_DYLIB="$(pwd)/target/debug/liblodedb_ffi.dylib" \
  swift test --package-path swift/LodeDBCore
env -u LODEDB_FFI_DYLIB swift test --package-path swift/LodeDBCore
LODEDB_XCFRAMEWORK_TARGETS="aarch64-apple-ios aarch64-apple-darwin" \
  swift/LodeDBCore/scripts/package_xcframework.sh
```
