# Embedded open and first-query memory

The probe opens a committed store read-only, optionally refreshes, then runs the
same query twice. Each stage reports elapsed time, current process memory, the
lifetime memory high-water mark, live allocated bytes, and the stage's heap peak.
On Linux, process memory is RSS (`VmRSS` / `VmHWM` from `/proc/self/status`); on
macOS it is the task's physical footprint. These metrics are not interchangeable.
`settled` reports steady residency after 1.5 seconds, without forcing the allocator
to release memory. Every mode runs in a fresh process.

To create a reproducible synthetic input (10,000 documents, 384 dimensions, 4-bit
vectors, seed 134, 4,096 generated terms plus six recurring phrases, about 64
tokens per document, one category metadata field):

```bash
cargo build --release --manifest-path benchmarks/embedded_open_memory/Cargo.toml \
  --target-dir target/embedded-open-memory/head
target/embedded-open-memory/head/release/create_store target/memory-store-10k
benchmarks/embedded_open_memory/run.sh --runs 3 target/memory-store-10k
```

Use `OPENBLAS_NUM_THREADS=1` for reproducible CPU threading. Reuse the generated
store for every revision; do not regenerate it between measurements. This
synthetic input is not the original issue's store, which is not in the repository.

`run.sh --base <revision> --runs 3 <store>` also builds the base revision with the
same probe, and compares byte-identical result dumps (including score bits) for
vector, lexical and hybrid queries, filters and two k values, on read-only handles
and writable copies after upsert/delete/persist/reopen. It never changes the
supplied store. The Rust core also has fixed-seed differential tests.

To repeat a saved baseline without recompilation, copy `embedded-open-memory`
and `results_dump` from the release directory before editing the core, then use
`run.sh --probe-dir <saved-binary-dir> --runs 3 <store>`. This runs the same stages
in fresh processes and allows measurements after other builds have finished.
