# LodeDB Patches to Vendored TurboVec

This vendored tree is upstream `https://github.com/RyanCodrai/turbovec`
at tag `v0.9.0` (Python 0.8.0 + Rust crate 0.9.0, commit
`1e7200cfd8f26c92ce2855652db64bc7f85bc039`, sources verified identical to
the PyPI `turbovec-0.8.0` sdist, sha256
`4ff24956ef159cd8ccdb19c561b07eea3c191c344a23368cd0adad9cdd87382c`),
plus the local patches below. Patched builds are detected at runtime by feature presence
(`hasattr(index, "add_encoded")`) rather than version metadata, distinguishing them
from unpatched PyPI wheels; PyPI `turbovec==0.8.0` does NOT contain these
APIs.

## Encoded-row export/import surface (delta persistence)

Added for LodeDB's O(changed-rows) `.tvim` delta-persistence path:

- `turbovec/src/error.rs`: `EncodedRowsError`.
- `turbovec/src/lib.rs`: `TurboQuantIndex::bytes_per_vector`,
  `calibration_fingerprint` (FNV-1a over dim/bit-width/frozen TQ+ state),
  and `pub(crate)` per-slot `export_row`/`overwrite_row`/
  `append_encoded_row` (each mutation invalidates the derived blocked
  SIMD cache).
- `turbovec/src/id_map.rs`: `IdMapIndex::remove_many`,
  `bytes_per_vector`, `calibration_fingerprint`, `export_encoded`,
  `add_encoded` (upsert semantics: overwrite in place or append;
  validates fully before mutating).
- `turbovec/tests/encoded_rows.rs`: integration tests including
  export→import search equivalence and blocked-cache invalidation.
- `turbovec-python/src/lib.rs`: bindings for the five methods
  (`KeyError` for unknown export ids, `ValueError` otherwise).

## Lifecycle additions (second pass)

- `TurboQuantIndex::calibration_fitted` + `IdMapIndex::calibration_fitted`
  (+ binding): whether a data-dependent TQ+ calibration was fitted, as
  opposed to the identity calibration committed by a sub-threshold first
  add — surfaces the trickle-ingest "lost TQ+ lift" condition.
- `TurboQuantIndex::encode_rows` (`pub(crate)`): encode against the frozen
  coordinate system without appending; asserts calibration is committed so
  it can never refit global state.
- `IdMapIndex::upsert_with_ids_2d` (+ `upsert_with_ids` binding):
  slot-preserving in-place replace for existing ids plus append for new
  ids, validated before mutation; delegates to `add_with_ids_2d` on an
  empty index so first-batch calibration fitting is unchanged.

Encoded rows are only portable between indexes with equal
`calibration_fingerprint()` values; callers must check before
`add_encoded` (the delta store records the fingerprint per segment and
fails closed on mismatch).

## Cluster-contiguous physical layout

Added for LodeDB's cluster-pruned SIMD scan path:

- `turbovec/src/lib.rs`: `TurboQuantIndex::permute_rows` (`pub(crate)`), an
  in-place cycle-following row permutation with one row-sized scratch buffer;
  `perm[new_slot] = old_slot` and every permutation invalidates the derived
  blocked SIMD cache.
- `turbovec/src/error.rs`: `TurboVecError` for validated ID-order requests.
- `turbovec/src/id_map.rs`: `IdMapIndex::reorder_to_ids`, which validates an
  exact stable-id permutation before reordering quantized rows and rebuilding
  the slot/id maps.

## Row reconstruction surface (GPU-resident exact serving)

Added for LodeDB's GPU-resident exact batch path:

- `turbovec/src/lib.rs`: `pub(crate) TurboQuantIndex::reconstruct_row_into`
  (decode packed codes through the kernel's calibrated score math:
  `y[d] = scale * (centroids[code[d]] / scale_tq[d] - shift[d])`) and
  `rotation_matrix_copy` (deterministic `(dim, dim)` row-major rotation,
  seeded from `(dim, ROTATION_SEED=42)`).
- `turbovec/src/id_map.rs`: `IdMapIndex::reconstruct_rows(ids)`,
  `reconstruct_all()` (slot order; `(ids, rows)`), `rotation_matrix()`.
- `turbovec-python/src/lib.rs`: bindings for the three methods
  (`KeyError` for unknown ids; `rotation_matrix()` returns `None` before
  the dim commits).
- `turbovec/tests/reconstruction.rs`: score-parity tests pinning the
  coordinate-space contract — rows are exported in ROTATED calibrated
  space scaled by the stored per-row scale, and queries are rotated as
  `q_rot = q @ rotation.T` (the exact GEMM the search path runs), so
  `<q_rot, y>` reproduces the kernel score up to its uint8 LUT
  quantization error (~`1/sqrt(dim)`-scaled; between 1e-2 and 2e-2 of
  max score at dim=32, tighter at production dims). The exact
  reconstructed score is the MORE faithful estimate of the quantized
  representation; identity-calibration, 2-bit, and slot-churn
  (remove/upsert) cases are covered.

## Late-interaction MaxSim kernel (multi-vector retrieval)

Added for LodeDB's late-interaction retrieval (multi-vector / MaxSim, issue #25):

- `turbovec/src/maxsim.rs`: `maxsim_scores(query, n_query, dim, docs,
  doc_patch_counts)` — exact MaxSim of one multi-vector query against a set of
  candidate documents (the documents' patch vectors concatenated row-major,
  partitioned by per-document patch counts). Each document is scored in parallel
  (rayon) by a small `query @ doc^T` faer GEMM followed by a max-over-patches,
  sum-over-query-tokens reduction. Vectors are assumed L2-normalized, so each dot
  is a cosine similarity. Lib unit tests cover the reference value, empty
  documents, and mixed empty/non-empty bands.
- `turbovec/src/lib.rs`: `pub mod maxsim;` + `pub use maxsim::maxsim_scores;`.
- `turbovec-python/src/lib.rs`: `maxsim_scores(query, docs, doc_patch_counts)`
  module function (validates shapes, raises `ValueError` on mismatch, holds the GIL
  while the kernel reads the borrowed NumPy buffers so a concurrent mutation cannot
  race the read) registered on the `_turbovec` module.

This is purely additive (a new module + one exported function + one binding); it
does not touch the quantized index, its storage format, or any existing API.

## Mask block-skip benchmark counter

- `turbovec/src/search.rs` already exposes the process-global
  `blocks_skipped_by_mask` counter and reset helper used by the filtering tests.
- `turbovec-python/src/lib.rs` registers both as module-level
  `blocks_skipped_by_mask()` and `reset_blocks_skipped_by_mask()` functions for
  benchmark harnesses. No Rust visibility change was needed because the search
  helpers are already public.

## Native-core packaging bridge

Added while LodeDB migrates toward a shared Rust engine:

- `turbovec-python/Cargo.toml`: local path dependency on `crates/lodedb-core`.
- `turbovec-python/src/lib.rs`: registers the private native-core JSON helpers
  and `CoreEngine` handle on the bundled `_turbovec` extension.
- `src/lodedb/_native_core.py`: re-exports those symbols at the stable private
  import path `lodedb._native_core` without importing them during `import lodedb`.
