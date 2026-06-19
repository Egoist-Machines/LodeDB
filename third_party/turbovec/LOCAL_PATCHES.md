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
