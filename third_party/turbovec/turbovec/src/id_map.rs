//! Stable external IDs on top of [`TurboQuantIndex`].
//!
//! [`TurboQuantIndex`] stores vectors positionally: calling `swap_remove`
//! invalidates external references because the previously-last vector
//! moves into the deleted slot. `IdMapIndex` wraps the positional index
//! with a bidirectional `id ↔ slot` mapping so callers can identify
//! vectors by a stable `u64` ID that doesn't change when other vectors
//! are inserted or removed.
//!
//! A bidirectional hash-table-backed `u64 ↔ slot` mapping layered over
//! the inner [`TurboQuantIndex`]. The wrapper delegates all vector
//! storage, rotation, scoring and serialization questions to the inner
//! index and only owns the ID table.
//!
//! ```no_run
//! use turbovec::IdMapIndex;
//!
//! let mut index = IdMapIndex::new(1536, 4).unwrap();
//! let vectors: Vec<f32> = vec![0.0; 1536 * 3];
//! index.add_with_ids(&vectors, &[1001, 1002, 1003]).unwrap();
//!
//! let queries: Vec<f32> = vec![0.0; 1536];
//! let (scores, ids) = index.search(&queries, 3);
//!
//! index.remove(1002);
//! assert_eq!(index.len(), 2);
//! ```
//!
//! # Complexity
//!
//! - `add_with_ids(n vectors)` — O(n) encode + O(n) HashMap inserts.
//! - `remove(id)` — O(1): one HashMap lookup, one HashMap update for the
//!   vector that moved into the deleted slot, and the inner
//!   [`TurboQuantIndex::swap_remove`].
//! - `search` — same as the inner index, plus an O(nq·k) ID translation
//!   pass over the returned slot indices.

use std::collections::HashMap;
use std::path::Path;

use crate::io;
use crate::{AddError, ConstructError, EncodedRowsError, TurboQuantIndex};

/// ID-addressed wrapper around [`TurboQuantIndex`].
pub struct IdMapIndex {
    inner: TurboQuantIndex,
    /// slot → external id. `slot_to_id[i]` is the id of the vector
    /// currently stored in slot `i` of `inner`.
    slot_to_id: Vec<u64>,
    /// external id → slot. Kept in sync with `slot_to_id`.
    id_to_slot: HashMap<u64, usize>,
}

impl IdMapIndex {
    /// Construct an id-map index with a known dim. The dim is locked at
    /// construction. Propagates the same errors as
    /// [`TurboQuantIndex::new`].
    pub fn new(dim: usize, bit_width: usize) -> Result<Self, ConstructError> {
        Ok(Self {
            inner: TurboQuantIndex::new(dim, bit_width)?,
            slot_to_id: Vec::new(),
            id_to_slot: HashMap::new(),
        })
    }

    /// Construct an empty id-map index without committing to a dim. The
    /// dim is inferred and locked on the first [`Self::add_with_ids_2d`]
    /// call. Propagates the same errors as [`TurboQuantIndex::new_lazy`].
    pub fn new_lazy(bit_width: usize) -> Result<Self, ConstructError> {
        Ok(Self {
            inner: TurboQuantIndex::new_lazy(bit_width)?,
            slot_to_id: Vec::new(),
            id_to_slot: HashMap::new(),
        })
    }

    /// Add `n = vectors.len() / dim` vectors with the given external ids.
    /// Requires the inner index's dim to already be set (eager constructor
    /// or a previous lazy add).
    ///
    /// Returns the same errors as
    /// [`Self::add_with_ids_2d`]. Panics only if the inner index is still
    /// in lazy/uninitialized state — that signals API misuse (use
    /// `add_with_ids_2d` on a lazy index), not bad input.
    pub fn add_with_ids(&mut self, vectors: &[f32], ids: &[u64]) -> Result<(), AddError> {
        let dim = self.inner.dim_opt().expect(
            "IdMapIndex dim is not set; use add_with_ids_2d(vectors, dim, ids) \
             on the first add or construct with IdMapIndex::new(dim, bit_width)",
        );
        self.add_with_ids_2d(vectors, dim, ids)
    }

    /// Add `vectors` of dimensionality `dim` with the given external ids.
    /// On a lazy index this locks the dim; on an already-dim'd index
    /// `dim` must match.
    ///
    /// This is the form bindings with shape information (e.g. the Python
    /// binding receiving a 2D ndarray) should use, since a flat
    /// `&[f32]` alone is ambiguous about shape.
    ///
    /// Returns
    /// [`AddError::VectorBufferNotMultipleOfDim`](crate::AddError::VectorBufferNotMultipleOfDim),
    /// [`AddError::IdsCountMismatch`](crate::AddError::IdsCountMismatch),
    /// [`AddError::IdAlreadyPresent`](crate::AddError::IdAlreadyPresent),
    /// or any error returned by
    /// [`TurboQuantIndex::add_2d`](crate::TurboQuantIndex::add_2d).
    pub fn add_with_ids_2d(
        &mut self,
        vectors: &[f32],
        dim: usize,
        ids: &[u64],
    ) -> Result<(), AddError> {
        if dim == 0 || vectors.len() % dim != 0 {
            return Err(AddError::VectorBufferNotMultipleOfDim {
                vectors_len: vectors.len(),
                dim,
            });
        }
        let n = vectors.len() / dim;
        if ids.len() != n {
            return Err(AddError::IdsCountMismatch {
                expected: n,
                got: ids.len(),
            });
        }

        // Validate all ids up-front so a partial failure is impossible.
        // Reject both ids already in the index and duplicates within
        // this call.
        let mut seen_this_call: std::collections::HashSet<u64> =
            std::collections::HashSet::with_capacity(n);
        for &id in ids {
            if self.id_to_slot.contains_key(&id) || !seen_this_call.insert(id) {
                return Err(AddError::IdAlreadyPresent(id));
            }
        }

        // Capture the slot the first new vector will occupy BEFORE we
        // touch the inner index, then run the inner add first. If `add_2d`
        // returns Err (e.g. DimMismatch on a committed-dim index) the ID
        // tables stay untouched — otherwise we'd leave `n` ghost entries
        // pointing at slots that don't exist in the inner index, and the
        // next search_with_allowlist / remove would corrupt further.
        let base_slot = self.inner.len();
        self.inner.add_2d(vectors, dim)?;

        self.id_to_slot.reserve(n);
        self.slot_to_id.reserve(n);
        for (i, &id) in ids.iter().enumerate() {
            self.id_to_slot.insert(id, base_slot + i);
        }
        self.slot_to_id.extend_from_slice(ids);

        Ok(())
    }

    /// Remove the vector with the given external id.
    ///
    /// Returns `true` if the id was present and removed, `false`
    /// otherwise. O(1) via the inner [`TurboQuantIndex::swap_remove`].
    pub fn remove(&mut self, id: u64) -> bool {
        let Some(slot) = self.id_to_slot.remove(&id) else {
            return false;
        };
        let last = self.slot_to_id.len() - 1;

        let moved_from = self.inner.swap_remove(slot);
        debug_assert_eq!(moved_from, last);

        // Mirror the swap-and-pop in our tables.
        if slot != last {
            let moved_id = self.slot_to_id[last];
            self.slot_to_id[slot] = moved_id;
            // The previously-last id now lives at `slot`.
            self.id_to_slot.insert(moved_id, slot);
        }
        self.slot_to_id.pop();

        true
    }

    /// Remove every id in `ids`, returning how many were present. Missing
    /// ids are skipped (callers that require strictness compare the count).
    /// One bound call amortizes per-id call overhead for large batches.
    ///
    /// Local appliance extension on top of upstream v0.9.0 — see
    /// `LOCAL_PATCHES.md` at the repository root.
    pub fn remove_many(&mut self, ids: &[u64]) -> usize {
        let mut removed = 0;
        for &id in ids {
            if self.remove(id) {
                removed += 1;
            }
        }
        removed
    }

    /// Packed code bytes per vector, or `None` before the dim commits.
    ///
    /// Local appliance extension on top of upstream v0.9.0.
    pub fn bytes_per_vector(&self) -> Option<usize> {
        self.inner.bytes_per_vector()
    }

    /// Fingerprint of the calibrated coordinate system (dim, bit width,
    /// frozen TQ+ state). Encoded rows are only portable between indexes
    /// with equal fingerprints.
    ///
    /// Local appliance extension on top of upstream v0.9.0.
    pub fn calibration_fingerprint(&self) -> u64 {
        self.inner.calibration_fingerprint()
    }

    /// Whether a data-dependent TQ+ calibration was fitted (small first
    /// adds commit an identity calibration with no TQ+ recall lift).
    ///
    /// Local appliance extension on top of upstream v0.9.0.
    pub fn calibration_fitted(&self) -> bool {
        self.inner.calibration_fitted()
    }

    /// Export the packed code bytes and per-vector scale for each id, in
    /// id order, for incremental delta persistence. Returns
    /// `(codes, scales)` where `codes.len() == ids.len() * bytes_per_vector`.
    ///
    /// Local appliance extension on top of upstream v0.9.0.
    pub fn export_encoded(&self, ids: &[u64]) -> Result<(Vec<u8>, Vec<f32>), EncodedRowsError> {
        let bytes_per_vec = self
            .inner
            .bytes_per_vector()
            .ok_or(EncodedRowsError::DimNotCommitted)?;
        let mut codes = Vec::with_capacity(ids.len() * bytes_per_vec);
        let mut scales = Vec::with_capacity(ids.len());
        for &id in ids {
            let slot = *self
                .id_to_slot
                .get(&id)
                .ok_or(EncodedRowsError::UnknownId(id))?;
            let (row, scale) = self.inner.export_row(slot);
            codes.extend_from_slice(row);
            scales.push(scale);
        }
        Ok((codes, scales))
    }

    /// Decode the stored rows for `ids` (in id order) into rotated-space
    /// float vectors scaled by their per-row scales — exactly the vectors
    /// the search kernel scores rotated queries against, without the
    /// kernel's uint8 LUT quantization. Returns a flat `(ids.len() * dim)`
    /// buffer. See [`TurboQuantIndex::reconstruct_row_into`] for the math.
    ///
    /// Local appliance extension on top of upstream v0.9.0 — see
    /// `LOCAL_PATCHES.md` at the repository root.
    pub fn reconstruct_rows(&self, ids: &[u64]) -> Result<Vec<f32>, EncodedRowsError> {
        let dim = self
            .inner
            .dim_opt()
            .ok_or(EncodedRowsError::DimNotCommitted)?;
        let mut out = vec![0.0f32; ids.len() * dim];
        for (row_index, &id) in ids.iter().enumerate() {
            let slot = *self
                .id_to_slot
                .get(&id)
                .ok_or(EncodedRowsError::UnknownId(id))?;
            self.inner
                .reconstruct_row_into(slot, &mut out[row_index * dim..(row_index + 1) * dim]);
        }
        Ok(out)
    }

    /// Decode every stored row in slot order, returning `(ids, rows)` where
    /// `rows.len() == ids.len() * dim`. Cheaper than [`Self::reconstruct_rows`]
    /// over all ids because it skips per-id hash lookups; used to build
    /// GPU-resident dequantized copies of the whole index. Returns empty
    /// buffers when the dim has not committed.
    ///
    /// Local appliance extension on top of upstream v0.9.0 — see
    /// `LOCAL_PATCHES.md` at the repository root.
    pub fn reconstruct_all(&self) -> (Vec<u64>, Vec<f32>) {
        let Some(dim) = self.inner.dim_opt() else {
            return (Vec::new(), Vec::new());
        };
        let mut rows = vec![0.0f32; self.slot_to_id.len() * dim];
        for (slot, chunk) in rows.chunks_mut(dim).enumerate() {
            self.inner.reconstruct_row_into(slot, chunk);
        }
        (self.slot_to_id.clone(), rows)
    }

    /// A copy of the deterministic rotation matrix, or `None` before the
    /// dim commits. See [`TurboQuantIndex::rotation_matrix_copy`].
    ///
    /// Local appliance extension on top of upstream v0.9.0.
    pub fn rotation_matrix(&self) -> Option<Vec<f32>> {
        self.inner.rotation_matrix_copy()
    }

    /// Import pre-encoded rows by external id with upsert semantics:
    /// existing ids are overwritten in place, new ids are appended.
    /// Returns `(replaced, appended)`.
    ///
    /// All validation happens before any mutation so a failure cannot
    /// leave the index partially updated. The rows must come from an
    /// index with an equal [`Self::calibration_fingerprint`]; the caller
    /// is responsible for checking that (this method cannot see the
    /// exporting index).
    ///
    /// Local appliance extension on top of upstream v0.9.0.
    pub fn add_encoded(
        &mut self,
        ids: &[u64],
        codes: &[u8],
        scales: &[f32],
    ) -> Result<(usize, usize), EncodedRowsError> {
        let bytes_per_vec = self
            .inner
            .bytes_per_vector()
            .ok_or(EncodedRowsError::DimNotCommitted)?;
        if codes.len() != ids.len() * bytes_per_vec {
            return Err(EncodedRowsError::CodesLengthMismatch {
                expected: ids.len() * bytes_per_vec,
                got: codes.len(),
            });
        }
        if scales.len() != ids.len() {
            return Err(EncodedRowsError::ScalesCountMismatch {
                expected: ids.len(),
                got: scales.len(),
            });
        }
        for (index, &scale) in scales.iter().enumerate() {
            if !scale.is_finite() {
                return Err(EncodedRowsError::NonFiniteScale { index, value: scale });
            }
        }
        let mut seen_this_call: std::collections::HashSet<u64> =
            std::collections::HashSet::with_capacity(ids.len());
        for &id in ids {
            if !seen_this_call.insert(id) {
                return Err(EncodedRowsError::DuplicateId(id));
            }
        }

        let mut replaced = 0;
        let mut appended = 0;
        for (row_index, &id) in ids.iter().enumerate() {
            let row = &codes[row_index * bytes_per_vec..(row_index + 1) * bytes_per_vec];
            let scale = scales[row_index];
            match self.id_to_slot.get(&id) {
                Some(&slot) => {
                    self.inner.overwrite_row(slot, row, scale);
                    replaced += 1;
                }
                None => {
                    let slot = self.inner.append_encoded_row(row, scale);
                    debug_assert_eq!(slot, self.slot_to_id.len());
                    self.slot_to_id.push(id);
                    self.id_to_slot.insert(id, slot);
                    appended += 1;
                }
            }
        }
        Ok((replaced, appended))
    }

    /// Upsert `n = vectors.len() / dim` float vectors by external id:
    /// existing ids are re-encoded and overwritten in their current slots
    /// (no swap churn), new ids are appended. Returns
    /// `(replaced, appended)`.
    ///
    /// On an empty index this delegates to [`Self::add_with_ids_2d`] so
    /// the first batch fits calibration normally; afterwards vectors are
    /// encoded against the frozen coordinate system, never refitting it.
    /// Duplicate ids within one call are rejected with
    /// [`AddError::IdAlreadyPresent`]. All validation happens before any
    /// mutation.
    ///
    /// Local appliance extension on top of upstream v0.9.0 — see
    /// `LOCAL_PATCHES.md` at the repository root.
    pub fn upsert_with_ids_2d(
        &mut self,
        vectors: &[f32],
        dim: usize,
        ids: &[u64],
    ) -> Result<(usize, usize), AddError> {
        if dim == 0 || vectors.len() % dim != 0 {
            return Err(AddError::VectorBufferNotMultipleOfDim {
                vectors_len: vectors.len(),
                dim,
            });
        }
        let n = vectors.len() / dim;
        if ids.len() != n {
            return Err(AddError::IdsCountMismatch {
                expected: n,
                got: ids.len(),
            });
        }
        if self.inner.is_empty() {
            return self.add_with_ids_2d(vectors, dim, ids).map(|()| (0, n));
        }
        if let Some(existing_dim) = self.inner.dim_opt() {
            if dim != existing_dim {
                return Err(AddError::DimMismatch {
                    existing: existing_dim,
                    got: dim,
                });
            }
        }
        if let Some((vector_index, coord_index, value)) =
            crate::first_invalid_coord(vectors, dim)
        {
            return Err(AddError::InvalidInputValue {
                vector_index,
                coord_index,
                value,
            });
        }
        let mut seen_this_call: std::collections::HashSet<u64> =
            std::collections::HashSet::with_capacity(n);
        for &id in ids {
            if !seen_this_call.insert(id) {
                return Err(AddError::IdAlreadyPresent(id));
            }
        }
        let (packed, scales) = self.inner.encode_rows(vectors, n, dim);
        let bytes_per_vec = self
            .inner
            .bytes_per_vector()
            .expect("non-empty index has a committed dim");
        let mut replaced = 0;
        let mut appended = 0;
        for (row_index, &id) in ids.iter().enumerate() {
            let row = &packed[row_index * bytes_per_vec..(row_index + 1) * bytes_per_vec];
            let scale = scales[row_index];
            match self.id_to_slot.get(&id) {
                Some(&slot) => {
                    self.inner.overwrite_row(slot, row, scale);
                    replaced += 1;
                }
                None => {
                    let slot = self.inner.append_encoded_row(row, scale);
                    debug_assert_eq!(slot, self.slot_to_id.len());
                    self.slot_to_id.push(id);
                    self.id_to_slot.insert(id, slot);
                    appended += 1;
                }
            }
        }
        Ok((replaced, appended))
    }

    /// Search for the top-`k` nearest ids for each query.
    ///
    /// Returns `(scores, ids)` flattened row-major: row `qi` occupies
    /// indices `qi * k .. (qi + 1) * k` in both arrays. Number of rows
    /// is `queries.len() / dim`.
    pub fn search(&self, queries: &[f32], k: usize) -> (Vec<f32>, Vec<u64>) {
        self.search_with_allowlist(queries, k, None)
    }

    /// Search restricted to the given `allowlist` of external ids.
    ///
    /// `allowlist`, when `Some`, restricts the returned top-`k` to ids in the
    /// allowlist. The effective result count per query is
    /// `min(k, allowlist.len())` (after de-duplication).
    ///
    /// Panics if `allowlist` is empty or contains an id not currently
    /// present in the index. Duplicate ids in the allowlist are accepted
    /// and deduplicated.
    ///
    /// Passing `allowlist = None` is equivalent to [`Self::search`].
    pub fn search_with_allowlist(
        &self,
        queries: &[f32],
        k: usize,
        allowlist: Option<&[u64]>,
    ) -> (Vec<f32>, Vec<u64>) {
        let mask_buf: Option<Vec<bool>> = allowlist.map(|ids| {
            assert!(!ids.is_empty(), "allowlist is empty");
            let mut mask = vec![false; self.inner.len()];
            for &id in ids {
                let slot = match self.id_to_slot.get(&id) {
                    Some(&s) => s,
                    None => panic!("id {id} in allowlist is not present in index"),
                };
                mask[slot] = true;
            }
            mask
        });

        let res = self
            .inner
            .search_with_mask(queries, k, mask_buf.as_deref());

        let mut ids = Vec::with_capacity(res.indices.len());
        for &slot in &res.indices {
            // Inner returns i64 slot indices. Convert via slot_to_id.
            // Slot indices are always in-bounds (the kernel never
            // returns negative or out-of-range values for a valid
            // index), so this lookup cannot fail in practice; the
            // bounds check makes that invariant crash-loud if it ever
            // does.
            let id = self.slot_to_id[slot as usize];
            ids.push(id);
        }
        (res.scores, ids)
    }

    /// True if the index currently contains a vector with this id.
    pub fn contains(&self, id: u64) -> bool {
        self.id_to_slot.contains_key(&id)
    }

    pub fn len(&self) -> usize {
        self.slot_to_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slot_to_id.is_empty()
    }

    /// Vector dimensionality, or `0` if the index is lazy and hasn't
    /// seen an add yet (matches [`TurboQuantIndex::dim`] semantics).
    pub fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Vector dimensionality as an [`Option`], where `None` means the
    /// index is lazy and uncommitted.
    pub fn dim_opt(&self) -> Option<usize> {
        self.inner.dim_opt()
    }

    pub fn bit_width(&self) -> usize {
        self.inner.bit_width()
    }

    /// Eagerly populate the inner search caches. See
    /// [`TurboQuantIndex::prepare`].
    pub fn prepare(&self) {
        self.inner.prepare();
    }

    /// Serialize to a `.tvim` file — the inner quantized index plus the
    /// id-map side-tables. Round-trips exactly through [`Self::load`].
    pub fn write(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        // Mirror TurboQuantIndex::write: dim=0 means lazy-uninitialized.
        io::write_id_map(
            path,
            self.inner.bit_width(),
            self.inner.dim_opt().unwrap_or(0),
            self.inner.len(),
            self.inner.packed_codes(),
            self.inner.scales(),
            self.inner.tqplus_shift(),
            self.inner.tqplus_scale(),
            &self.slot_to_id,
        )
    }

    /// Load a `.tvim` file previously written by [`Self::write`].
    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let (bit_width, dim, n_vectors, packed_codes, scales, tqplus_shift, tqplus_scale, slot_to_id) =
            io::load_id_map(path)?;
        let dim_opt = if dim == 0 { None } else { Some(dim) };
        let inner = TurboQuantIndex::from_parts(
            dim_opt, bit_width, n_vectors, packed_codes, scales, tqplus_shift, tqplus_scale,
        );
        let id_to_slot: HashMap<u64, usize> = slot_to_id
            .iter()
            .enumerate()
            .map(|(slot, &id)| (id, slot))
            .collect();
        // Reject corrupt files where the id table contains duplicates —
        // this would desync the two tables.
        if id_to_slot.len() != slot_to_id.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "duplicate ids in .tvim file",
            ));
        }
        Ok(Self {
            inner,
            slot_to_id,
            id_to_slot,
        })
    }
}
