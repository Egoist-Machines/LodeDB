use lodedb_core::{
    engine::{CoreEngine as RustCoreEngine, IngestPlan, QueryPlan},
    CoreDocument, CoreError, CoreErrorCode, CoreIndexConfig, CoreIndexCreateOptions,
    CoreMutationResult, CoreOpenOptions, CoreQuery, CoreRoutePolicy, CoreSearchResults,
    CoreSecurityOptions, CoreStats, CoreVectorDocument,
};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyType;
use serde::de::DeserializeOwned;
use serde_json::Value;

fn not_contiguous_err(kind: &str) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!(
        "{kind} must be C-contiguous; call np.ascontiguousarray(...) first",
    ))
}

/// Map a numpy shape error from reassembling search results into a typed
/// RuntimeError. The result dimensions are derived from the core's own
/// output, so this never fires today — but a future change to result shaping
/// would otherwise surface as an uncatchable panic instead of a catchable
/// exception.
fn shape_err(e: numpy::ndarray::ShapeError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!(
        "internal error: malformed search result shape: {e}"
    ))
}

/// Reject NaN / Inf / overflow-magnitude query coordinates with a typed
/// `ValueError`. The core `search` panics on invalid values (its documented
/// Rust contract), which would otherwise surface to Python as an uncatchable
/// `PanicException`. `add` already maps the same condition to `ValueError`;
/// this keeps `search` consistent.
fn validate_queries(values: &[f32], dim: usize) -> PyResult<()> {
    if let Some((vi, ci, v)) = turbovec_core::first_invalid_coord(values, dim) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "invalid query value at query {vi}, coord {ci}: {v} \
             (must be finite and |value| < 1e16)",
        )));
    }
    Ok(())
}

#[pyclass]
struct TurboQuantIndex {
    inner: turbovec_core::TurboQuantIndex,
}

#[pymethods]
impl TurboQuantIndex {
    /// Construct an index. `dim` is optional: when omitted, the
    /// underlying quantized index is created lazily on the first
    /// `add` call, picking up the dimensionality from the input
    /// array's shape.
    #[new]
    #[pyo3(signature = (dim=None, bit_width=4))]
    fn new(dim: Option<usize>, bit_width: usize) -> PyResult<Self> {
        let inner = match dim {
            Some(d) => turbovec_core::TurboQuantIndex::new(d, bit_width),
            None => turbovec_core::TurboQuantIndex::new_lazy(bit_width),
        }
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn add(&mut self, vectors: PyReadonlyArray2<f32>) -> PyResult<()> {
        let arr = vectors.as_array();
        let dim = arr.ncols();
        let slice = arr
            .as_slice()
            .ok_or_else(|| not_contiguous_err("vectors"))?;
        // `add_2d` handles both eager (dim must match) and lazy (locks
        // dim on first call) cases.
        self.inner
            .add_2d(slice, dim)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Run a top-`k` search against the index.
    ///
    /// `mask`, when given, is a bool array of length `len(self)`. Only slots
    /// with `mask[i] == True` contribute to the returned top-`k`. The
    /// returned result count per query is `min(k, mask.sum())`.
    #[pyo3(signature = (queries, k, *, mask=None))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        k: usize,
        mask: Option<PyReadonlyArray1<bool>>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<i64>>)> {
        let arr = queries.as_array();
        let nq = arr.nrows();
        let q_slice = arr
            .as_slice()
            .ok_or_else(|| not_contiguous_err("queries"))?;
        // Reject wrong-dim queries cleanly. Previously the inner
        // `assert_eq!(queries.len(), nq * dim)` would fire as a Rust
        // panic and surface to Python as a PanicException, not the
        // ValueError users expect for input-shape mismatch.
        if let Some(idx_dim) = self.inner.dim_opt() {
            if arr.ncols() != idx_dim {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "query dim {} does not match index dim {}",
                    arr.ncols(),
                    idx_dim,
                )));
            }
        }
        validate_queries(q_slice, arr.ncols())?;

        let mask_arr = mask.as_ref().map(|m| m.as_array());
        let mask_slice: Option<&[bool]> = match mask_arr.as_ref() {
            Some(m_arr) => {
                let expected = self.inner.len();
                if m_arr.len() != expected {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "mask length {} does not match index size {}",
                        m_arr.len(),
                        expected,
                    )));
                }
                Some(m_arr.as_slice().ok_or_else(|| not_contiguous_err("mask"))?)
            }
            None => None,
        };

        let results = self.inner.search_with_mask(q_slice, k, mask_slice);
        let effective_k = results.k;

        let scores = numpy::ndarray::Array2::from_shape_vec((nq, effective_k), results.scores)
            .map_err(shape_err)?
            .into_pyarray(py);
        let indices = numpy::ndarray::Array2::from_shape_vec((nq, effective_k), results.indices)
            .map_err(shape_err)?
            .into_pyarray(py);

        Ok((scores, indices))
    }

    fn write(&self, path: &str) -> PyResult<()> {
        self.inner
            .write(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))
    }

    #[classmethod]
    fn load(_cls: &Bound<PyType>, path: &str) -> PyResult<Self> {
        let inner = turbovec_core::TurboQuantIndex::load(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;
        Ok(Self { inner })
    }

    /// Warm up the search caches (rotation matrix, Lloyd-Max centroids,
    /// SIMD-blocked code layout) so the first `search` call does not pay
    /// the one-time initialisation cost.
    fn prepare(&self) {
        self.inner.prepare();
    }

    /// Remove the vector at `idx` in O(1) by swapping with the last vector.
    ///
    /// The last vector moves into the deleted slot — order is not
    /// preserved. Returns the old index of the moved vector; equals `idx`
    /// when `idx` was already the last element.
    ///
    /// Raises ``IndexError`` if ``idx`` is out of range.
    fn swap_remove(&mut self, idx: usize) -> PyResult<usize> {
        let len = self.inner.len();
        if idx >= len {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "index {idx} out of range for index of length {len}",
            )));
        }
        Ok(self.inner.swap_remove(idx))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        let dim = self
            .inner
            .dim_opt()
            .map_or_else(|| "None".to_string(), |d| d.to_string());
        format!(
            "turbovec.TurboQuantIndex(dim={}, bit_width={}, n_vectors={})",
            dim,
            self.inner.bit_width(),
            self.inner.len()
        )
    }

    /// Vector dimensionality. Returns ``None`` when the index was
    /// constructed lazily (no ``dim=``) and hasn't seen an add yet;
    /// otherwise an ``int``.
    #[getter]
    fn dim(&self) -> Option<usize> {
        self.inner.dim_opt()
    }

    #[getter]
    fn bit_width(&self) -> usize {
        self.inner.bit_width()
    }
}

#[pyclass]
struct IdMapIndex {
    inner: turbovec_core::IdMapIndex,
}

#[pymethods]
impl IdMapIndex {
    /// Construct an id-mapped index. `dim` is optional: when omitted,
    /// the underlying quantized index is created lazily on the first
    /// `add_with_ids` call, picking up dim from the input array shape.
    #[new]
    #[pyo3(signature = (dim=None, bit_width=4))]
    fn new(dim: Option<usize>, bit_width: usize) -> PyResult<Self> {
        let inner = match dim {
            Some(d) => turbovec_core::IdMapIndex::new(d, bit_width),
            None => turbovec_core::IdMapIndex::new_lazy(bit_width),
        }
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Add `n = vectors.shape[0]` vectors with the given external `ids`.
    ///
    /// `ids` must be a 1-D array of `uint64` with length equal to
    /// `vectors.shape[0]`. Raises `ValueError` if any id is already
    /// present or if the lengths don't match. On a lazy index, this
    /// call commits the dimensionality from `vectors.shape[1]`.
    fn add_with_ids(
        &mut self,
        vectors: PyReadonlyArray2<f32>,
        ids: PyReadonlyArray1<u64>,
    ) -> PyResult<()> {
        let v = vectors.as_array();
        let dim = v.ncols();
        let v_slice = v.as_slice().ok_or_else(|| not_contiguous_err("vectors"))?;
        let i = ids.as_array();
        let i_slice = i.as_slice().ok_or_else(|| not_contiguous_err("ids"))?;
        self.inner
            .add_with_ids_2d(v_slice, dim, i_slice)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Remove the vector with external id `id`. Returns `True` if it was
    /// present, `False` otherwise.
    fn remove(&mut self, id: u64) -> bool {
        self.inner.remove(id)
    }

    /// Remove every id in `ids`, returning how many were present. Missing
    /// ids are skipped; strict callers compare the returned count.
    ///
    /// Local appliance extension on top of upstream v0.9.0 — see
    /// `LOCAL_PATCHES.md` at the repository root.
    fn remove_many(&mut self, ids: PyReadonlyArray1<u64>) -> PyResult<usize> {
        let i = ids.as_array();
        let i_slice = i.as_slice().ok_or_else(|| not_contiguous_err("ids"))?;
        Ok(self.inner.remove_many(i_slice))
    }

    /// Packed code bytes per vector, or `None` before the dim commits.
    ///
    /// Local appliance extension on top of upstream v0.9.0.
    fn bytes_per_vector(&self) -> Option<usize> {
        self.inner.bytes_per_vector()
    }

    /// Whether a data-dependent TQ+ calibration was fitted (small first
    /// adds commit an identity calibration that permanently misses the
    /// TQ+ recall lift).
    ///
    /// Local appliance extension on top of upstream v0.9.0.
    fn calibration_fitted(&self) -> bool {
        self.inner.calibration_fitted()
    }

    /// Upsert vectors by external id: existing ids are re-encoded and
    /// overwritten in place, new ids are appended. Returns
    /// `(replaced, appended)`. On an empty index this behaves like
    /// `add_with_ids`; afterwards encoding reuses the frozen calibrated
    /// coordinate system.
    ///
    /// Local appliance extension on top of upstream v0.9.0.
    fn upsert_with_ids(
        &mut self,
        vectors: PyReadonlyArray2<f32>,
        ids: PyReadonlyArray1<u64>,
    ) -> PyResult<(usize, usize)> {
        let v = vectors.as_array();
        let dim = v.ncols();
        let v_slice = v.as_slice().ok_or_else(|| not_contiguous_err("vectors"))?;
        let i = ids.as_array();
        let i_slice = i.as_slice().ok_or_else(|| not_contiguous_err("ids"))?;
        self.inner
            .upsert_with_ids_2d(v_slice, dim, i_slice)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Fingerprint of the calibrated coordinate system (dim, bit width,
    /// frozen TQ+ state). Encoded rows are only portable between indexes
    /// with equal fingerprints.
    ///
    /// Local appliance extension on top of upstream v0.9.0.
    fn calibration_fingerprint(&self) -> u64 {
        self.inner.calibration_fingerprint()
    }

    /// Export packed code bytes and per-vector scales for `ids`, in id
    /// order, for incremental delta persistence. Returns
    /// `(codes, scales)` shaped `(n, bytes_per_vector)` `uint8` and
    /// `(n,)` `float32`. Raises `KeyError` for an unknown id.
    ///
    /// Local appliance extension on top of upstream v0.9.0.
    fn export_encoded<'py>(
        &self,
        py: Python<'py>,
        ids: PyReadonlyArray1<u64>,
    ) -> PyResult<(Bound<'py, PyArray2<u8>>, Bound<'py, PyArray1<f32>>)> {
        let i = ids.as_array();
        let i_slice = i.as_slice().ok_or_else(|| not_contiguous_err("ids"))?;
        let (codes, scales) = self.inner.export_encoded(i_slice).map_err(|e| match e {
            turbovec_core::EncodedRowsError::UnknownId(_) => {
                pyo3::exceptions::PyKeyError::new_err(e.to_string())
            }
            _ => pyo3::exceptions::PyValueError::new_err(e.to_string()),
        })?;
        let bytes_per_vec = self
            .inner
            .bytes_per_vector()
            .expect("export_encoded succeeded so dim is committed");
        let codes_arr =
            numpy::ndarray::Array2::from_shape_vec((i_slice.len(), bytes_per_vec), codes)
                .map_err(shape_err)?
                .into_pyarray(py);
        let scales_arr = numpy::ndarray::Array1::from_vec(scales).into_pyarray(py);
        Ok((codes_arr, scales_arr))
    }

    /// Import pre-encoded rows by external id with upsert semantics:
    /// existing ids are overwritten in place, new ids are appended.
    /// Returns `(replaced, appended)`. Validation happens before any
    /// mutation. The rows must come from an index whose
    /// `calibration_fingerprint()` equals this index's; callers check
    /// that before calling.
    ///
    /// Local appliance extension on top of upstream v0.9.0.
    fn add_encoded(
        &mut self,
        ids: PyReadonlyArray1<u64>,
        codes: PyReadonlyArray2<u8>,
        scales: PyReadonlyArray1<f32>,
    ) -> PyResult<(usize, usize)> {
        let i = ids.as_array();
        let i_slice = i.as_slice().ok_or_else(|| not_contiguous_err("ids"))?;
        let c = codes.as_array();
        let c_slice = c.as_slice().ok_or_else(|| not_contiguous_err("codes"))?;
        let s = scales.as_array();
        let s_slice = s.as_slice().ok_or_else(|| not_contiguous_err("scales"))?;
        self.inner
            .add_encoded(i_slice, c_slice, s_slice)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Decode the stored rows for `ids` (in id order) into rotated-space
    /// float vectors scaled by their per-row scales — exactly the vectors
    /// the search kernel scores rotated queries against, without the
    /// kernel's uint8 LUT quantization. Returns an `(n, dim)` `float32`
    /// array. Score contract: `index.search(q, k)` scores equal
    /// `(q @ rotation_matrix().T) @ reconstruct_rows(ids).T` up to LUT
    /// quantization error. Raises `KeyError` for an unknown id.
    ///
    /// Local appliance extension on top of upstream v0.9.0.
    fn reconstruct_rows<'py>(
        &self,
        py: Python<'py>,
        ids: PyReadonlyArray1<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let i = ids.as_array();
        let i_slice = i.as_slice().ok_or_else(|| not_contiguous_err("ids"))?;
        let rows = self.inner.reconstruct_rows(i_slice).map_err(|e| match e {
            turbovec_core::EncodedRowsError::UnknownId(_) => {
                pyo3::exceptions::PyKeyError::new_err(e.to_string())
            }
            _ => pyo3::exceptions::PyValueError::new_err(e.to_string()),
        })?;
        let dim = self
            .inner
            .dim_opt()
            .expect("reconstruct_rows succeeded so dim is committed");
        Ok(
            numpy::ndarray::Array2::from_shape_vec((i_slice.len(), dim), rows)
                .map_err(shape_err)?
                .into_pyarray(py),
        )
    }

    /// Decode every stored row in slot order, returning `(ids, rows)` as
    /// `(n,)` `uint64` and `(n, dim)` `float32` arrays. Cheaper than
    /// `reconstruct_rows` over all ids; used to build GPU-resident
    /// dequantized copies. Empty arrays before the dim commits.
    ///
    /// Local appliance extension on top of upstream v0.9.0.
    fn reconstruct_all<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray1<u64>>, Bound<'py, PyArray2<f32>>)> {
        let (ids, rows) = self.inner.reconstruct_all();
        let dim = self.inner.dim_opt().unwrap_or(0);
        let n = ids.len();
        let ids_arr = numpy::ndarray::Array1::from_vec(ids).into_pyarray(py);
        let rows_arr = numpy::ndarray::Array2::from_shape_vec((n, dim), rows)
            .map_err(shape_err)?
            .into_pyarray(py);
        Ok((ids_arr, rows_arr))
    }

    /// A copy of the deterministic rotation matrix as a `(dim, dim)`
    /// `float32` array, or `None` before the dim commits. Queries are
    /// rotated as `q_rot = q @ rotation.T`, matching the search path's
    /// GEMM exactly.
    ///
    /// Local appliance extension on top of upstream v0.9.0.
    fn rotation_matrix<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f32>>>> {
        match self.inner.rotation_matrix() {
            None => Ok(None),
            Some(rotation) => {
                let dim = self
                    .inner
                    .dim_opt()
                    .expect("rotation matrix exists so dim is committed");
                Ok(Some(
                    numpy::ndarray::Array2::from_shape_vec((dim, dim), rotation)
                        .map_err(shape_err)?
                        .into_pyarray(py),
                ))
            }
        }
    }

    /// Search for the top-`k` nearest external ids for each query.
    ///
    /// `allowlist`, when given, is a `uint64` array of external ids; the
    /// returned top-`k` is restricted to ids in this list. The returned
    /// result count per query is `min(k, len(allowlist))` (after
    /// de-duplication).
    ///
    /// Returns `(scores, ids)` as `(nq, effective_k)` arrays, `ids` typed
    /// `uint64`. Raises `ValueError` for an empty allowlist and `KeyError`
    /// if any allowlist id is not present in the index.
    #[pyo3(signature = (queries, k, *, allowlist=None))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        k: usize,
        allowlist: Option<PyReadonlyArray1<u64>>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<u64>>)> {
        let arr = queries.as_array();
        let nq = arr.nrows();
        let q_slice = arr
            .as_slice()
            .ok_or_else(|| not_contiguous_err("queries"))?;
        if let Some(idx_dim) = self.inner.dim_opt() {
            if arr.ncols() != idx_dim {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "query dim {} does not match index dim {}",
                    arr.ncols(),
                    idx_dim,
                )));
            }
        }
        validate_queries(q_slice, arr.ncols())?;

        let allow_arr = allowlist.as_ref().map(|a| a.as_array());
        let allow_slice: Option<&[u64]> = match allow_arr.as_ref() {
            Some(a_arr) => {
                if a_arr.is_empty() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "allowlist is empty",
                    ));
                }
                let slice = a_arr
                    .as_slice()
                    .ok_or_else(|| not_contiguous_err("allowlist"))?;
                let mut unknown: Vec<u64> = Vec::new();
                for &id in slice {
                    if !self.inner.contains(id) {
                        if unknown.len() < 5 {
                            unknown.push(id);
                        } else {
                            unknown.push(id);
                            break;
                        }
                    }
                }
                if !unknown.is_empty() {
                    let preview: Vec<u64> = unknown.iter().take(5).copied().collect();
                    return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                        "allowlist contains id(s) not present in index: {:?}{}",
                        preview,
                        if unknown.len() > 5 { ", ..." } else { "" },
                    )));
                }
                Some(slice)
            }
            None => None,
        };

        let (scores, ids) = self.inner.search_with_allowlist(q_slice, k, allow_slice);
        // For empty queries (nq=0), match TurboQuantIndex's shape
        // contract: effective_k is `min(k, n_vectors, n_allowed)`. The
        // kernel dedups the allowlist via a packed bool mask for nq>0,
        // so we have to dedup here too — otherwise `allowlist=[1, 1, 1]`
        // returns shape `(0, 3)` for empty queries but `(N, 1)` for
        // non-empty queries, a silent shape divergence.
        let effective_k = if nq == 0 {
            let n_allowed = match allow_slice {
                Some(s) => {
                    let mut seen: std::collections::HashSet<u64> =
                        std::collections::HashSet::with_capacity(s.len());
                    s.iter().filter(|id| seen.insert(**id)).count()
                }
                None => self.inner.len(),
            };
            k.min(self.inner.len()).min(n_allowed)
        } else {
            scores.len() / nq
        };

        let scores_arr = numpy::ndarray::Array2::from_shape_vec((nq, effective_k), scores)
            .map_err(shape_err)?
            .into_pyarray(py);
        let ids_arr = numpy::ndarray::Array2::from_shape_vec((nq, effective_k), ids)
            .map_err(shape_err)?
            .into_pyarray(py);
        Ok((scores_arr, ids_arr))
    }

    fn contains(&self, id: u64) -> bool {
        self.inner.contains(id)
    }

    fn prepare(&self) {
        self.inner.prepare();
    }

    /// Serialize the index and id-map side-tables to a `.tvim` file.
    fn write(&self, path: &str) -> PyResult<()> {
        self.inner
            .write(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))
    }

    /// Load an `IdMapIndex` from a `.tvim` file previously written by
    /// [`IdMapIndex.write`].
    #[classmethod]
    fn load(_cls: &Bound<PyType>, path: &str) -> PyResult<Self> {
        let inner = turbovec_core::IdMapIndex::load(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;
        Ok(Self { inner })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        let dim = self
            .inner
            .dim_opt()
            .map_or_else(|| "None".to_string(), |d| d.to_string());
        format!(
            "turbovec.IdMapIndex(dim={}, bit_width={}, n_vectors={})",
            dim,
            self.inner.bit_width(),
            self.inner.len()
        )
    }

    fn __contains__(&self, id: u64) -> bool {
        self.inner.contains(id)
    }

    /// Vector dimensionality. Returns ``None`` when the index was
    /// constructed lazily and hasn't seen an add yet; otherwise ``int``.
    #[getter]
    fn dim(&self) -> Option<usize> {
        self.inner.dim_opt()
    }

    #[getter]
    fn bit_width(&self) -> usize {
        self.inner.bit_width()
    }
}

/// Exact late-interaction MaxSim scoring of one multi-vector query against a set
/// of candidate documents.
///
/// `query` is a `(n_query, dim)` float32 matrix. `docs` is the candidate
/// documents' patch vectors concatenated row-major into one `(total_patches,
/// dim)` float32 matrix, partitioned by `doc_patch_counts` (patches per document,
/// in order; must sum to `total_patches`). Returns one float32 score per
/// document, in document order: the sum over query tokens of the maximum dot
/// product against that document's patches. Vectors are assumed L2-normalized, so
/// each dot is a cosine similarity. Documents are scored in parallel.
#[pyfunction]
fn maxsim_scores<'py>(
    py: Python<'py>,
    query: PyReadonlyArray2<f32>,
    docs: PyReadonlyArray2<f32>,
    doc_patch_counts: PyReadonlyArray1<i64>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let q_arr = query.as_array();
    let n_query = q_arr.nrows();
    let dim = q_arr.ncols();
    let q_slice = q_arr
        .as_slice()
        .ok_or_else(|| not_contiguous_err("query"))?;

    let d_arr = docs.as_array();
    let total_patches = d_arr.nrows();
    if d_arr.ncols() != dim {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "query dim {} does not match docs dim {}",
            dim,
            d_arr.ncols(),
        )));
    }
    let d_slice = d_arr.as_slice().ok_or_else(|| not_contiguous_err("docs"))?;

    let counts_arr = doc_patch_counts.as_array();
    let counts_slice = counts_arr
        .as_slice()
        .ok_or_else(|| not_contiguous_err("doc_patch_counts"))?;
    let mut counts: Vec<usize> = Vec::with_capacity(counts_slice.len());
    for &value in counts_slice {
        if value < 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "doc_patch_counts must be non-negative",
            ));
        }
        counts.push(value as usize);
    }
    let sum: usize = counts.iter().sum();
    if sum != total_patches {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "doc_patch_counts sum {sum} does not match docs row count {total_patches}",
        )));
    }

    // The GIL is held while the kernel reads the borrowed NumPy buffers. A
    // PyReadonlyArray keeps the arrays alive but does NOT stop other Python threads
    // from mutating the same memory, so releasing the GIL here would let the kernel
    // read moving buffers (unsound for a public binding). Holding it keeps the read
    // of `q_slice` / `d_slice` consistent; the kernel is a bounded GEMM + reduction.
    let scores = turbovec_core::maxsim_scores(q_slice, n_query, dim, d_slice, &counts);
    Ok(scores.into_pyarray(py))
}

/// Private Python-owned LodeDB native engine handle.
#[pyclass(name = "CoreEngine", unsendable)]
struct PyCoreEngine {
    inner: RustCoreEngine,
}

#[pymethods]
impl PyCoreEngine {
    #[new]
    fn new() -> Self {
        Self {
            inner: RustCoreEngine::new_in_memory(),
        }
    }

    #[staticmethod]
    fn open(options_json: &str) -> PyResult<Self> {
        let options = native_from_json::<CoreOpenOptions>(options_json)?;
        Ok(Self {
            inner: RustCoreEngine::open(options).map_err(native_core_error_to_py)?,
        })
    }

    #[staticmethod]
    fn open_readonly(path: String, options_json: &str) -> PyResult<Self> {
        let options = native_from_json::<CoreOpenOptions>(options_json)?;
        Ok(Self {
            inner: RustCoreEngine::open_readonly(path, options).map_err(native_core_error_to_py)?,
        })
    }

    fn create_index(
        &mut self,
        index_id: String,
        vector_dim: usize,
        bit_width: usize,
    ) -> PyResult<()> {
        self.inner
            .create_index(index_id, vector_dim, bit_width)
            .map_err(native_core_error_to_py)
    }

    fn create_index_with_options(&mut self, options_json: &str) -> PyResult<()> {
        let options = native_from_json::<CoreIndexCreateOptions>(options_json)?;
        self.inner
            .create_index_with_options(options)
            .map_err(native_core_error_to_py)
    }

    fn upsert_vectors(&mut self, index_id: &str, documents_json: &str) -> PyResult<String> {
        let documents = native_from_json::<Vec<CoreVectorDocument>>(documents_json)?;
        native_to_json(
            &self
                .inner
                .upsert_vectors(index_id, &documents)
                .map_err(native_core_error_to_py)?,
        )
    }

    /// Array-input fast path for vector upsert.
    ///
    /// `vectors` is a contiguous `(n, dim)` `f32` matrix; `sidecar_json` is a
    /// parallel array of `{document_id, metadata, text}` objects (no vector
    /// floats). This removes the per-document JSON-encode of the embedding (the
    /// bulk of the durable-add payload) while keeping ids/metadata as a small
    /// batched sidecar.
    fn upsert_vectors_array(
        &mut self,
        index_id: &str,
        vectors: PyReadonlyArray2<f32>,
        sidecar_json: &str,
    ) -> PyResult<String> {
        #[derive(serde::Deserialize)]
        struct VectorRowSidecar {
            document_id: String,
            #[serde(default)]
            metadata: std::collections::BTreeMap<String, String>,
            #[serde(default)]
            text: Option<String>,
        }
        let arr = vectors.as_array();
        let dim = arr.ncols();
        let rows = arr.nrows();
        let slice = arr
            .as_slice()
            .ok_or_else(|| not_contiguous_err("vectors"))?;
        let sidecars = native_from_json::<Vec<VectorRowSidecar>>(sidecar_json)?;
        if sidecars.len() != rows {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "vector row count {} does not match sidecar length {}",
                rows,
                sidecars.len(),
            )));
        }
        if rows > 0 && dim == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "vectors must have a non-zero dimension",
            ));
        }
        let documents: Vec<CoreVectorDocument> = sidecars
            .into_iter()
            .zip(slice.chunks(dim.max(1)))
            .map(|(sidecar, vector)| CoreVectorDocument {
                document_id: sidecar.document_id,
                vector: vector.to_vec(),
                metadata: sidecar.metadata,
                text: sidecar.text,
            })
            .collect();
        native_to_json(
            &self
                .inner
                .upsert_vectors(index_id, &documents)
                .map_err(native_core_error_to_py)?,
        )
    }

    fn delete_documents(&mut self, index_id: &str, document_ids_json: &str) -> PyResult<String> {
        let document_ids = native_from_json::<Vec<String>>(document_ids_json)?;
        native_to_json(
            &self
                .inner
                .delete_documents(index_id, &document_ids)
                .map_err(native_core_error_to_py)?,
        )
    }

    fn update_document_payload(
        &mut self,
        index_id: &str,
        document_id: &str,
        metadata_json: Option<&str>,
        text_json: Option<&str>,
    ) -> PyResult<String> {
        let metadata: Option<std::collections::BTreeMap<String, String>> =
            metadata_json.map(native_from_json).transpose()?;
        let text: Option<Option<String>> = text_json.map(native_from_json).transpose()?;
        native_to_json(
            &self
                .inner
                .update_document_payload(index_id, document_id, metadata, text)
                .map_err(native_core_error_to_py)?,
        )
    }

    fn query_vector(
        &self,
        index_id: &str,
        query_vector_json: &str,
        top_k: usize,
        filter_json: Option<&str>,
    ) -> PyResult<String> {
        let query_vector = native_from_json::<Vec<f32>>(query_vector_json)?;
        let filter = native_optional_value(filter_json)?;
        native_to_json(
            &self
                .inner
                .query_vector(index_id, &query_vector, top_k, filter.as_ref())
                .map_err(native_core_error_to_py)?,
        )
    }

    fn query_vectors_batch(
        &self,
        index_id: &str,
        query_vectors_json: &str,
        top_k: usize,
        filter_json: Option<&str>,
    ) -> PyResult<String> {
        let query_vectors = native_from_json::<Vec<Vec<f32>>>(query_vectors_json)?;
        let filter = native_optional_value(filter_json)?;
        native_to_json(
            &self
                .inner
                .query_vectors_batch(index_id, &query_vectors, top_k, filter.as_ref())
                .map_err(native_core_error_to_py)?,
        )
    }

    /// Array-input fast path for a single vector query.
    ///
    /// Takes the query as a contiguous `f32` array instead of a JSON float list,
    /// removing the per-query Python list-build + `json.dumps` + serde-parse cost
    /// that dominates the single-query path. The top-k result stays JSON (it is
    /// small) so the caller contract is identical to `query_vector`.
    fn query_vector_array(
        &self,
        index_id: &str,
        query_vector: PyReadonlyArray1<f32>,
        top_k: usize,
        filter_json: Option<&str>,
    ) -> PyResult<String> {
        let query_vector = query_embedding_from_array(query_vector)?;
        if !query_vector.is_empty() {
            validate_queries(&query_vector, query_vector.len())?;
        }
        let filter = native_optional_value(filter_json)?;
        native_to_json(
            &self
                .inner
                .query_vector(index_id, &query_vector, top_k, filter.as_ref())
                .map_err(native_core_error_to_py)?,
        )
    }

    /// Array-input fast path for a batch of vector queries.
    ///
    /// `query_vectors` is a contiguous `(n_query, dim)` `f32` matrix. Mirrors
    /// `query_vectors_batch` but skips JSON-encoding the query matrix.
    fn query_vectors_batch_array(
        &self,
        index_id: &str,
        query_vectors: PyReadonlyArray2<f32>,
        top_k: usize,
        filter_json: Option<&str>,
    ) -> PyResult<String> {
        let arr = query_vectors.as_array();
        let dim = arr.ncols();
        let slice = arr
            .as_slice()
            .ok_or_else(|| not_contiguous_err("query_vectors"))?;
        if dim == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "query_vectors must have a non-zero dimension",
            ));
        }
        validate_queries(slice, dim)?;
        let queries: Vec<Vec<f32>> = slice.chunks(dim).map(|row| row.to_vec()).collect();
        let filter = native_optional_value(filter_json)?;
        native_to_json(
            &self
                .inner
                .query_vectors_batch(index_id, &queries, top_k, filter.as_ref())
                .map_err(native_core_error_to_py)?,
        )
    }

    fn prepare_text_upsert(
        &mut self,
        index_id: &str,
        documents_json: &str,
        store_text: bool,
        index_text: bool,
        chunk_character_limit: usize,
    ) -> PyResult<String> {
        let documents = native_from_json::<Vec<CoreDocument>>(documents_json)?;
        native_to_json(
            &self
                .inner
                .prepare_text_upsert(
                    index_id,
                    &documents,
                    store_text,
                    index_text,
                    chunk_character_limit,
                )
                .map_err(native_core_error_to_py)?,
        )
    }

    fn apply_text_upsert(
        &mut self,
        plan_json: &str,
        embeddings_json: &str,
        embedding_time_ms: f64,
    ) -> PyResult<String> {
        let plan = native_from_json::<IngestPlan>(plan_json)?;
        let embeddings = native_from_json::<Vec<Vec<f32>>>(embeddings_json)?;
        native_to_json(
            &self
                .inner
                .apply_text_upsert(&plan, &embeddings, embedding_time_ms)
                .map_err(native_core_error_to_py)?,
        )
    }

    fn apply_text_upsert_array(
        &mut self,
        plan_json: &str,
        embeddings: PyReadonlyArray2<f32>,
        embedding_time_ms: f64,
    ) -> PyResult<String> {
        let plan = native_from_json::<IngestPlan>(plan_json)?;
        let embeddings = embeddings_from_array(embeddings)?;
        native_to_json(
            &self
                .inner
                .apply_text_upsert(&plan, &embeddings, embedding_time_ms)
                .map_err(native_core_error_to_py)?,
        )
    }

    fn prepare_query_text(&self, query: &str, mode: &str) -> PyResult<String> {
        native_to_json(
            &self
                .inner
                .prepare_query_text(query, mode)
                .map_err(native_core_error_to_py)?,
        )
    }

    fn search_embedded_text(
        &self,
        index_id: &str,
        query_plan_json: &str,
        query_embedding_json: Option<&str>,
        top_k: usize,
        filter_json: Option<&str>,
    ) -> PyResult<String> {
        let query_plan = native_from_json::<QueryPlan>(query_plan_json)?;
        let query_embedding: Option<Vec<f32>> =
            query_embedding_json.map(native_from_json).transpose()?;
        let filter = native_optional_value(filter_json)?;
        native_to_json(
            &self
                .inner
                .search_embedded_text(
                    index_id,
                    &query_plan,
                    query_embedding.as_deref(),
                    top_k,
                    filter.as_ref(),
                )
                .map_err(native_core_error_to_py)?,
        )
    }

    fn search_embedded_text_array(
        &self,
        index_id: &str,
        query_plan_json: &str,
        query_embedding: PyReadonlyArray1<f32>,
        top_k: usize,
        filter_json: Option<&str>,
    ) -> PyResult<String> {
        let query_plan = native_from_json::<QueryPlan>(query_plan_json)?;
        let query_embedding = query_embedding_from_array(query_embedding)?;
        let filter = native_optional_value(filter_json)?;
        native_to_json(
            &self
                .inner
                .search_embedded_text(
                    index_id,
                    &query_plan,
                    Some(&query_embedding),
                    top_k,
                    filter.as_ref(),
                )
                .map_err(native_core_error_to_py)?,
        )
    }

    fn stats(&self, index_id: &str) -> PyResult<String> {
        native_to_json(
            &self
                .inner
                .stats(index_id)
                .map_err(native_core_error_to_py)?,
        )
    }

    fn document_token_lists(&self, index_id: &str) -> PyResult<String> {
        native_to_json(
            &self
                .inner
                .document_token_lists(index_id)
                .map_err(native_core_error_to_py)?,
        )
    }

    fn get_document_text(&self, index_id: &str, document_id: &str) -> PyResult<String> {
        native_to_json(
            &self
                .inner
                .get_document_text(index_id, document_id)
                .map_err(native_core_error_to_py)?,
        )
    }

    fn get_document_texts(&self, index_id: &str, document_ids_json: &str) -> PyResult<String> {
        let document_ids = native_from_json::<Vec<String>>(document_ids_json)?;
        native_to_json(
            &self
                .inner
                .get_document_texts(index_id, &document_ids)
                .map_err(native_core_error_to_py)?,
        )
    }

    fn get_document(&self, index_id: &str, document_id: &str) -> PyResult<String> {
        native_to_json(
            &self
                .inner
                .get_document(index_id, document_id)
                .map_err(native_core_error_to_py)?,
        )
    }

    fn list_documents(&self, index_id: &str, filter_json: Option<&str>) -> PyResult<String> {
        let filter = native_optional_value(filter_json)?;
        native_to_json(
            &self
                .inner
                .list_documents(index_id, filter.as_ref())
                .map_err(native_core_error_to_py)?,
        )
    }

    fn persist(&mut self) -> PyResult<()> {
        self.inner.persist().map_err(native_core_error_to_py)
    }

    fn close(&mut self) -> PyResult<()> {
        self.inner.close().map_err(native_core_error_to_py)
    }
}

#[pyfunction]
fn native_core_version() -> &'static str {
    lodedb_core::CORE_VERSION
}

#[pyfunction]
fn native_core_abi_version() -> u32 {
    lodedb_core::NATIVE_CORE_ABI_VERSION
}

#[pyfunction]
fn storage_schema_version() -> u32 {
    lodedb_core::STORAGE_SCHEMA_VERSION
}

#[pyfunction]
fn core_document_to_json(
    document_id: String,
    text: String,
    metadata: std::collections::BTreeMap<String, String>,
) -> PyResult<String> {
    native_to_json(&CoreDocument {
        document_id,
        text,
        metadata,
    })
}

#[pyfunction]
fn round_trip_core_json(type_name: &str, json: &str) -> PyResult<String> {
    match type_name {
        "CoreDocument" => native_round_trip::<CoreDocument>(json),
        "CoreIndexConfig" => native_round_trip::<CoreIndexConfig>(json),
        "CoreMutationResult" => native_round_trip::<CoreMutationResult>(json),
        "CoreOpenOptions" => native_round_trip::<CoreOpenOptions>(json),
        "CoreQuery" => native_round_trip::<CoreQuery>(json),
        "CoreRoutePolicy" => native_round_trip::<CoreRoutePolicy>(json),
        "CoreSearchResults" => native_round_trip::<CoreSearchResults>(json),
        "CoreSecurityOptions" => native_round_trip::<CoreSecurityOptions>(json),
        "CoreStats" => native_round_trip::<CoreStats>(json),
        "CoreVectorDocument" => native_round_trip::<CoreVectorDocument>(json),
        _ => Err(native_core_error_to_py(CoreError::new(
            CoreErrorCode::InvalidArgument,
            "unknown core type",
        ))),
    }
}

fn native_round_trip<T>(json: &str) -> PyResult<String>
where
    T: serde::Serialize + DeserializeOwned,
{
    let value: T = native_from_json(json)?;
    native_to_json(&value)
}

fn native_from_json<T>(json: &str) -> PyResult<T>
where
    T: DeserializeOwned,
{
    serde_json::from_str(json).map_err(|error| {
        native_core_error_to_py(CoreError::new(
            CoreErrorCode::InvalidArgument,
            format!("invalid JSON payload: {error}"),
        ))
    })
}

fn native_to_json<T: serde::Serialize>(value: &T) -> PyResult<String> {
    serde_json::to_string(value).map_err(|error| {
        native_core_error_to_py(CoreError::new(
            CoreErrorCode::Internal,
            format!("failed to serialize core type: {error}"),
        ))
    })
}

fn native_optional_value(json: Option<&str>) -> PyResult<Option<Value>> {
    json.map(native_from_json).transpose()
}

fn embeddings_from_array(embeddings: PyReadonlyArray2<f32>) -> PyResult<Vec<Vec<f32>>> {
    let arr = embeddings.as_array();
    // No rows (e.g. a re-add whose chunks all already exist, so nothing needs
    // embedding) is valid and must not reach `chunks(ncols)`: an empty `(0, 0)`
    // array has `ncols == 0`, and `slice.chunks(0)` panics.
    if arr.nrows() == 0 {
        return Ok(Vec::new());
    }
    let slice = arr
        .as_slice()
        .ok_or_else(|| not_contiguous_err("embeddings"))?;
    Ok(slice
        .chunks(arr.ncols())
        .map(|row| row.to_vec())
        .collect::<Vec<_>>())
}

fn query_embedding_from_array(embedding: PyReadonlyArray1<f32>) -> PyResult<Vec<f32>> {
    let arr = embedding.as_array();
    let slice = arr
        .as_slice()
        .ok_or_else(|| not_contiguous_err("query_embedding"))?;
    Ok(slice.to_vec())
}

fn native_core_error_to_py(error: CoreError) -> PyErr {
    match error.code() {
        CoreErrorCode::InvalidArgument | CoreErrorCode::PlanStale => {
            pyo3::exceptions::PyValueError::new_err(error.to_string())
        }
        CoreErrorCode::NotFound => pyo3::exceptions::PyKeyError::new_err(error.to_string()),
        CoreErrorCode::CorruptStore | CoreErrorCode::Unsupported | CoreErrorCode::Internal => {
            pyo3::exceptions::PyRuntimeError::new_err(error.to_string())
        }
    }
}

#[pymodule]
fn _turbovec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TurboQuantIndex>()?;
    m.add_class::<IdMapIndex>()?;
    m.add_function(wrap_pyfunction!(maxsim_scores, m)?)?;
    m.add_class::<PyCoreEngine>()?;
    m.add_function(wrap_pyfunction!(native_core_version, m)?)?;
    m.add_function(wrap_pyfunction!(native_core_abi_version, m)?)?;
    m.add_function(wrap_pyfunction!(storage_schema_version, m)?)?;
    m.add_function(wrap_pyfunction!(core_document_to_json, m)?)?;
    m.add_function(wrap_pyfunction!(round_trip_core_json, m)?)?;
    Ok(())
}
