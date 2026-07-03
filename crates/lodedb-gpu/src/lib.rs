//! Optional CUDA GPU-resident exact scan for the LodeDB native core.
//!
//! Mirrors the Python `engine/gpu_turbovec.py` path natively: a dequantized copy
//! of the index rows stays resident on the GPU and eligible batched queries are
//! scored with an exact GEMM plus a device top-k. The contract matches the
//! vendored reconstruction parity tests
//! (`third_party/turbovec/turbovec/tests/reconstruction.rs`): rows are exported in
//! ROTATED CALIBRATED space, and
//!
//! ```text
//! score(q, row) = <q @ rotation^T, reconstructed_row>
//! ```
//!
//! reproduces the CPU kernel's calibrated score. Queries are rotated on the host
//! with the deterministic rotation (cheap at batch scale), then a single cuBLAS
//! GEMM produces the batch x corpus score matrix and a custom top-k kernel selects
//! the per-query top-k on device, copying back only the top-k slots and scores.
//! The final ordering is fixed on the host (descending score, ascending stable id
//! on ties), matching the Python GPU path.
//!
//! Resident rows are fp32: at the corpus sizes the native core serves (<= ~1M x
//! 128) the matrix is a few hundred MB at most, and fp32 is both simpler and more
//! faithful than the Python path's fp16 rows. Mutation invalidates the session so
//! the next eligible batch rebuilds; O(changed) patching is a later refinement.
//!
//! This is the single crate in the workspace permitted to use `unsafe`: CUDA
//! kernel launches and the cuBLAS GEMM are inherently unsafe FFI. The core stays
//! `unsafe_code = "forbid"` and consumes only the safe API below. Built with
//! cudarc's `dynamic-loading`, the crate links nothing at build time and engages a
//! GPU only where a driver loads at runtime; every entry point fails closed to an
//! error the caller turns into a CPU fallback.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig};
use cudarc::driver::PushKernelArg;
use cudarc::nvrtc::compile_ptx;

/// Threads per block for the top-k kernel. Must stay a power of two for the
/// shared-memory reduction.
const TOPK_BLOCK: u32 = 256;

/// Threads per block for the scatter (in-place patch) kernel.
const SCATTER_BLOCK: u32 = 256;

/// Rows per chunk in the opt-in fused two-stage top-k (`LODEDB_GPU_FUSED_TOPK`).
///
/// Each stage-1 block stages this many scores into shared memory once, so this
/// bounds the stage-1 shared footprint (`chunk_rows` f32 + `TOPK_BLOCK` reduction
/// slots) and, with `k`, the stage-2 partial count (`num_chunks * k`). Chosen well
/// under the 48 KB default dynamic-shared limit; it is a correctness-neutral tuning
/// knob, so Modal (A10/L40S) is the place to sweep it.
const FUSED_CHUNK_ROWS: usize = 2048;

/// Largest batch the fused path serves. Its stage-1 grid maps `nq` onto
/// `gridDim.y`, which every CUDA device caps at 65,535; a larger batch falls back
/// to the default `topk_argmax` path (which maps the batch onto the ~2^31 `gridDim.x`)
/// rather than failing the launch. `num_chunks` rides `gridDim.x`, so corpus size is
/// unaffected. Batches this large never reach the GPU in practice, but the fallback
/// keeps the opt-in flag from ever regressing a shape the default path handles.
const FUSED_MAX_NQ: usize = 65_535;

/// The CUDA kernels compiled into the session's module.
///
/// `topk_argmax` is the exact per-query top-k over a dense `batch x corpus` score
/// matrix: one block scores one query, each pass finds the block-wide argmax of
/// its score row with a shared-memory reduction (ties resolve to the lower slot),
/// records it in descending order, then masks it to the sentinel for the next
/// pass. `k` is bounded by the corpus size by the caller, so every pass finds a
/// real row.
///
/// `scatter_rows` writes `c` already-baked rows into the resident row buffer at
/// arbitrary slots, so an in-place patch touches only the changed rows. One
/// thread copies one element: `dst[slots[i]*dim + d] = baked[i*dim + d]`.
const KERNELS: &str = r#"
#define NEG_SENTINEL (-3.4028234e38f)
extern "C" __global__ void topk_argmax(
    float* scores,
    const int n,
    const int k,
    unsigned int* out_idx,
    float* out_val)
{
    const int q = blockIdx.x;
    const int t = threadIdx.x;
    const int nt = blockDim.x;
    float* row = scores + (size_t)q * (size_t)n;
    extern __shared__ char smem[];
    float* sval = (float*)smem;
    int* sidx = (int*)(sval + nt);
    for (int pass = 0; pass < k; ++pass) {
        float best = NEG_SENTINEL;
        int bi = -1;
        for (int i = t; i < n; i += nt) {
            float v = row[i];
            if (v > best) { best = v; bi = i; }
        }
        sval[t] = best;
        sidx[t] = bi;
        __syncthreads();
        for (int s = nt >> 1; s > 0; s >>= 1) {
            if (t < s) {
                float ov = sval[t + s];
                int oi = sidx[t + s];
                float cv = sval[t];
                int ci = sidx[t];
                if (ov > cv || (ov == cv && oi >= 0 && (ci < 0 || oi < ci))) {
                    sval[t] = ov;
                    sidx[t] = oi;
                }
            }
            __syncthreads();
        }
        if (t == 0) {
            int idx = sidx[0];
            out_idx[(size_t)q * (size_t)k + pass] = (unsigned int)(idx < 0 ? 0 : idx);
            out_val[(size_t)q * (size_t)k + pass] = sval[0];
            if (idx >= 0) row[idx] = NEG_SENTINEL;
        }
        __syncthreads();
    }
}

extern "C" __global__ void scatter_rows(
    const float* baked,
    const unsigned int* slots,
    float* dst,
    const int c,
    const int dim)
{
    size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total = (size_t)c * (size_t)dim;
    if (tid >= total) return;
    size_t i = tid / (size_t)dim;
    size_t d = tid % (size_t)dim;
    dst[(size_t)slots[i] * (size_t)dim + d] = baked[i * (size_t)dim + d];
}

#define UINT_INVALID (0xffffffffu)

// Strict "is candidate (va,ia) better than (vb,ib)": higher score wins, ties break
// to the lower slot, and an INVALID slot is always worst. This is the exact
// (score desc, slot asc) total order topk_argmax realizes, so the two-stage fused
// path below selects the identical top-k from the identical cuBLAS scores.
__device__ __forceinline__ int fused_gt(float va, unsigned int ia, float vb, unsigned int ib) {
    if (ia == UINT_INVALID) return 0;
    if (ib == UINT_INVALID) return 1;
    if (va > vb) return 1;
    if (va < vb) return 0;
    return ia < ib;
}

// Stage 1 of the opt-in fused top-k. Each block reads ONE query's slice of ONE
// row-chunk of the cuBLAS score matrix into shared memory exactly once, then picks
// that chunk's top-k by (score desc, slot asc) with an on-chip k-pass argmax (no
// global re-read), emitting k (slot, score) partials for the (query, chunk). The
// default topk_argmax rescans the whole n-score row from global memory k times;
// reading each score once instead is the HBM win, and because the scores are the
// cuBLAS output untouched, the selected top-k is bit-identical to the default path.
extern "C" __global__ void chunk_topk(
    const float* scores,          // nq * n, row-major cuBLAS output: scores[q*n + row]
    const int n,
    const int k,
    const int chunk_rows,
    const int num_chunks,
    unsigned int* part_idx,       // nq * num_chunks * k
    float* part_val)
{
    const int chunk = blockIdx.x;
    const int q = blockIdx.y;
    const int t = threadIdx.x;
    const int nt = blockDim.x;
    const long row0 = (long)chunk * (long)chunk_rows;
    int rows_this = (int)((long)n - row0);
    if (rows_this > chunk_rows) rows_this = chunk_rows;
    if (rows_this <= 0) return;

    // Dynamic shared: [chunk scores: chunk_rows f32][reduce vals: nt f32][reduce idx: nt u32].
    extern __shared__ char smem[];
    float* schunk = (float*)smem;
    float* rval = schunk + chunk_rows;
    unsigned int* ridx = (unsigned int*)(rval + nt);

    // Stage the chunk's scores into shared memory: the single read of scores_dev.
    const float* srow = scores + (size_t)q * (size_t)n + (size_t)row0;
    for (int r = t; r < rows_this; r += nt) {
        schunk[r] = srow[r];
    }
    __syncthreads();

    for (int pass = 0; pass < k; ++pass) {
        float best = NEG_SENTINEL;
        unsigned int bi = UINT_INVALID;
        for (int r = t; r < rows_this; r += nt) {
            float v = schunk[r];
            if (v == NEG_SENTINEL) continue;   // an already-selected (masked) row
            unsigned int slot = (unsigned int)(row0 + r);
            if (fused_gt(v, slot, best, bi)) { best = v; bi = slot; }
        }
        rval[t] = best;
        ridx[t] = bi;
        __syncthreads();
        for (int s = nt >> 1; s > 0; s >>= 1) {
            if (t < s) {
                if (fused_gt(rval[t + s], ridx[t + s], rval[t], ridx[t])) {
                    rval[t] = rval[t + s];
                    ridx[t] = ridx[t + s];
                }
            }
            __syncthreads();
        }
        if (t == 0) {
            unsigned int win = ridx[0];
            size_t out = ((size_t)q * (size_t)num_chunks + (size_t)chunk) * (size_t)k + (size_t)pass;
            part_idx[out] = win;
            part_val[out] = (win == UINT_INVALID) ? NEG_SENTINEL : rval[0];
            if (win != UINT_INVALID) {
                schunk[(long)win - row0] = NEG_SENTINEL;   // mask for the next pass
            }
        }
        __syncthreads();
    }
}

// Stage 2 of the opt-in fused top-k. One block per query merges that query's
// num_chunks * k partials into the final top-k under the same (score desc, slot
// asc) order. The partials are a union of per-chunk top-ks, which provably contains
// the global top-k, so the merge equals topk_argmax exactly. Slots are globally
// unique, so masking the selected slot each pass excludes exactly one partial.
extern "C" __global__ void merge_topk(
    unsigned int* part_idx,       // nq * num_chunks * k, consumed/masked in place
    float* part_val,
    const int k,
    const int num_chunks,
    unsigned int* out_idx,        // nq * k
    float* out_val)
{
    const int q = blockIdx.x;
    const int t = threadIdx.x;
    const int nt = blockDim.x;
    const int total = num_chunks * k;

    extern __shared__ char smem[];
    float* sval = (float*)smem;
    unsigned int* sidx = (unsigned int*)(sval + nt);
    const size_t base = (size_t)q * (size_t)total;

    for (int pass = 0; pass < k; ++pass) {
        float best = NEG_SENTINEL;
        unsigned int bi = UINT_INVALID;
        for (int i = t; i < total; i += nt) {
            float v = part_val[base + (size_t)i];
            unsigned int idx = part_idx[base + (size_t)i];
            if (fused_gt(v, idx, best, bi)) { best = v; bi = idx; }
        }
        sval[t] = best;
        sidx[t] = bi;
        __syncthreads();
        for (int s = nt >> 1; s > 0; s >>= 1) {
            if (t < s) {
                if (fused_gt(sval[t + s], sidx[t + s], sval[t], sidx[t])) {
                    sval[t] = sval[t + s];
                    sidx[t] = sidx[t + s];
                }
            }
            __syncthreads();
        }
        unsigned int win = sidx[0];
        if (t == 0) {
            out_idx[(size_t)q * (size_t)k + (size_t)pass] = (win == UINT_INVALID) ? 0u : win;
            out_val[(size_t)q * (size_t)k + (size_t)pass] = sval[0];
        }
        __syncthreads();
        if (win != UINT_INVALID) {
            for (int i = t; i < total; i += nt) {
                if (part_idx[base + (size_t)i] == win) {
                    part_idx[base + (size_t)i] = UINT_INVALID;
                    part_val[base + (size_t)i] = NEG_SENTINEL;
                }
            }
            __syncthreads();
        }
    }
}
"#;

/// Why the GPU scan could not run; every variant is a signal to fall back to the
/// CPU kernel, never an error the caller should surface to the user.
#[derive(Debug, Clone)]
pub enum GpuScanError {
    /// No CUDA driver is loadable (CPU-only host) — do not retry this process.
    Unavailable(String),
    /// The driver loaded but a CUDA call failed (no device, OOM, kernel error).
    Device(String),
    /// The caller passed inconsistent shapes; a programming error, surfaced loudly.
    Invalid(String),
}

impl std::fmt::Display for GpuScanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuScanError::Unavailable(reason) => write!(f, "gpu unavailable: {reason}"),
            GpuScanError::Device(reason) => write!(f, "gpu device error: {reason}"),
            GpuScanError::Invalid(reason) => write!(f, "gpu invalid input: {reason}"),
        }
    }
}

impl std::error::Error for GpuScanError {}

/// Whether opt-in GPU diagnostics are enabled (`LODEDB_GPU_DEBUG` set).
///
/// Mirrors the Python serving layer's one-time backend log: a built session and
/// the first engaged scan each emit one stderr line, so a deployment can confirm
/// the GPU path is actually serving rather than silently falling back.
fn debug_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("LODEDB_GPU_DEBUG").is_some())
}

/// Whether the opt-in fused two-stage top-k is enabled (`LODEDB_GPU_FUSED_TOPK`).
///
/// Off by default: the proven cuBLAS GEMM + `topk_argmax` path stays the default
/// until the fused path is validated on real hardware (A10/L40S), since it cannot
/// be compiled or profiled on a CPU-only host. Enabled for any value other than the
/// usual falsey set. It only changes how the top-k is selected from the identical
/// cuBLAS scores, so results are unchanged; the flag exists to de-risk an
/// unprofiled kernel, not to alter output.
fn fused_topk_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("LODEDB_GPU_FUSED_TOPK") {
        Ok(value) => {
            let v = value.trim().to_ascii_lowercase();
            !(v.is_empty() || v == "0" || v == "off" || v == "false" || v == "no")
        }
        Err(_) => false,
    })
}

/// Returns whether a CUDA driver library can be opened at all.
///
/// Cached after the first probe. cudarc panics (rather than returning an error) if
/// the driver library is entirely absent, so this gate must run before any other
/// CUDA touch; it only attempts a `dlopen` and is documented panic-free.
pub fn cuda_runtime_available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        // SAFETY: `is_culib_present` performs no CUDA API calls. It only tries to
        // `dlopen` the candidate driver libraries and reports whether any opened,
        // and is documented to never panic.
        unsafe { cudarc::driver::sys::is_culib_present() }
    })
}

/// Per-shape device scratch reused across queries of the same `(nq, k)`.
///
/// Allocating the score matrix and top-k buffers, and zeroing them, dominated the
/// per-call cost at the small batch sizes the serving layer drives; caching them
/// keyed by shape turns every steady-state query into copies + kernels with no
/// device allocation. The query buffer is overwritten in place each call.
struct GpuScratch {
    q_dev: CudaSlice<f32>,      // nq * dim, rotated queries
    scores_dev: CudaSlice<f32>, // nq * n, GEMM output (beta=0 overwrites, no zeroing needed)
    idx_dev: CudaSlice<u32>,    // nq * k, top-k slots
    val_dev: CudaSlice<f32>,    // nq * k, top-k scores
    // nq * num_chunks * k stage-1 partials, allocated the first time the opt-in
    // fused top-k runs for this shape; None while only the default path is used.
    part_idx: Option<CudaSlice<u32>>,
    part_val: Option<CudaSlice<f32>>,
}

/// A generation's reconstructed rows resident on the GPU, plus the host-side
/// rotation and stable-id mapping needed to score and label a query batch.
pub struct GpuScanSession {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    // Keeps the loaded module alive for as long as `func`/`scatter_func` reference it.
    _module: Arc<CudaModule>,
    func: CudaFunction,
    // Scatters baked rows into `rows_dev` at given slots for an in-place patch.
    scatter_func: CudaFunction,
    // Opt-in fused two-stage top-k over the cuBLAS score matrix (chunk then merge).
    chunk_topk_func: CudaFunction,
    merge_topk_func: CudaFunction,
    // Resident rows pre-multiplied by the rotation (rows @ rotation), so queries
    // are scored raw (no per-query rotation). See `build_inner`.
    rows_dev: CudaSlice<f32>,
    stable_ids: Vec<u64>,
    // Reverse of `stable_ids` (id -> slot), so an in-place patch resolves the
    // resident slot of each changed id in O(1). Stays valid across in-place
    // updates (which keep ids at their slots) and is rebuilt with the session.
    slot_by_id: HashMap<u64, usize>,
    dim: usize,
    n: usize,
    // Device scratch reused across calls, keyed by (nq, k). Interior mutability so
    // the hot query path stays `&self`; the engine is thread-confined.
    scratch: RefCell<HashMap<(usize, usize), GpuScratch>>,
}

impl GpuScanSession {
    /// Builds a resident session from reconstructed rotated-calibrated rows.
    ///
    /// `rows` is the flat `[n * dim]` row buffer in slot order (as returned by
    /// `IdMapIndex::reconstruct_all`), `stable_ids` the `[n]` slot-to-id map, and
    /// `rotation` the flat `[dim * dim]` row-major rotation matrix. Returns
    /// [`GpuScanError::Unavailable`] when no driver loads (caller uses the CPU
    /// kernel and should not retry) and [`GpuScanError::Device`] on any CUDA call
    /// failure.
    pub fn build(
        rows: &[f32],
        stable_ids: &[u64],
        rotation: &[f32],
        dim: usize,
    ) -> Result<Self, GpuScanError> {
        if dim == 0 {
            return Err(GpuScanError::Invalid("dim must be positive".into()));
        }
        let n = stable_ids.len();
        if rows.len() != n * dim {
            return Err(GpuScanError::Invalid(format!(
                "rows length {} does not match n*dim {}",
                rows.len(),
                n * dim
            )));
        }
        if rotation.len() != dim * dim {
            return Err(GpuScanError::Invalid(format!(
                "rotation length {} does not match dim*dim {}",
                rotation.len(),
                dim * dim
            )));
        }
        if n == 0 {
            return Err(GpuScanError::Invalid("cannot build a session over 0 rows".into()));
        }
        if !cuda_runtime_available() {
            return Err(GpuScanError::Unavailable("no CUDA driver library".into()));
        }
        // cudarc loads libcublas/libnvrtc lazily on first use and PANICS (rather
        // than returning an error) when a library is absent or on a path its
        // `dlopen` does not search. A GPU host with an unusual CUDA layout must
        // fall back to the CPU kernel, never crash the process, so the CUDA build
        // runs under `catch_unwind` and a panic becomes an `Unavailable` fallback.
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Self::build_inner(rows, stable_ids, rotation, dim, n)
        })) {
            Ok(result) => result,
            Err(_) => Err(GpuScanError::Unavailable(
                "CUDA library load panicked (missing libcublas/libnvrtc); using CPU kernel".into(),
            )),
        }
    }

    /// The CUDA-touching half of [`Self::build`], isolated so a cudarc panic from a
    /// missing/unfindable library is contained by the caller's `catch_unwind`.
    fn build_inner(
        rows: &[f32],
        stable_ids: &[u64],
        rotation: &[f32],
        dim: usize,
        n: usize,
    ) -> Result<Self, GpuScanError> {
        let ctx = CudaContext::new(0).map_err(|err| GpuScanError::Unavailable(err.to_string()))?;
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone())
            .map_err(|err| GpuScanError::Device(err.to_string()))?;

        // Bake the rotation into the resident rows ONCE at build, so per-query the
        // queries go in raw. Since score(q, row) = <q @ rotation^T, row> reassociates
        // to <q, row @ rotation>, storing `rows @ rotation` makes the per-batch host
        // rotation (the dominant cost: a naive O(nq * dim^2) loop) disappear, and the
        // scores GEMM then takes raw queries unchanged. Done on the GPU as a single
        // GEMM, so the build stays fast.
        let recon_dev = stream
            .clone_htod(rows)
            .map_err(|err| GpuScanError::Device(err.to_string()))?;
        let rotation_dev = stream
            .clone_htod(rotation)
            .map_err(|err| GpuScanError::Device(err.to_string()))?;
        let mut rows_dev = stream
            .alloc_zeros::<f32>(n * dim)
            .map_err(|err| GpuScanError::Device(err.to_string()))?;
        // Row-major rowrot[r][d] = sum_d' rows[r][d'] * rotation[d'][d] = (rows @ rotation).
        // Computed column-major as rowrot^T (dim x n): a = rotation (OP_N), b = rows (OP_N).
        let bake = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: dim as i32,
            n: n as i32,
            k: dim as i32,
            alpha: 1.0f32,
            lda: dim as i32,
            ldb: dim as i32,
            beta: 0.0f32,
            ldc: dim as i32,
        };
        // SAFETY: recon_dev is n*dim, rotation_dev is dim*dim, rows_dev is n*dim; the
        // GEMM is synchronized below before the transient inputs are dropped.
        unsafe { blas.gemm(bake, &rotation_dev, &recon_dev, &mut rows_dev) }
            .map_err(|err| GpuScanError::Device(err.to_string()))?;
        stream
            .synchronize()
            .map_err(|err| GpuScanError::Device(err.to_string()))?;
        drop(recon_dev);
        drop(rotation_dev);

        let ptx = compile_ptx(KERNELS)
            .map_err(|err| GpuScanError::Device(format!("nvrtc compile failed: {err}")))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|err| GpuScanError::Device(err.to_string()))?;
        let func = module
            .load_function("topk_argmax")
            .map_err(|err| GpuScanError::Device(err.to_string()))?;
        let scatter_func = module
            .load_function("scatter_rows")
            .map_err(|err| GpuScanError::Device(err.to_string()))?;
        let chunk_topk_func = module
            .load_function("chunk_topk")
            .map_err(|err| GpuScanError::Device(err.to_string()))?;
        let merge_topk_func = module
            .load_function("merge_topk")
            .map_err(|err| GpuScanError::Device(err.to_string()))?;
        // Reverse map for O(1) slot resolution during a patch. Slot order is the
        // index's slot order, which is unique by construction; guard anyway.
        let mut slot_by_id = HashMap::with_capacity(n);
        for (slot, &id) in stable_ids.iter().enumerate() {
            if slot_by_id.insert(id, slot).is_some() {
                return Err(GpuScanError::Invalid(format!(
                    "duplicate stable id {id} among session rows"
                )));
            }
        }
        if debug_enabled() {
            eprintln!("lodedb_gpu: resident GPU session built (rows={n}, dim={dim})");
        }
        Ok(Self {
            stream,
            blas,
            _module: module,
            func,
            scatter_func,
            chunk_topk_func,
            merge_topk_func,
            rows_dev,
            stable_ids: stable_ids.to_vec(),
            slot_by_id,
            dim,
            n,
            scratch: RefCell::new(HashMap::new()),
        })
    }

    /// Number of resident rows.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the session holds no rows (never true for a built session).
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Patches the resident rows for a set of already-present ids, in place.
    ///
    /// `ids` are stable ids that must already be resident (an upsert that only
    /// overwrote existing rows leaves their slots unchanged); `rows` is the flat
    /// `[ids.len() * dim]` buffer of their freshly reconstructed calibrated rows (as
    /// from `IdMapIndex::reconstruct_rows`), and `rotation` the `[dim * dim]`
    /// row-major rotation. Only the changed rows are re-baked and written, so the
    /// cost is `O(changed)` rather than a full rebuild. Returns
    /// [`GpuScanError::Invalid`] (the caller rebuilds the session) when an id is not
    /// resident, ids resolve to duplicate slots, or shapes disagree, and
    /// [`GpuScanError::Device`] on a CUDA failure.
    pub fn patch(
        &mut self,
        ids: &[u64],
        rows: &[f32],
        rotation: &[f32],
    ) -> Result<(), GpuScanError> {
        // A built session has already loaded its CUDA libraries, so a panic is not
        // expected, but a patch must never crash the host; a panic fails the patch
        // and the caller rebuilds the session from scratch.
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.patch_inner(ids, rows, rotation)
        })) {
            Ok(result) => result,
            Err(_) => Err(GpuScanError::Device("CUDA patch panicked; rebuild session".into())),
        }
    }

    fn patch_inner(
        &mut self,
        ids: &[u64],
        rows: &[f32],
        rotation: &[f32],
    ) -> Result<(), GpuScanError> {
        let c = ids.len();
        if c == 0 {
            return Err(GpuScanError::Invalid("patch needs at least one row".into()));
        }
        let elems = c
            .checked_mul(self.dim)
            .ok_or_else(|| GpuScanError::Invalid("patch row count overflow".into()))?;
        if rows.len() != elems {
            return Err(GpuScanError::Invalid(format!(
                "rows length {} does not match ids*dim {}",
                rows.len(),
                elems
            )));
        }
        let rot_elems = self
            .dim
            .checked_mul(self.dim)
            .ok_or_else(|| GpuScanError::Invalid("rotation size overflow".into()))?;
        if rotation.len() != rot_elems {
            return Err(GpuScanError::Invalid(format!(
                "rotation length {} does not match dim*dim {}",
                rotation.len(),
                rot_elems
            )));
        }
        // Resolve each id to its resident slot, rejecting unknown ids and duplicate
        // slots (duplicate destinations would race in the scatter kernel).
        let mut slots = Vec::with_capacity(c);
        let mut seen = HashSet::with_capacity(c);
        for &id in ids {
            let slot = *self
                .slot_by_id
                .get(&id)
                .ok_or_else(|| GpuScanError::Invalid(format!("patch id {id} is not resident")))?;
            if !seen.insert(slot) {
                return Err(GpuScanError::Invalid(
                    "patch ids resolve to duplicate slots".into(),
                ));
            }
            slots.push(
                u32::try_from(slot)
                    .map_err(|_| GpuScanError::Invalid("resident slot exceeds u32".into()))?,
            );
        }
        let c_i32 =
            i32::try_from(c).map_err(|_| GpuScanError::Invalid("patch rows exceed i32".into()))?;
        let dim_i32 = i32::try_from(self.dim)
            .map_err(|_| GpuScanError::Invalid("dim exceeds i32".into()))?;
        let total = u32::try_from(elems)
            .map_err(|_| GpuScanError::Invalid("patch elements exceed u32".into()))?;

        // Upload the changed rows + rotation, then bake `rows @ rotation` into a
        // contiguous temp with the SAME column-major GEMM as `build_inner` (n = c),
        // so `baked[i*dim + d]` matches the resident `rows_dev` row-major layout.
        let raw_dev = self
            .stream
            .clone_htod(rows)
            .map_err(|err| GpuScanError::Device(err.to_string()))?;
        let rotation_dev = self
            .stream
            .clone_htod(rotation)
            .map_err(|err| GpuScanError::Device(err.to_string()))?;
        let mut baked_dev = self
            .stream
            .alloc_zeros::<f32>(elems)
            .map_err(|err| GpuScanError::Device(err.to_string()))?;
        let bake = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: dim_i32,
            n: c_i32,
            k: dim_i32,
            alpha: 1.0f32,
            lda: dim_i32,
            ldb: dim_i32,
            beta: 0.0f32,
            ldc: dim_i32,
        };
        // SAFETY: rotation_dev is dim*dim, raw_dev is c*dim, baked_dev is c*dim; the
        // GEMM result is consumed by the scatter launch + synchronize below before
        // the transient inputs drop.
        unsafe { self.blas.gemm(bake, &rotation_dev, &raw_dev, &mut baked_dev) }
            .map_err(|err| GpuScanError::Device(err.to_string()))?;

        let slots_dev = self
            .stream
            .clone_htod(slots.as_slice())
            .map_err(|err| GpuScanError::Device(err.to_string()))?;
        // ceil(total / SCATTER_BLOCK) without `div_ceil` (MSRV 1.70) or overflow.
        let grid = (total / SCATTER_BLOCK) + u32::from(total % SCATTER_BLOCK != 0);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (SCATTER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = self.stream.launch_builder(&self.scatter_func);
        builder
            .arg(&baked_dev)
            .arg(&slots_dev)
            .arg(&mut self.rows_dev)
            .arg(&c_i32)
            .arg(&dim_i32);
        // SAFETY: the kernel writes only `dst[slots[i]*dim + d]` for i in 0..c and
        // d in 0..dim; every slot is < self.n (resolved via slot_by_id) and unique,
        // so all writes land in distinct, in-bounds elements of rows_dev (n*dim);
        // baked_dev is c*dim and slots_dev is c, matching the bound c/dim scalars.
        unsafe { builder.launch(cfg) }
            .map_err(|err| GpuScanError::Device(err.to_string()))?;
        self.stream
            .synchronize()
            .map_err(|err| GpuScanError::Device(err.to_string()))?;
        if debug_enabled() {
            eprintln!("lodedb_gpu: patched {c} resident row(s) in place");
        }
        Ok(())
    }

    /// Scores a query batch and returns flat `[nq * k]` `(scores, stable_ids)`.
    ///
    /// `queries` is the flat `[nq * dim]` unrotated query buffer. The result rows
    /// are ordered descending by score, ascending by stable id on exact ties, so
    /// the layout matches `IdMapIndex::search` and the caller's row assembly is
    /// reused unchanged. `k` must be `<= len()`.
    pub fn search(
        &self,
        queries: &[f32],
        nq: usize,
        k: usize,
    ) -> Result<(Vec<f32>, Vec<u64>), GpuScanError> {
        self.search_variant(queries, nq, k, fused_topk_enabled())
    }

    /// [`Self::search`] with the top-k backend chosen explicitly rather than from
    /// the env flag. Both backends read the same cuBLAS scores and select the same
    /// (score desc, slot asc) top-k, so they return identical results; the parameter
    /// exists so a parity test can exercise both on one built session.
    fn search_variant(
        &self,
        queries: &[f32],
        nq: usize,
        k: usize,
        fused: bool,
    ) -> Result<(Vec<f32>, Vec<u64>), GpuScanError> {
        // Defense in depth: a built session has already loaded its CUDA libraries,
        // so a panic here is not expected, but the scan must never crash the host
        // process; a panic falls back to the CPU kernel like any device error.
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.search_inner(queries, nq, k, fused)
        })) {
            Ok(result) => result,
            Err(_) => Err(GpuScanError::Device("CUDA scan panicked; using CPU kernel".into())),
        }
    }

    fn search_inner(
        &self,
        queries: &[f32],
        nq: usize,
        k: usize,
        fused: bool,
    ) -> Result<(Vec<f32>, Vec<u64>), GpuScanError> {
        if nq == 0 || k == 0 {
            return Err(GpuScanError::Invalid("nq and k must be positive".into()));
        }
        if k > self.n {
            return Err(GpuScanError::Invalid(format!(
                "k {} exceeds resident rows {}",
                k, self.n
            )));
        }
        if queries.len() != nq * self.dim {
            return Err(GpuScanError::Invalid(format!(
                "queries length {} does not match nq*dim {}",
                queries.len(),
                nq * self.dim
            )));
        }

        // The fused stage-1 grid maps nq onto gridDim.y (capped at 65,535); a larger
        // batch falls back to the default path rather than failing the launch. This
        // effective flag drives the scratch allocation and the launch below alike.
        let fused = fused && nq <= FUSED_MAX_NQ;

        // Queries are scored raw: the rotation is baked into the resident rows at
        // build (rows @ rotation), so there is no per-query host rotation.

        // Reuse device scratch for this (nq, k) shape; allocate only on first sight.
        // The serving layer drives a small set of shapes, so steady-state queries
        // do no device allocation or zeroing — only copies and kernels.
        // ceil(n / FUSED_CHUNK_ROWS) without div_ceil (MSRV 1.70); only used when fused.
        let num_chunks = self.n / FUSED_CHUNK_ROWS + usize::from(self.n % FUSED_CHUNK_ROWS != 0);

        let mut cache = self.scratch.borrow_mut();
        if let std::collections::hash_map::Entry::Vacant(entry) = cache.entry((nq, k)) {
            entry.insert(GpuScratch {
                q_dev: self
                    .stream
                    .alloc_zeros::<f32>(nq * self.dim)
                    .map_err(|err| GpuScanError::Device(err.to_string()))?,
                scores_dev: self
                    .stream
                    .alloc_zeros::<f32>(nq * self.n)
                    .map_err(|err| GpuScanError::Device(err.to_string()))?,
                idx_dev: self
                    .stream
                    .alloc_zeros::<u32>(nq * k)
                    .map_err(|err| GpuScanError::Device(err.to_string()))?,
                val_dev: self
                    .stream
                    .alloc_zeros::<f32>(nq * k)
                    .map_err(|err| GpuScanError::Device(err.to_string()))?,
                part_idx: None,
                part_val: None,
            });
        }
        // Allocate the stage-1 partial buffers the first time the fused path runs for
        // this shape (nq * num_chunks * k). Done before the destructure so the shared
        // `&mut` borrow stays simple.
        if fused {
            let entry = cache.get_mut(&(nq, k)).expect("scratch was just inserted");
            if entry.part_idx.is_none() {
                let part_len = nq
                    .checked_mul(num_chunks)
                    .and_then(|value| value.checked_mul(k))
                    .ok_or_else(|| GpuScanError::Invalid("fused partial count overflow".into()))?;
                entry.part_idx = Some(
                    self.stream
                        .alloc_zeros::<u32>(part_len)
                        .map_err(|err| GpuScanError::Device(err.to_string()))?,
                );
                entry.part_val = Some(
                    self.stream
                        .alloc_zeros::<f32>(part_len)
                        .map_err(|err| GpuScanError::Device(err.to_string()))?,
                );
            }
        }
        let GpuScratch {
            q_dev,
            scores_dev,
            idx_dev,
            val_dev,
            part_idx,
            part_val,
        } = cache.get_mut(&(nq, k)).expect("scratch was just inserted");

        // Upload the raw queries into the cached query buffer (in place).
        self.stream
            .memcpy_htod(queries, q_dev)
            .map_err(|err| GpuScanError::Device(err.to_string()))?;

        // Row-major scores[query*n + row] = q_rot(nq x dim) @ rows(n x dim)^T,
        // computed column-major as scores^T (n x nq): a = rows (OP_T), b = q_rot.
        // beta=0 overwrites every element of scores_dev, so it needs no zeroing.
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: self.n as i32,
            n: nq as i32,
            k: self.dim as i32,
            alpha: 1.0f32,
            lda: self.dim as i32,
            ldb: self.dim as i32,
            beta: 0.0f32,
            ldc: self.n as i32,
        };
        // SAFETY: shapes/leading dims are validated above; all buffers live on
        // `self.stream` and are sized nq*n / nq*dim for this (nq, k) shape.
        unsafe { self.blas.gemm(cfg, &self.rows_dev, &*q_dev, &mut *scores_dev) }
            .map_err(|err| GpuScanError::Device(err.to_string()))?;

        let n_i32 = self.n as i32;
        let k_i32 = k as i32;
        let reduce_bytes =
            TOPK_BLOCK * (std::mem::size_of::<f32>() + std::mem::size_of::<u32>()) as u32;

        if fused {
            // Two-stage top-k over the cuBLAS score matrix: stage 1 reads each score
            // once into shared memory and emits per-chunk top-k partials; stage 2
            // merges them. Same scores, same (score desc, slot asc) order, so the
            // result equals the default topk_argmax path.
            let chunk_rows_i32 = FUSED_CHUNK_ROWS as i32;
            let num_chunks_i32 = num_chunks as i32;
            let part_idx = part_idx.as_mut().expect("fused partials allocated above");
            let part_val = part_val.as_mut().expect("fused partials allocated above");

            let stage1 = LaunchConfig {
                grid_dim: (num_chunks as u32, nq as u32, 1),
                block_dim: (TOPK_BLOCK, 1, 1),
                shared_mem_bytes: FUSED_CHUNK_ROWS as u32 * std::mem::size_of::<f32>() as u32
                    + reduce_bytes,
            };
            let mut builder = self.stream.launch_builder(&self.chunk_topk_func);
            builder
                .arg(&*scores_dev)
                .arg(&n_i32)
                .arg(&k_i32)
                .arg(&chunk_rows_i32)
                .arg(&num_chunks_i32)
                .arg(&mut *part_idx)
                .arg(&mut *part_val);
            // SAFETY: grid is (num_chunks, nq); each block reads its query's chunk of
            // `scores_dev` (nq*n) and writes k partials into `part_*` (nq*num_chunks*k);
            // dynamic shared holds chunk_rows f32 plus the block's reduction scratch.
            unsafe { builder.launch(stage1) }
                .map_err(|err| GpuScanError::Device(err.to_string()))?;

            let stage2 = LaunchConfig {
                grid_dim: (nq as u32, 1, 1),
                block_dim: (TOPK_BLOCK, 1, 1),
                shared_mem_bytes: reduce_bytes,
            };
            let mut builder = self.stream.launch_builder(&self.merge_topk_func);
            builder
                .arg(&mut *part_idx)
                .arg(&mut *part_val)
                .arg(&k_i32)
                .arg(&num_chunks_i32)
                .arg(&mut *idx_dev)
                .arg(&mut *val_dev);
            // SAFETY: grid is nq blocks; each merges its num_chunks*k partials
            // (`part_*` is nq*num_chunks*k) into `idx_dev`/`val_dev` (nq*k); dynamic
            // shared matches the block's reduction scratch.
            unsafe { builder.launch(stage2) }
                .map_err(|err| GpuScanError::Device(err.to_string()))?;
        } else {
            let cfg_launch = LaunchConfig {
                grid_dim: (nq as u32, 1, 1),
                block_dim: (TOPK_BLOCK, 1, 1),
                shared_mem_bytes: reduce_bytes,
            };
            let mut builder = self.stream.launch_builder(&self.func);
            builder
                .arg(&mut *scores_dev)
                .arg(&n_i32)
                .arg(&k_i32)
                .arg(&mut *idx_dev)
                .arg(&mut *val_dev);
            // SAFETY: the kernel reads/writes only the bound buffers within bounds
            // (`scores_dev` is nq*n, `idx_dev`/`val_dev` are nq*k, grid is nq blocks);
            // shared memory matches `block_dim`'s two per-thread scratch arrays.
            unsafe { builder.launch(cfg_launch) }
                .map_err(|err| GpuScanError::Device(err.to_string()))?;
        }

        if debug_enabled() {
            static LOGGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
            LOGGED.get_or_init(|| {
                let path = if fused { "fused" } else { "cublas" };
                eprintln!(
                    "lodedb_gpu: GPU scan engaged ({path}, nq={nq}, k={k}, rows={})",
                    self.n
                );
            });
        }

        // clone_dtoh issues a synchronous copy, which waits on the GEMM and kernel
        // queued above, so no separate stream synchronize is needed.
        let host_idx = self
            .stream
            .clone_dtoh(&*idx_dev)
            .map_err(|err| GpuScanError::Device(err.to_string()))?;
        let host_val = self
            .stream
            .clone_dtoh(&*val_dev)
            .map_err(|err| GpuScanError::Device(err.to_string()))?;

        Ok(self.finalize(&host_idx, &host_val, nq, k))
    }

    /// Maps device slots to stable ids and fixes the deterministic ordering
    /// (descending score, ascending stable id on exact ties) per query.
    fn finalize(
        &self,
        host_idx: &[u32],
        host_val: &[f32],
        nq: usize,
        k: usize,
    ) -> (Vec<f32>, Vec<u64>) {
        let mut scores = vec![0.0f32; nq * k];
        let mut ids = vec![0u64; nq * k];
        let mut row: Vec<(f32, u64)> = Vec::with_capacity(k);
        for q in 0..nq {
            row.clear();
            for j in 0..k {
                let slot = host_idx[q * k + j] as usize;
                let stable_id = self.stable_ids.get(slot).copied().unwrap_or(0);
                row.push((host_val[q * k + j], stable_id));
            }
            row.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            for (j, (score, id)) in row.iter().enumerate() {
                scores[q * k + j] = *score;
                ids[q * k + j] = *id;
            }
        }
        (scores, ids)
    }
}

/// Rotates a query batch on the host: `q_rot[b][out] = sum_in q[b][in] *
/// rotation[out*dim + in]` (row-major rotation), i.e. `q @ rotation^T`.
///
/// Only the parity test uses this now: the production path bakes the rotation into
/// the resident rows at build (see `build_inner`) so queries are scored raw.
#[cfg(test)]
fn rotate_queries(queries: &[f32], nq: usize, dim: usize, rotation: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; nq * dim];
    for b in 0..nq {
        let query = &queries[b * dim..(b + 1) * dim];
        let target = &mut out[b * dim..(b + 1) * dim];
        for (out_d, slot) in target.iter_mut().enumerate() {
            let rotation_row = &rotation[out_d * dim..(out_d + 1) * dim];
            let mut acc = 0.0f32;
            for (q_value, r_value) in query.iter().zip(rotation_row) {
                acc += q_value * r_value;
            }
            *slot = acc;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotate_queries_matches_manual_inner_product() {
        // 2x2 rotation, two queries; verify q @ rotation^T against hand math.
        let rotation = vec![1.0, 2.0, 3.0, 4.0]; // [[1,2],[3,4]] row-major
        let queries = vec![1.0, 0.0, 0.0, 1.0]; // e0, e1
        let rotated = rotate_queries(&queries, 2, 2, &rotation);
        // q=e0: out0 = 1*1+0*2 = 1; out1 = 1*3+0*4 = 3
        // q=e1: out0 = 0*1+1*2 = 2; out1 = 0*3+1*4 = 4
        assert_eq!(rotated, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn cuda_probe_is_stable() {
        // Must not panic on a CPU-only host; just exercises the cached probe.
        let _ = cuda_runtime_available();
    }

    /// Deterministic pseudo-random f32 in [-1, 1) via an LCG, so the parity test
    /// needs no `rand` dependency and is reproducible on any host.
    fn pseudo_fill(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bits = (state >> 33) as u32;
                (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn gpu_topk_matches_cpu_reference() {
        // Runs only where a CUDA driver is present; a no-op elsewhere (CI/macOS).
        if !cuda_runtime_available() {
            eprintln!("skipping gpu_topk_matches_cpu_reference: no CUDA runtime");
            return;
        }
        let (n, dim, nq, k) = (300usize, 64usize, 8usize, 10usize);
        let rows = pseudo_fill(n * dim, 0x1234_5678);
        let rotation = pseudo_fill(dim * dim, 0x9abc_def0);
        let queries = pseudo_fill(nq * dim, 0x55aa_55aa);
        let stable_ids: Vec<u64> = (0..n as u64).map(|i| i.wrapping_mul(2_654_435_761) | 1).collect();

        let session = GpuScanSession::build(&rows, &stable_ids, &rotation, dim)
            .expect("session build on a CUDA host");
        let (scores, ids) = session.search(&queries, nq, k).expect("gpu search");
        assert_eq!(scores.len(), nq * k);
        assert_eq!(ids.len(), nq * k);

        let id_to_slot: std::collections::HashMap<u64, usize> = stable_ids
            .iter()
            .enumerate()
            .map(|(slot, &id)| (id, slot))
            .collect();

        for q in 0..nq {
            let rotated = rotate_queries(&queries[q * dim..(q + 1) * dim], 1, dim, &rotation);
            // CPU reference: the exact maximum score over all rows for this query.
            let mut best = f32::NEG_INFINITY;
            for slot in 0..n {
                let row = &rows[slot * dim..(slot + 1) * dim];
                let score: f32 = rotated.iter().zip(row).map(|(a, b)| a * b).sum();
                if score > best {
                    best = score;
                }
            }
            // The top-1 score must equal the true maximum (id may be ambiguous only
            // under an exact tie, which random data does not produce).
            let tol_top = 1e-2 * best.abs().max(1.0);
            assert!(
                (scores[q * k] - best).abs() <= tol_top,
                "query {q} top-1 score {} != cpu max {best}",
                scores[q * k]
            );
            // Each returned row is in descending order and carries the correct score.
            let mut prev = f32::INFINITY;
            for j in 0..k {
                let score = scores[q * k + j];
                assert!(score <= prev + 1e-3, "query {q} rank {j} not descending");
                prev = score;
                let slot = id_to_slot[&ids[q * k + j]];
                let row = &rows[slot * dim..(slot + 1) * dim];
                let cpu_score: f32 = rotated.iter().zip(row).map(|(a, b)| a * b).sum();
                let tol = 1e-2 * cpu_score.abs().max(1.0);
                assert!(
                    (score - cpu_score).abs() <= tol,
                    "query {q} rank {j} id {} score {score} != cpu {cpu_score}",
                    ids[q * k + j]
                );
            }
        }
    }

    #[test]
    fn gpu_patch_matches_rebuild() {
        // An in-place patch of a subset of rows must leave the session scoring
        // identically to a session freshly built over the same patched corpus.
        // Runs only where a CUDA driver is present; a no-op elsewhere (CI/macOS).
        if !cuda_runtime_available() {
            eprintln!("skipping gpu_patch_matches_rebuild: no CUDA runtime");
            return;
        }
        let (n, dim, nq, k) = (256usize, 64usize, 6usize, 8usize);
        let rows = pseudo_fill(n * dim, 0x0f0f_1111);
        let rotation = pseudo_fill(dim * dim, 0x2222_3333);
        let queries = pseudo_fill(nq * dim, 0x4444_5555);
        let stable_ids: Vec<u64> =
            (0..n as u64).map(|i| i.wrapping_mul(2_654_435_761) | 1).collect();

        let mut session = GpuScanSession::build(&rows, &stable_ids, &rotation, dim)
            .expect("session build on a CUDA host");

        // Fresh content for a scattered subset of slots (boundaries included).
        let changed_slots = [0usize, 7, 63, 128, 200, 255];
        let changed_ids: Vec<u64> = changed_slots.iter().map(|&slot| stable_ids[slot]).collect();
        let new_rows = pseudo_fill(changed_slots.len() * dim, 0x6666_7777);

        // Reference: the whole corpus with the same edits applied, built from scratch.
        let mut patched_rows = rows.clone();
        for (i, &slot) in changed_slots.iter().enumerate() {
            patched_rows[slot * dim..(slot + 1) * dim]
                .copy_from_slice(&new_rows[i * dim..(i + 1) * dim]);
        }
        let rebuilt = GpuScanSession::build(&patched_rows, &stable_ids, &rotation, dim)
            .expect("rebuild on a CUDA host");

        // Patch the live session in place; it must then agree with the rebuild.
        session
            .patch(&changed_ids, &new_rows, &rotation)
            .expect("in-place patch on a CUDA host");

        let (patched_scores, patched_ids) =
            session.search(&queries, nq, k).expect("patched search");
        let (rebuilt_scores, rebuilt_ids) =
            rebuilt.search(&queries, nq, k).expect("rebuilt search");
        assert_eq!(patched_ids, rebuilt_ids, "patched ranking != rebuilt ranking");
        for (patched, rebuilt) in patched_scores.iter().zip(&rebuilt_scores) {
            let tol = 1e-3 * rebuilt.abs().max(1.0);
            assert!(
                (patched - rebuilt).abs() <= tol,
                "patched score {patched} != rebuilt score {rebuilt}"
            );
        }
    }

    /// Device top-k order: higher score first, ties broken by lower slot, using the
    /// same float `>`/`<`/`==` comparisons as the device `fused_gt`.
    fn kernel_order(a: &(f32, u32), b: &(f32, u32)) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        if a.0 > b.0 {
            Ordering::Less
        } else if a.0 < b.0 {
            Ordering::Greater
        } else {
            a.1.cmp(&b.1)
        }
    }

    /// Brute-force global top-k: the reference the fused decomposition must match.
    fn brute_topk_ref(scores: &[f32], k: usize) -> Vec<(f32, u32)> {
        let mut all: Vec<(f32, u32)> =
            scores.iter().enumerate().map(|(i, &s)| (s, i as u32)).collect();
        all.sort_by(kernel_order);
        all.truncate(k);
        all
    }

    /// Host model of the two device stages: stage 1 selects each chunk's top-k, stage
    /// 2 merges the partials. Exercises the chunking, merge, and tie-break logic the
    /// `chunk_topk`/`merge_topk` kernels implement, with no GPU.
    fn chunked_topk_ref(scores: &[f32], k: usize, chunk_rows: usize) -> Vec<(f32, u32)> {
        let n = scores.len();
        let num_chunks = n / chunk_rows + usize::from(n % chunk_rows != 0);
        let mut partials: Vec<(f32, u32)> = Vec::new();
        for c in 0..num_chunks {
            let row0 = c * chunk_rows;
            let end = (row0 + chunk_rows).min(n);
            let mut chunk: Vec<(f32, u32)> = (row0..end).map(|r| (scores[r], r as u32)).collect();
            chunk.sort_by(kernel_order);
            chunk.truncate(k); // a chunk with < k rows contributes fewer (implicit INVALID pad)
            partials.extend(chunk);
        }
        partials.sort_by(kernel_order);
        partials.truncate(k);
        partials
    }

    #[test]
    fn fused_chunked_topk_matches_bruteforce() {
        // The fused path's correctness rests on two claims: the union of per-chunk
        // top-ks contains the global top-k, and both stages apply one strict
        // (score desc, slot asc) order. Fuzz the host model of that decomposition
        // against brute force across sizes, k, chunk sizes, and heavy ties (small
        // integer scores) so the algorithm is validated without a GPU. k <= n and
        // chunk_rows may be smaller or larger than k, matching the kernel's inputs.
        let mut seed = 0x00C0_FFEE_u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        for case in 0..4000 {
            let n = 1 + (next() as usize % 300);
            let k = 1 + (next() as usize % n.min(40));
            let chunk_rows = 1 + (next() as usize % 40);
            // Small integer scores force frequent exact ties, exercising slot tie-break.
            let modulo = 1 + (next() % 6);
            let scores: Vec<f32> = (0..n).map(|_| (next() % modulo) as f32).collect();
            let brute = brute_topk_ref(&scores, k);
            let chunked = chunked_topk_ref(&scores, k, chunk_rows);
            assert_eq!(
                chunked, brute,
                "case {case}: n={n} k={k} chunk_rows={chunk_rows} disagreed with brute force"
            );
        }
    }

    #[test]
    fn gpu_fused_topk_matches_two_pass() {
        // The opt-in fused two-stage top-k must return exactly what the default
        // cuBLAS + topk_argmax path returns: both read the same cuBLAS scores and
        // select the same total-order top-k. Runs only where a CUDA driver is
        // present; a no-op elsewhere (CI/macOS).
        if !cuda_runtime_available() {
            eprintln!("skipping gpu_fused_topk_matches_two_pass: no CUDA runtime");
            return;
        }
        // Corpus spans several fused chunks with a ragged last chunk (n not a
        // multiple of FUSED_CHUNK_ROWS) and a large k, so stage 1 hits chunk
        // exhaustion and stage 2 does real merging.
        let (dim, nq, k) = (96usize, 12usize, 100usize);
        let n = FUSED_CHUNK_ROWS * 2 + 777;
        let rows = pseudo_fill(n * dim, 0x1357_9bdf);
        let rotation = pseudo_fill(dim * dim, 0x2468_ace0);
        let queries = pseudo_fill(nq * dim, 0x0bad_f00d);
        let stable_ids: Vec<u64> =
            (0..n as u64).map(|i| i.wrapping_mul(2_654_435_761) | 1).collect();

        let session = GpuScanSession::build(&rows, &stable_ids, &rotation, dim)
            .expect("session build on a CUDA host");

        let (base_scores, base_ids) = session
            .search_variant(&queries, nq, k, false)
            .expect("cublas + topk_argmax path");
        let (fused_scores, fused_ids) = session
            .search_variant(&queries, nq, k, true)
            .expect("fused two-stage path");

        // Same cuBLAS scores + same total order => identical selection and values.
        assert_eq!(fused_ids, base_ids, "fused ids differ from the default path");
        assert_eq!(
            fused_scores, base_scores,
            "fused scores differ from the default path"
        );
    }

    #[test]
    fn gpu_fused_topk_timing() {
        // Perf comparison of the default cuBLAS + topk_argmax path against the fused
        // two-stage top-k, across batch sizes. Opt-in (needs a large corpus and many
        // iterations): runs only with a CUDA driver AND LODEDB_GPU_BENCH set, so it
        // is a no-op on CI/macOS and in ordinary test runs. Corpus size overridable
        // via LODEDB_GPU_BENCH_N. This is where the fused path's win (or regression)
        // is measured on real hardware; it asserts nothing.
        use std::time::Instant;
        if !cuda_runtime_available() || std::env::var_os("LODEDB_GPU_BENCH").is_none() {
            eprintln!("skipping gpu_fused_topk_timing: set LODEDB_GPU_BENCH=1 on a CUDA host");
            return;
        }
        let dim = 128usize;
        let k = 100usize;
        let n = std::env::var("LODEDB_GPU_BENCH_N")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(200_000);
        let rows = pseudo_fill(n * dim, 0x5eed_1234);
        let rotation = pseudo_fill(dim * dim, 0x5eed_5678);
        let stable_ids: Vec<u64> =
            (0..n as u64).map(|i| i.wrapping_mul(2_654_435_761) | 1).collect();
        let session = GpuScanSession::build(&rows, &stable_ids, &rotation, dim)
            .expect("session build on a CUDA host");

        eprintln!("gpu_fused_topk_timing: n={n} dim={dim} k={k}");
        let iters = 20;
        for &nq in &[4usize, 16, 64, 256, 1024] {
            let queries = pseudo_fill(nq * dim, 0xa11ce ^ nq as u64);
            // Warm both paths (allocates scratch, primes caches) before timing.
            let _ = session.search_variant(&queries, nq, k, false).expect("warm default");
            let _ = session.search_variant(&queries, nq, k, true).expect("warm fused");

            let t0 = Instant::now();
            for _ in 0..iters {
                session.search_variant(&queries, nq, k, false).expect("default");
            }
            let base = t0.elapsed().as_secs_f64() / iters as f64;

            let t1 = Instant::now();
            for _ in 0..iters {
                session.search_variant(&queries, nq, k, true).expect("fused");
            }
            let fused = t1.elapsed().as_secs_f64() / iters as f64;

            eprintln!(
                "  nq={nq:>4}: default={:>8.3} ms  fused={:>8.3} ms  speedup={:.2}x",
                base * 1e3,
                fused * 1e3,
                base / fused
            );
        }
    }
}
