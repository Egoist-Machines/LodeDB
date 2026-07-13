//! Native Rust core for LodeDB.
//!
//! The Python engine remains authoritative during the migration. This crate
//! starts as a dependency-light home for deterministic core semantics that will
//! move from Python into Rust milestone by milestone.

pub mod engine;
pub mod error;
pub mod filter;
pub mod lexical;
pub mod storage;
pub mod text;
pub mod types;
pub mod vector;
pub mod version;

pub use error::{CoreError, CoreErrorCode};
pub use types::{
    CoreApiVersion, CoreDocument, CoreIndexConfig, CoreIndexCreateOptions, CoreMetadata,
    CoreMutationResult, CoreOpenOptions, CoreQuery, CoreRescoreOptions, CoreRoutePolicy, CoreSearchHit,
    CoreSearchResults, CoreSecurityOptions, CoreStats, CoreVectorDocument,
};
pub use vector::stable_id::{stable_uint64_for_text, stable_uint64_ids_for_chunk_ids};
pub use version::{CORE_VERSION, NATIVE_CORE_ABI_VERSION, STORAGE_SCHEMA_VERSION};

/// Returns whether a CUDA runtime is present for the native GPU-resident scan.
///
/// Re-exports `lodedb_gpu::cuda_runtime_available` so callers (including the
/// Python binding and `doctor`) can probe the real driver the cudarc scan in
/// `vector::turbovec` gates on, without depending on torch or CuPy.
pub fn cuda_runtime_available() -> bool {
    lodedb_gpu::cuda_runtime_available()
}
