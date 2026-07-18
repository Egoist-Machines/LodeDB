//! Verify a committed generation before trusting it.
//!
//! Two levels of assurance, both read-only:
//!
//! - [`verify_generation`] is store-agnostic (works on a remote/object store as
//!   well as a local directory): it re-hashes every artifact the committed root
//!   pins and compares against the recorded checksum, failing closed on the first
//!   mismatch. Reading the pointer through the engine's `read_commit_manifest`
//!   already validates the body checksum, so a corrupt root fails before any
//!   artifact is read.
//! - [`verify_local_generation_opens`] is the strongest check for a restored
//!   *local* copy: it opens the store read-only through `lodedb-core`'s own load
//!   path, proving the committed manifest and its artifacts actually parse and
//!   load as the engine would read them.

use crate::artifact_store::ArtifactStore;
use crate::digest::sha256_hex;
use crate::error::{ArtifactStoreError, Result};
use crate::generation_inventory::inventory_from_body;
use lodedb_core::storage::{load_store, LoadOptions};
use std::path::Path;

/// Metrics-only summary of a checksum verification.
///
/// Carries the generation and counts/bytes only (safe to log). A returned report
/// means every artifact matched its recorded checksum; a mismatch is an error, not
/// a report field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyReport {
    pub index_key: String,
    pub generation: u64,
    pub artifacts_verified: usize,
    pub bytes_verified: u64,
}

/// Re-hashes every artifact `index_key`'s committed generation pins and compares
/// against the manifest's recorded checksum.
///
/// Returns [`ArtifactStoreError::NotFound`] when the store holds no committed
/// generation for `index_key`, and [`ArtifactStoreError::Integrity`] on the first
/// artifact whose bytes do not match — failing closed rather than reporting a
/// partial success. Reading the pointer validates the body checksum as a
/// side effect (the engine's `read_commit_manifest` fails closed on a garbled
/// root), so this checks the whole chain: root body, then every referenced blob.
pub fn verify_generation(store: &dyn ArtifactStore, index_key: &str) -> Result<VerifyReport> {
    let body = store.read_pointer(index_key)?.ok_or_else(|| {
        ArtifactStoreError::NotFound(format!(
            "no committed generation to verify for index key {index_key:?}"
        ))
    })?;
    let inventory = inventory_from_body(index_key, Some(&body))?
        .expect("inventory is Some when the body is Some");

    let mut bytes_verified = 0u64;
    for artifact in &inventory.artifacts {
        let data = store.read_bytes(&artifact.name)?;
        let digest = sha256_hex(&data);
        if digest != artifact.sha256 {
            return Err(ArtifactStoreError::Integrity(format!(
                "artifact {:?} failed checksum: manifest records {}, computed {}",
                artifact.name, artifact.sha256, digest
            )));
        }
        bytes_verified += data.len() as u64;
    }

    Ok(VerifyReport {
        index_key: index_key.to_string(),
        generation: inventory.generation,
        artifacts_verified: inventory.artifacts.len(),
        bytes_verified,
    })
}

/// Metrics-only summary of a successful read-only open.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenReport {
    pub index_key: String,
    pub document_count: usize,
    pub chunk_count: usize,
}

/// Confirms a restored local generation opens read-only through the engine.
///
/// This reuses `lodedb-core`'s `load_store` with `read_only` (which takes no writer
/// lock and reads the exact committed manifest, never the `.wal` tail), so it
/// proves a restored directory is loadable exactly as the embedded engine would
/// read it — the acceptance check the roadmap calls for after a restore. Returns
/// the loaded document/chunk counts. `persistence_dir` is the local directory the
/// generation was restored into.
pub fn verify_local_generation_opens(
    persistence_dir: &Path,
    index_key: &str,
) -> Result<OpenReport> {
    let store = load_store(
        persistence_dir,
        index_key,
        LoadOptions {
            read_only: true,
            read_wal: false,
        },
    )?;
    Ok(OpenReport {
        index_key: index_key.to_string(),
        document_count: store.document_count(),
        chunk_count: store.chunk_count(),
    })
}
