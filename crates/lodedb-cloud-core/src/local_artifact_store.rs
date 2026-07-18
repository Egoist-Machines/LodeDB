//! Local-filesystem [`ArtifactStore`]: the default backend.
//!
//! A committed generation already lives on disk as immutable `g<epoch>.*`
//! artifacts under `<key>.gen/` pinned by `<key>.commit.json`, so the local store
//! is a thin wrapper over `lodedb-core`'s commit-manifest primitives — there is
//! no second format. Object-storage backends (S3/GCS/Azure) belong in a later
//! milestone, not here.

use crate::artifact_store::{body_generation, ArtifactStore};
use crate::digest::sha256_hex;
use crate::error::{ArtifactStoreError, Result};
use crate::paths::resolve_within;
use lodedb_core::storage::commit_manifest::{
    commit_manifest_path, read_commit_manifest, write_commit_manifest,
};
use serde_json::Value;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Stores artifacts as files under a directory; the pointer is `<key>.commit.json`.
pub struct LocalArtifactStore {
    root: PathBuf,
    fsync: bool,
}

impl LocalArtifactStore {
    /// Binds the store to a persistence directory (the same directory a `LodeDB`
    /// handle persists into).
    ///
    /// `fsync` mirrors the engine durability flag: when true, each artifact write
    /// and the pointer swap are fsynced (file + directory) so a pushed artifact
    /// survives power loss; the default (false) keeps the fast
    /// atomic-but-not-durable path.
    pub fn new(root: impl Into<PathBuf>, fsync: bool) -> Self {
        Self {
            root: root.into(),
            fsync,
        }
    }
}

/// Atomic publish: write to a sibling `.tmp`, fsync (gated), rename into place,
/// then fsync the parent directory so the rename is durable. This is the engine's
/// `durable_replace` discipline — never write a persisted artifact in place.
fn atomic_write(path: &Path, data: &[u8], fsync: bool) -> Result<()> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("artifact");
    let tmp = path.with_file_name(format!("{file_name}.tmp"));
    let mut handle = File::create(&tmp)?;
    handle.write_all(data)?;
    if fsync {
        handle.sync_all()?;
    }
    drop(handle);
    fs::rename(&tmp, path)?;
    if fsync {
        if let Some(parent) = path.parent() {
            File::open(parent)?.sync_all()?;
        }
    }
    Ok(())
}

impl ArtifactStore for LocalArtifactStore {
    fn read_bytes(&self, name: &str) -> Result<Vec<u8>> {
        let path = resolve_within(&self.root, &self.root.join(name))?;
        fs::read(&path).map_err(|error| match error.kind() {
            std::io::ErrorKind::NotFound => ArtifactStoreError::NotFound(name.to_string()),
            _ => ArtifactStoreError::Io(error),
        })
    }

    fn write_bytes_if_absent(&self, name: &str, data: &[u8], sha256: &str) -> Result<()> {
        // Verify the incoming bytes before any write, so corruption can never be
        // stored.
        let digest = sha256_hex(data);
        if digest != sha256 {
            return Err(ArtifactStoreError::Integrity(format!(
                "artifact {name:?} failed checksum: expected {sha256}, computed {digest}"
            )));
        }
        let path = resolve_within(&self.root, &self.root.join(name))?;
        if path.exists() {
            // Names are epoch-addressed, not sha-derived, so two independent
            // lineages can collide on a name. Identical bytes are an idempotent
            // no-op; different bytes are a genuine conflict we refuse rather than
            // clobber (the immutability invariant).
            let existing = sha256_hex(&fs::read(&path)?);
            if existing == sha256 {
                return Ok(());
            }
            return Err(ArtifactStoreError::Integrity(format!(
                "artifact {name:?} already exists with different content (stored {existing}, \
                 incoming {sha256}); refusing to overwrite an immutable artifact"
            )));
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        atomic_write(&path, data, self.fsync)
    }

    fn contains(&self, name: &str) -> Result<bool> {
        // A metadata probe, not a read — `try_exists` surfaces genuine I/O
        // failures (unlike `Path::exists`, which would mask them as absence).
        let path = resolve_within(&self.root, &self.root.join(name))?;
        Ok(path.try_exists()?)
    }

    fn read_pointer(&self, key: &str) -> Result<Option<Value>> {
        let pointer = resolve_within(&self.root, &commit_manifest_path(&self.root, key))?;
        // `read_commit_manifest` validates the schema version and body checksum,
        // failing closed on a garbled pointer.
        Ok(read_commit_manifest(&pointer)?.map(|manifest| manifest.body))
    }

    fn compare_and_swap_pointer(
        &self,
        key: &str,
        old_body: Option<&Value>,
        new_body: &Value,
    ) -> Result<()> {
        // On a local filesystem this is a read-check-then-replace: the final
        // rename inside `write_commit_manifest` is atomic per file, but the
        // read+swap pair is not a true cross-process CAS. That is safe under
        // LodeDB's single-writer model and for out-of-band backup use; an
        // object-store backend must instead use a real conditional write.
        //
        // The precondition compares the full committed body, not just its
        // generation number: two lineages can share a generation with different
        // content, so a numeric check would be ABA-prone.
        let pointer = resolve_within(&self.root, &commit_manifest_path(&self.root, key))?;
        let current = read_commit_manifest(&pointer)?.map(|manifest| manifest.body);
        if current.as_ref() != old_body {
            return Err(ArtifactStoreError::PointerConflict {
                key: key.to_string(),
                expected: old_body.and_then(body_generation),
                found: current.as_ref().and_then(body_generation),
            });
        }
        write_commit_manifest(&pointer, new_body, self.fsync)?;
        Ok(())
    }
}
