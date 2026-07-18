//! Object-storage [`ArtifactStore`]: the cloud backend (S3/GCS/Azure).
//!
//! Wraps any [`object_store::ObjectStore`] so a committed generation ships to
//! object storage under the same artifact names it has on disk. Two properties
//! the local filesystem gave for free must be re-established here:
//!
//! - **Atomic root-pointer commit.** Object stores have no `os.replace`, so the
//!   pointer swap uses the backend's *conditional write*: `PutMode::Create` (fails
//!   if the pointer exists) and `PutMode::Update{e_tag}` (fails unless the pointer
//!   still matches the version we read). That is a real strongly-consistent
//!   compare-and-swap, never an eventually-consistent list-then-write.
//! - **Per-tenant isolation.** Content addressing deduplicates by checksum, so in a
//!   shared bucket every name is prefixed with a per-tenant key. A caller can only
//!   reach blobs under its own prefix; a checksum known to one tenant cannot fetch
//!   another tenant's object.
//!
//! The [`ArtifactStore`] trait is synchronous, but `object_store` is async, so this
//! type owns a current-thread Tokio runtime and blocks on each operation. Its
//! consumers (transfer/verify) are sequential batch operations, and the managed
//! serving tier hydrates to a local directory before opening the engine, so a sync
//! surface fits every M2 caller.

use crate::artifact_store::{body_generation, ArtifactStore};
use crate::digest::sha256_hex;
use crate::error::{ArtifactStoreError, Result};
use lodedb_core::storage::commit_manifest::read_commit_manifest;
use object_store::path::Path as ObjectPath;
use object_store::{Error as ObjectError, ObjectStore, PutMode, PutOptions, UpdateVersion};
use serde_json::Value;
use std::io::Write;
use std::sync::Arc;
use tempfile::NamedTempFile;
use tokio::runtime::{Builder, Runtime};

/// Stores artifacts as objects under a per-tenant prefix; the pointer is
/// `<prefix>/<key>.commit.json`, swapped with a conditional write.
pub struct ObjectArtifactStore {
    store: Arc<dyn ObjectStore>,
    prefix: String,
    runtime: Runtime,
}

impl ObjectArtifactStore {
    /// Binds the store to `object_store` under `prefix` (the per-tenant namespace).
    ///
    /// `prefix` is prepended to every artifact name and pointer key; pass an empty
    /// string for a single-tenant bucket. Fails only if the current-thread Tokio
    /// runtime cannot be constructed.
    pub fn new(store: Arc<dyn ObjectStore>, prefix: impl Into<String>) -> Result<Self> {
        let runtime = Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| {
                ArtifactStoreError::Backend(format!("failed to build async runtime: {error}"))
            })?;
        Ok(Self {
            store,
            prefix: prefix.into(),
            runtime,
        })
    }

    /// Joins a store-relative artifact name onto the tenant prefix to form the
    /// object key. `object_store`'s `Path` normalises the segments, so a name can
    /// never traverse above the prefix.
    fn object_path(&self, name: &str) -> ObjectPath {
        if self.prefix.is_empty() {
            ObjectPath::from(name)
        } else {
            ObjectPath::from(format!("{}/{}", self.prefix.trim_end_matches('/'), name))
        }
    }

    /// The pointer object key for an index key: `<prefix>/<key>.commit.json`.
    fn pointer_path(&self, key: &str) -> ObjectPath {
        self.object_path(&format!("{key}.commit.json"))
    }

    /// Fetches an object's bytes, mapping a missing object to `None` rather than an
    /// error, so callers can distinguish absence from a transport failure.
    fn get_optional(&self, path: &ObjectPath) -> Result<Option<GetBytes>> {
        match self.runtime.block_on(self.store.get(path)) {
            Ok(result) => {
                let version = UpdateVersion {
                    e_tag: result.meta.e_tag.clone(),
                    version: result.meta.version.clone(),
                };
                let bytes = self
                    .runtime
                    .block_on(result.bytes())
                    .map_err(map_backend_error)?;
                Ok(Some(GetBytes {
                    bytes: bytes.to_vec(),
                    version,
                }))
            }
            Err(ObjectError::NotFound { .. }) => Ok(None),
            Err(error) => Err(map_backend_error(error)),
        }
    }
}

/// An object's bytes plus the version handle needed for a conditional overwrite.
struct GetBytes {
    bytes: Vec<u8>,
    version: UpdateVersion,
}

impl ArtifactStore for ObjectArtifactStore {
    fn read_bytes(&self, name: &str) -> Result<Vec<u8>> {
        let path = self.object_path(name);
        self.get_optional(&path)?
            .map(|got| got.bytes)
            .ok_or_else(|| ArtifactStoreError::NotFound(name.to_string()))
    }

    fn write_bytes_if_absent(&self, name: &str, data: &[u8], sha256: &str) -> Result<()> {
        // Verify before any upload so corruption is never stored.
        let digest = sha256_hex(data);
        if digest != sha256 {
            return Err(ArtifactStoreError::Integrity(format!(
                "artifact {name:?} failed checksum: expected {sha256}, computed {digest}"
            )));
        }
        let path = self.object_path(name);
        let options = PutOptions {
            mode: PutMode::Create,
            ..PutOptions::default()
        };
        match self
            .runtime
            .block_on(self.store.put_opts(&path, data.to_vec().into(), options))
        {
            Ok(_) => Ok(()),
            // The name already exists. Identical bytes are an idempotent no-op;
            // different bytes are a genuine conflict we refuse rather than clobber
            // (artifacts are immutable/content-addressed).
            Err(ObjectError::AlreadyExists { .. }) => {
                let existing = self.read_bytes(name)?;
                if sha256_hex(&existing) == sha256 {
                    Ok(())
                } else {
                    Err(ArtifactStoreError::Integrity(format!(
                        "artifact {name:?} already exists with different content; refusing to \
                         overwrite an immutable artifact"
                    )))
                }
            }
            Err(error) => Err(map_backend_error(error)),
        }
    }

    fn contains(&self, name: &str) -> Result<bool> {
        // A HEAD request: presence without fetching the object's bytes.
        let path = self.object_path(name);
        match self.runtime.block_on(self.store.head(&path)) {
            Ok(_) => Ok(true),
            Err(ObjectError::NotFound { .. }) => Ok(false),
            Err(error) => Err(map_backend_error(error)),
        }
    }

    fn read_pointer(&self, key: &str) -> Result<Option<Value>> {
        let path = self.pointer_path(key);
        match self.get_optional(&path)? {
            Some(got) => Ok(Some(validate_pointer_document(&got.bytes)?)),
            None => Ok(None),
        }
    }

    fn compare_and_swap_pointer(
        &self,
        key: &str,
        old_body: Option<&Value>,
        new_body: &Value,
    ) -> Result<()> {
        let path = self.pointer_path(key);
        let document = serialize_pointer_document(new_body)?;
        let current = self.get_optional(&path)?;

        // Precondition on the full committed body (a generation number is not a
        // unique version token). The e_tag from the same read then arms the
        // conditional put below as a second, race-tight guard.
        let current_body = current
            .as_ref()
            .map(|got| validate_pointer_document(&got.bytes))
            .transpose()?;
        if current_body.as_ref() != old_body {
            return Err(ArtifactStoreError::PointerConflict {
                key: key.to_string(),
                expected: old_body.and_then(body_generation),
                found: current_body.as_ref().and_then(body_generation),
            });
        }

        // Conditional write: create when the pointer must not yet exist, else
        // overwrite only if it still matches the version we just read. Either
        // precondition failure means a concurrent writer moved the pointer between
        // our read and this write.
        let mode = match &current {
            Some(got) => PutMode::Update(got.version.clone()),
            None => PutMode::Create,
        };
        let options = PutOptions {
            mode,
            ..PutOptions::default()
        };
        match self
            .runtime
            .block_on(self.store.put_opts(&path, document.into(), options))
        {
            Ok(_) => Ok(()),
            Err(ObjectError::AlreadyExists { .. } | ObjectError::Precondition { .. }) => {
                Err(ArtifactStoreError::PointerConflict {
                    key: key.to_string(),
                    expected: old_body.and_then(body_generation),
                    // The read-side body is stale by now; report the generation we
                    // expected and let the caller re-read to learn the new value.
                    found: current_body.as_ref().and_then(body_generation),
                })
            }
            Err(error) => Err(map_backend_error(error)),
        }
    }
}

/// Serialises a committed body into pointer-document bytes — the shared
/// engine-writer round-trip in [`snapshot_identity`](crate::snapshot_identity).
fn serialize_pointer_document(body: &Value) -> Result<Vec<u8>> {
    crate::snapshot_identity::pointer_document(body)
}

/// Validates pointer-document bytes and returns the committed body, reusing the
/// engine's schema + `body_sha256` validation via a scratch file.
///
/// Fails closed on a garbled pointer (the engine's `read_commit_manifest` rejects
/// a bad schema version or body checksum), so a corrupted object never loads as a
/// valid generation.
fn validate_pointer_document(bytes: &[u8]) -> Result<Value> {
    let mut scratch = NamedTempFile::new()?;
    scratch.write_all(bytes)?;
    scratch.flush()?;
    let manifest = read_commit_manifest(scratch.path())?.ok_or_else(|| {
        ArtifactStoreError::Integrity("pointer document is empty or absent".to_string())
    })?;
    Ok(manifest.body)
}

/// Maps a non-`NotFound`, non-precondition object-store error to a backend error,
/// preserving the backend's own diagnostic.
fn map_backend_error(error: ObjectError) -> ArtifactStoreError {
    ArtifactStoreError::Backend(error.to_string())
}
