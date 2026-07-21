//! Local-filesystem [`ArtifactStore`]: the default backend.
//!
//! A committed generation already lives on disk as immutable `g<epoch>.*`
//! artifacts under `<key>.gen/` pinned by `<key>.commit.json`, so the local store
//! is a thin wrapper over `lodedb-core`'s commit-manifest primitives; there is
//! no second format. Object-storage backends (S3/GCS/Azure) belong in a later
//! milestone, not here.

use crate::artifact_store::{body_generation, ArtifactStore};
use crate::digest::{sha256_hex_finish, sha256_hex_reader, COPY_BUFFER_BYTES};
use crate::error::{ArtifactStoreError, Result};
use crate::paths::resolve_within;
use lodedb_core::storage::commit_manifest::{
    commit_manifest_path, read_commit_manifest, write_commit_manifest,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{Read, Write};
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

    /// The idempotence/immutability answer for an already-present name:
    /// identical content (compared by a streaming re-hash) is Ok, different
    /// content refuses. Artifacts are immutable.
    fn refuse_unless_identical(&self, name: &str, path: &Path, sha256: &str) -> Result<()> {
        let (existing, _bytes) = sha256_hex_reader(&mut File::open(path)?)?;
        if existing == sha256 {
            return Ok(());
        }
        Err(ArtifactStoreError::Integrity(format!(
            "artifact {name:?} already exists with different content (stored {existing}, \
             incoming {sha256}); refusing to overwrite an immutable artifact"
        )))
    }
}

impl ArtifactStore for LocalArtifactStore {
    fn open_read<'a>(&'a self, name: &str) -> Result<Box<dyn Read + 'a>> {
        let path = resolve_within(&self.root, &self.root.join(name))?;
        match File::open(&path) {
            Ok(handle) => Ok(Box::new(handle)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                Err(ArtifactStoreError::NotFound(name.to_string()))
            }
            Err(error) => Err(ArtifactStoreError::Io(error)),
        }
    }

    fn write_stream_if_absent(
        &self,
        name: &str,
        data: &mut dyn Read,
        sha256: &str,
        _size_hint: u64,
    ) -> Result<()> {
        let path = resolve_within(&self.root, &self.root.join(name))?;
        if path.exists() {
            // Names are epoch-addressed, not sha-derived, so two independent
            // lineages can collide on a name. Identical bytes are an idempotent
            // no-op; different bytes are a genuine conflict we refuse rather than
            // clobber (the immutability invariant). Hash the existing file
            // streaming; it can be as large as any transferred base.
            return self.refuse_unless_identical(name, &path, sha256);
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        // Stream into a UNIQUELY-NAMED sibling scratch (two concurrent writers
        // of one name must never share a scratch), hashing as we copy, and
        // verify the digest BEFORE publication. The publish is
        // `persist_noclobber`, an atomic no-replace claim, so the losing
        // writer of a name race falls into the identical-bytes check instead
        // of overwriting bytes a committed pointer may already reference.
        // Peak memory is one copy buffer regardless of artifact size.
        let parent = path.parent().unwrap_or(Path::new("."));
        let file_name = path
            .file_name()
            .and_then(|file_name| file_name.to_str())
            .unwrap_or("artifact");
        let mut scratch = tempfile::Builder::new()
            .prefix(&format!(".{file_name}."))
            .suffix(".tmp")
            .tempfile_in(parent)?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; COPY_BUFFER_BYTES];
        loop {
            let read = data.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            scratch.write_all(&buffer[..read])?;
        }
        let digest = sha256_hex_finish(hasher);
        if digest != sha256 {
            // Dropping the NamedTempFile unlinks the scratch.
            return Err(ArtifactStoreError::Integrity(format!(
                "artifact {name:?} failed checksum: expected {sha256}, computed {digest}"
            )));
        }
        if self.fsync {
            scratch.as_file().sync_all()?;
        }
        let scratch_path = scratch.path().to_path_buf();
        match scratch.persist_noclobber(&path) {
            Ok(_) => {
                // On most filesystems the no-clobber persist is an atomic
                // rename and the scratch name is gone; the hard-link+unlink
                // fallback some filesystems take can leave the scratch link
                // behind on unlink failure, so sweep it (litter, not
                // correctness; the published bytes are already in place).
                if scratch_path.exists() {
                    let _ = fs::remove_file(&scratch_path);
                }
            }
            Err(error) if error.error.kind() == std::io::ErrorKind::AlreadyExists => {
                // Lost a same-name race: never replace, compare instead.
                return self.refuse_unless_identical(name, &path, sha256);
            }
            Err(error) => return Err(ArtifactStoreError::Io(error.error)),
        }
        if self.fsync {
            File::open(parent)?.sync_all()?;
        }
        Ok(())
    }

    fn contains(&self, name: &str) -> Result<bool> {
        // A metadata probe, not a read. `try_exists` surfaces genuine I/O
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
