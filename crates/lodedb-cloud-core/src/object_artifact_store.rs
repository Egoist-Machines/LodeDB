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
use crate::error::{ArtifactStoreError, Result};
use bytes::{Buf, Bytes};
use futures::stream::BoxStream;
use futures::StreamExt;
use lodedb_core::storage::commit_manifest::parse_commit_manifest;
use object_store::path::Path as ObjectPath;
use object_store::{Error as ObjectError, ObjectStore, PutMode, PutOptions, UpdateVersion};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
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

    /// Claims the final artifact name from a completed scratch upload:
    /// `copy_if_not_exists` where the backend supports it (atomic — a
    /// concurrent claimant loses cleanly and falls into the identical-bytes
    /// check), probe-then-copy where it does not (the residual race, now
    /// confined to backends without any conditional copy, e.g. plain S3
    /// without `AWS_COPY_IF_NOT_EXISTS`).
    fn claim_scratch(&self, name: &str, sha256: &str, scratch: &ObjectPath) -> Result<()> {
        let path = self.object_path(name);
        match self
            .runtime
            .block_on(self.store.copy_if_not_exists(scratch, &path))
        {
            Ok(()) => Ok(()),
            Err(ObjectError::AlreadyExists { .. } | ObjectError::Precondition { .. }) => {
                self.refuse_unless_identical(name, sha256)
            }
            Err(ObjectError::NotSupported { .. } | ObjectError::NotImplemented) => {
                if self.contains(name)? {
                    return self.refuse_unless_identical(name, sha256);
                }
                self.runtime
                    .block_on(self.store.copy(scratch, &path))
                    .map_err(map_backend_error)
            }
            Err(error) => Err(map_backend_error(error)),
        }
    }

    /// The idempotence/immutability answer for an already-present name:
    /// identical content (compared by a streaming re-hash) is Ok, different
    /// content refuses — artifacts are immutable.
    fn refuse_unless_identical(&self, name: &str, sha256: &str) -> Result<()> {
        let mut existing = self.open_read(name)?;
        let (digest, _bytes) = crate::digest::sha256_hex_reader(&mut *existing)?;
        if digest == sha256 {
            Ok(())
        } else {
            Err(ArtifactStoreError::Integrity(format!(
                "artifact {name:?} already exists with different content; refusing to \
                 overwrite an immutable artifact"
            )))
        }
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

/// Uploads at or below this size use one conditional `PutMode::Create` (a true
/// atomic create-if-absent); anything larger streams as a sequential multipart
/// upload with `MULTIPART_CHUNK_BYTES` parts, bounding memory to one part.
/// Object stores offer no conditional create for multipart, so the large path
/// claims via a scratch key — see `write_stream_if_absent`.
const MULTIPART_THRESHOLD_BYTES: usize = 8 * 1024 * 1024;
const MULTIPART_CHUNK_BYTES: usize = 8 * 1024 * 1024;

/// S3 caps a single copy operation (and one `UploadPartCopy` part — what the
/// `AWS_COPY_IF_NOT_EXISTS=multipart` claim uses) at 5 GiB, so the
/// scratch-then-conditional-claim strategy only works below it. Larger
/// artifacts stream their multipart directly onto the final name behind the
/// existence probe: no object-store primitive can claim them atomically
/// through this API, and failing the whole upload at claim time (after
/// streaming every byte) would be strictly worse than the probe's documented
/// residual race. The size hint decides the strategy up front; an
/// unknown-size stream that crosses this ceiling mid-upload fails early
/// (never at claim time, after every byte moved), and a hint that overshoots
/// the ceiling merely routes a smaller object through the direct path — a
/// strategy choice, never an integrity one (the digest gate is unconditional).
const CLAIMABLE_LIMIT_BYTES: u64 = 5 * 1024 * 1024 * 1024;

/// Process-local uniquifier for scratch upload keys: two same-process uploads
/// of one name (or one nanosecond) must never share a scratch object.
static SCRATCH_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A synchronous [`Read`] over an object's byte stream: each `read` pulls the
/// next chunk through the store's own current-thread runtime, so a download's
/// peak memory is one network chunk — never the object.
struct BlockingRead<'a> {
    runtime: &'a Runtime,
    stream: BoxStream<'static, object_store::Result<Bytes>>,
    current: Bytes,
}

impl Read for BlockingRead<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if buf.is_empty() {
            // A zero-length read must answer immediately, never poll the
            // network (Read's contract: Ok(0) is also the empty-buffer case).
            return Ok(0);
        }
        while self.current.is_empty() {
            match self.runtime.block_on(self.stream.next()) {
                None => return Ok(0),
                Some(Ok(chunk)) => self.current = chunk,
                Some(Err(error)) => return Err(std::io::Error::other(error)),
            }
        }
        let take = buf.len().min(self.current.len());
        buf[..take].copy_from_slice(&self.current[..take]);
        self.current.advance(take);
        Ok(take)
    }
}

impl ArtifactStore for ObjectArtifactStore {
    fn open_read<'a>(&'a self, name: &str) -> Result<Box<dyn Read + 'a>> {
        let path = self.object_path(name);
        match self.runtime.block_on(self.store.get(&path)) {
            Ok(result) => Ok(Box::new(BlockingRead {
                runtime: &self.runtime,
                stream: result.into_stream(),
                current: Bytes::new(),
            })),
            Err(ObjectError::NotFound { .. }) => {
                Err(ArtifactStoreError::NotFound(name.to_string()))
            }
            Err(error) => Err(map_backend_error(error)),
        }
    }

    fn write_stream_if_absent(
        &self,
        name: &str,
        data: &mut dyn Read,
        sha256: &str,
        size_hint: u64,
    ) -> Result<()> {
        let path = self.object_path(name);
        // Read up to the multipart threshold first (grow-on-demand: a tiny
        // delta segment allocates its own size, never the full threshold).
        // EOF inside the threshold keeps the small path's true conditional
        // create; anything larger streams as multipart parts, holding one
        // part in memory at a time.
        let mut first = Vec::new();
        (&mut *data)
            .take((MULTIPART_THRESHOLD_BYTES + 1) as u64)
            .read_to_end(&mut first)?;

        if first.len() <= MULTIPART_THRESHOLD_BYTES {
            let mut hasher = Sha256::new();
            hasher.update(&first);
            let digest = crate::digest::sha256_hex_finish(hasher);
            if digest != sha256 {
                return Err(ArtifactStoreError::Integrity(format!(
                    "artifact {name:?} failed checksum: expected {sha256}, computed {digest}"
                )));
            }
            let options = PutOptions {
                mode: PutMode::Create,
                ..PutOptions::default()
            };
            return match self
                .runtime
                .block_on(self.store.put_opts(&path, first.into(), options))
            {
                Ok(_) => Ok(()),
                // The name already exists. Identical bytes are an idempotent
                // no-op; different bytes are a genuine conflict we refuse
                // rather than clobber (artifacts are immutable).
                Err(ObjectError::AlreadyExists { .. }) => self.refuse_unless_identical(name, sha256),
                Err(error) => Err(map_backend_error(error)),
            };
        }

        // Large object. Multipart completion has no conditional-create mode,
        // so completing directly on the final name could overwrite a
        // concurrent writer's artifact AFTER that writer's pointer committed
        // — silent corruption of a committed generation. Instead the parts
        // stream to a unique scratch key and the final name is claimed with
        // `copy_if_not_exists`, which is atomic wherever the backend supports
        // it (natively on the in-memory test store; via
        // `AWS_COPY_IF_NOT_EXISTS` on R2/MinIO/DynamoDB-locked S3), falling
        // back to probe-then-copy where it is not. Above the provider copy
        // ceiling (`CLAIMABLE_LIMIT_BYTES`) no claim primitive exists at all,
        // so those artifacts stream directly onto the final name behind the
        // probe — the documented residual race, confined to >5 GiB objects.
        if self.contains(name)? {
            return self.refuse_unless_identical(name, sha256);
        }
        let claim_via_scratch = size_hint == 0 || size_hint <= CLAIMABLE_LIMIT_BYTES;
        // The scratch key embeds the EXPECTED digest: even if two uploaders
        // somewhere collided on the rest of the key (pid + clock + counter
        // are only process-unique), they can only share a scratch when they
        // are writing identical content — a substitution can never smuggle
        // bytes past the digest gate into the claim.
        let scratch = claim_via_scratch.then(|| {
            self.object_path(&format!(
                "{name}.upload-{sha256}-{}-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|elapsed| elapsed.as_nanos())
                    .unwrap_or(0),
                SCRATCH_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            ))
        });
        let upload_target = scratch.as_ref().unwrap_or(&path);
        // One FIXED part size for the whole upload, chosen from the size
        // hint: R2 requires every non-final part to be the same size, and
        // S3 caps an upload at 10,000 parts — so the hint scales the part
        // size up front (with headroom for a hint that undershoots) instead
        // of growing parts mid-stream.
        let part_size = if size_hint > 0 {
            let for_part_limit = (size_hint / 9_000).max(1) as usize;
            for_part_limit
                .div_ceil(MULTIPART_CHUNK_BYTES)
                .max(1)
                .saturating_mul(MULTIPART_CHUNK_BYTES)
        } else {
            MULTIPART_CHUNK_BYTES
        };
        let mut upload = self
            .runtime
            .block_on(self.store.put_multipart(upload_target))
            .map_err(map_backend_error)?;
        let streamed = (|| -> Result<()> {
            // Re-chunk the lookahead + the rest of the stream into uniform
            // parts: fill each buffer to exactly `part_size` (only the final
            // part may be short), hashing as the parts fill.
            let mut hasher = Sha256::new();
            let mut source = std::io::Read::chain(std::io::Cursor::new(first), &mut *data);
            let mut total = 0u64;
            loop {
                let mut part = Vec::new();
                (&mut source)
                    .take(part_size as u64)
                    .read_to_end(&mut part)?;
                if part.is_empty() {
                    break;
                }
                total += part.len() as u64;
                if claim_via_scratch && total > CLAIMABLE_LIMIT_BYTES {
                    // An unknown-size (or undershooting-hint) stream just
                    // crossed the provider copy ceiling: the scratch object
                    // could never be claimed, so failing NOW beats streaming
                    // the rest and failing at claim time. The caller re-runs
                    // with the artifact's real size (transfers always pass
                    // the manifest-recorded size and never land here).
                    return Err(ArtifactStoreError::Backend(format!(
                        "artifact {name:?} exceeds the {CLAIMABLE_LIMIT_BYTES}-byte \
                         conditional-claim ceiling but was written without a size hint; \
                         pass the artifact's size so the upload can target the final \
                         key directly"
                    )));
                }
                hasher.update(&part);
                let is_final = part.len() < part_size;
                let payload: object_store::PutPayload = Bytes::from(part).into();
                self.runtime
                    .block_on(upload.put_part(payload))
                    .map_err(map_backend_error)?;
                if is_final {
                    break;
                }
            }
            let digest = crate::digest::sha256_hex_finish(hasher);
            if digest != sha256 {
                return Err(ArtifactStoreError::Integrity(format!(
                    "artifact {name:?} failed checksum: expected {sha256}, computed {digest}"
                )));
            }
            self.runtime
                .block_on(upload.complete())
                .map_err(map_backend_error)?;
            Ok(())
        })();
        if let Err(error) = streamed {
            // Release the backend's part storage; a failed abort leaves
            // billable parts behind, so its diagnostic rides along with the
            // primary failure instead of vanishing.
            return Err(match self.runtime.block_on(upload.abort()) {
                Ok(()) => error,
                Err(abort_error) => ArtifactStoreError::Backend(format!(
                    "{error} (aborting the multipart upload also failed, which can \
                     leave stored parts behind: {abort_error})"
                )),
            });
        }
        let Some(scratch) = scratch else {
            // Streamed directly onto the final name (above the claim ceiling).
            return Ok(());
        };
        let claimed = self.claim_scratch(name, sha256, &scratch);
        // The scratch object is redundant on every path once the claim
        // resolved; a failed delete is litter, not corruption.
        let _ = self.runtime.block_on(self.store.delete(&scratch));
        claimed
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

/// Serialises a committed body into pointer-document bytes — the engine's own
/// rendering, via [`snapshot_identity`](crate::snapshot_identity).
fn serialize_pointer_document(body: &Value) -> Result<Vec<u8>> {
    crate::snapshot_identity::pointer_document(body)
}

/// Validates pointer-document bytes and returns the committed body, through
/// the engine's own schema + `body_sha256` validation (`parse_commit_manifest`
/// — no scratch file; this runs on every pointer read).
///
/// Fails closed on a garbled pointer (bad schema version or body checksum), so
/// a corrupted object never loads as a valid generation.
fn validate_pointer_document(bytes: &[u8]) -> Result<Value> {
    // Core, not Integrity: a corrupt pointer read previously surfaced the
    // engine's structured error (with its source chain and error code)
    // through `read_commit_manifest`, and callers keep that contract.
    let manifest = parse_commit_manifest(bytes).map_err(ArtifactStoreError::Core)?;
    Ok(manifest.body)
}

/// Maps a non-`NotFound`, non-precondition object-store error to a backend error,
/// preserving the backend's own diagnostic.
fn map_backend_error(error: ObjectError) -> ArtifactStoreError {
    ArtifactStoreError::Backend(error.to_string())
}
