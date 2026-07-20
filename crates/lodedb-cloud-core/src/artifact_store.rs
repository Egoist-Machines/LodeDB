//! Storage abstraction for committed, generation-addressed artifacts.
//!
//! The substrate is the engine's existing on-disk layout: a committed generation
//! is a set of immutable, sha256-addressed `g<epoch>.*` artifacts under
//! `<key>.gen/` pinned by an atomic `<key>.commit.json` root pointer. An
//! [`ArtifactStore`] exposes exactly the operations that layout needs:
//!
//! - **names** are store-relative paths mirroring the on-disk layout, e.g.
//!   `idx.gen/g7.json` or `idx.gen/g7.json.json-delta/delta-00000000.jsd`;
//! - the **pointer key** is the index key (`idx` -> `idx.commit.json`).
//!
//! Implementations treat content artifacts as immutable and content-addressed
//! (written once, never overwritten), and commit a generation *only* by swapping
//! the root pointer. [`LocalArtifactStore`](crate::LocalArtifactStore) is the
//! default; object-storage backends (S3/GCS/Azure) land in later milestones with
//! the artifact names becoming object keys under a per-tenant prefix and the
//! pointer swap becoming a conditional write.

use crate::error::Result;
use serde_json::Value;
use std::io::Read;

/// Reads/writes immutable artifacts and swaps a root pointer atomically.
///
/// The interface is deliberately small: streaming artifact I/O plus a pointer
/// compare-and-swap. That is everything the generation-addressed commit format
/// needs as a cloud substrate. Streaming is the primitive. Vector bases run
/// to gigabytes, so a transfer's peak memory must be a fixed buffer, never a
/// function of artifact size; the buffered methods are conveniences for the
/// small payloads (pointer documents) built on top.
pub trait ArtifactStore {
    /// Opens one artifact for streaming reads;
    /// [`ArtifactStoreError::NotFound`] if the name is absent.
    ///
    /// [`ArtifactStoreError::NotFound`]: crate::ArtifactStoreError::NotFound
    fn open_read<'a>(&'a self, name: &str) -> Result<Box<dyn Read + 'a>>;

    /// Returns one artifact's bytes, fully buffered, for small payloads
    /// only; transfers and verification stream via [`open_read`](Self::open_read).
    fn read_bytes(&self, name: &str) -> Result<Vec<u8>> {
        let mut data = Vec::new();
        self.open_read(name)?.read_to_end(&mut data)?;
        Ok(data)
    }

    /// Streams one immutable artifact into the store unless it already
    /// exists, hashing incrementally as it copies.
    ///
    /// `sha256` is the expected lowercase-hex digest of the streamed bytes; a
    /// mismatch is an integrity error and nothing is stored, so corruption can
    /// never land, even when the source is another store's live stream. A
    /// name already present with identical bytes is a no-op (idempotent
    /// re-push); present with *different* bytes is a conflict. Artifacts are
    /// immutable and are never overwritten in place. On the already-present
    /// no-op path the incoming stream may be left partially (or wholly)
    /// unread and unvalidated: success then attests that the STORED bytes
    /// match `sha256`, not that the stream did.
    ///
    /// `size_hint` is the expected byte count (0 = unknown), advisory only:
    /// backends use it to pick an upload strategy (the object store's
    /// conditional-claim path has a provider copy-size ceiling), never to
    /// trust the stream's length. Transfers pass the manifest-recorded size.
    fn write_stream_if_absent(
        &self,
        name: &str,
        data: &mut dyn Read,
        sha256: &str,
        size_hint: u64,
    ) -> Result<()>;

    /// Buffered convenience over
    /// [`write_stream_if_absent`](Self::write_stream_if_absent), for small
    /// payloads a caller already holds in memory.
    fn write_bytes_if_absent(&self, name: &str, data: &[u8], sha256: &str) -> Result<()> {
        let mut cursor = data;
        self.write_stream_if_absent(name, &mut cursor, sha256, data.len() as u64)
    }

    /// Whether an artifact named `name` is present, without asserting anything
    /// about its content.
    ///
    /// The default opens the artifact and maps
    /// [`NotFound`](crate::ArtifactStoreError::NotFound) to `false`; backends
    /// with a cheap existence primitive (a filesystem stat, an object-store
    /// HEAD) should override it to avoid opening a stream.
    fn contains(&self, name: &str) -> Result<bool> {
        match self.open_read(name) {
            Ok(_) => Ok(true),
            Err(crate::ArtifactStoreError::NotFound(_)) => Ok(false),
            Err(error) => Err(error),
        }
    }

    /// Returns the committed root-manifest body for `key`, or `None`.
    ///
    /// The body carries the generation number, the base epoch, counts, and the
    /// per-store sub-manifests that name every artifact in the generation. It is
    /// the read side that [`compare_and_swap_pointer`](Self::compare_and_swap_pointer)
    /// and the inventory helpers need to learn the currently-committed generation.
    fn read_pointer(&self, key: &str) -> Result<Option<Value>>;

    /// Publishes `new_body` as the root pointer iff the currently committed body is
    /// exactly `old_body` (`None` means the pointer must not yet exist); otherwise
    /// [`ArtifactStoreError::PointerConflict`]. This swap is the only commit point:
    /// artifacts are published first, then the pointer flips all-or-nothing.
    ///
    /// The precondition is the full committed body, not just its generation number.
    /// A generation number is not a unique version token (two independent lineages
    /// can share one with different content), so comparing the whole body is what
    /// makes this a sound compare-and-swap rather than an ABA-prone check.
    ///
    /// [`ArtifactStoreError::PointerConflict`]: crate::ArtifactStoreError::PointerConflict
    fn compare_and_swap_pointer(
        &self,
        key: &str,
        old_body: Option<&Value>,
        new_body: &Value,
    ) -> Result<()>;
}

/// The generation number recorded in a committed body, if present.
///
/// Used only to annotate a [`PointerConflict`](crate::ArtifactStoreError::PointerConflict)
/// for readability; the swap precondition is the full body, not this number.
pub(crate) fn body_generation(body: &Value) -> Option<u64> {
    body.get("generation").and_then(Value::as_u64)
}
