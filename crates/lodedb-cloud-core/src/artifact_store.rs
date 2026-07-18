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

/// Reads/writes immutable artifacts and swaps a root pointer atomically.
///
/// The interface is deliberately small: byte-level artifact I/O plus a pointer
/// compare-and-swap. That is everything the generation-addressed commit format
/// needs as a cloud substrate.
pub trait ArtifactStore {
    /// Returns one artifact's bytes; [`ArtifactStoreError::NotFound`] if the name
    /// is absent.
    ///
    /// [`ArtifactStoreError::NotFound`]: crate::ArtifactStoreError::NotFound
    fn read_bytes(&self, name: &str) -> Result<Vec<u8>>;

    /// Writes one immutable artifact unless it already exists.
    ///
    /// `sha256` is the expected lowercase-hex digest of `data`; a mismatch is an
    /// integrity error so corruption is never stored. A name already present
    /// with identical bytes is a no-op (idempotent re-push); present with
    /// *different* bytes is a conflict — artifacts are immutable and are never
    /// overwritten in place.
    fn write_bytes_if_absent(&self, name: &str, data: &[u8], sha256: &str) -> Result<()>;

    /// Whether an artifact named `name` is present, without asserting anything
    /// about its content.
    ///
    /// The default reads the artifact and maps
    /// [`NotFound`](crate::ArtifactStoreError::NotFound) to `false`; backends
    /// with a cheap existence primitive (a filesystem stat, an object-store
    /// HEAD) should override it to avoid fetching the bytes.
    fn contains(&self, name: &str) -> Result<bool> {
        match self.read_bytes(name) {
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
    /// A generation number is not a unique version token — two independent lineages
    /// can share one with different content — so comparing the whole body is what
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
/// for readability — the swap precondition is the full body, not this number.
pub(crate) fn body_generation(body: &Value) -> Option<u64> {
    body.get("generation").and_then(Value::as_u64)
}
