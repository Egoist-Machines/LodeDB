//! Content identity for a committed generation body.
//!
//! Sync needs to ask "are these two committed generations the same?" across
//! machines and stores, where generation *numbers* are not unique version
//! tokens (two independent lineages can share one). The answer is the engine's
//! own canonical body digest:
//!
//! - [`snapshot_id`] — the `body_sha256` the engine records in
//!   `<key>.commit.json` for exactly this body. Two stores hold the same bytes
//!   iff their snapshot ids match; storage and authorization decisions key on
//!   it.
//! - [`logical_id`] — the snapshot id of the body's fully *redacted* form. A
//!   full push and a redacted push of one engine commit share a `logical_id`
//!   while their `snapshot_id`s differ, so lineage comparison ("is this the
//!   same commit?") runs on `logical_id` and never mistakes a redaction for a
//!   divergence.
//!
//! The digest is over the engine's canonical JSON, which this crate must never
//! reimplement. `body_sha256` itself is `pub(crate)` in `lodedb-core`, but
//! `write_commit_manifest` emits `{"body":…,"body_sha256":"…",…}`, so a
//! scratch-file round-trip through the engine's own writer yields the exact
//! canonical digest — the same trick `ObjectArtifactStore` uses for pointer
//! documents.
//!
//! TODO(upstream): ask lodedb-core to export `body_sha256(body)` directly and
//! drop the scratch-file round-trip.

use crate::error::{ArtifactStoreError, Result};
use crate::transfer_policy::TransferPolicy;
use lodedb_core::storage::commit_manifest::write_commit_manifest;
use serde_json::Value;
use tempfile::NamedTempFile;

/// Serialises `body` into the engine's canonical pointer-document bytes —
/// the exact `{"body":…,"body_sha256":…,"schema_version":…}` file a
/// `<key>.commit.json` pointer carrying this body holds on disk.
///
/// Routing through the engine's own `write_commit_manifest` (via a scratch
/// file) guarantees the body checksum and schema envelope are byte-identical
/// to an engine-written pointer, so `read_commit_manifest` validates the
/// result the same way. The managed transfer plane ships these bytes to the
/// control plane, which stores them verbatim for the object-store pointer
/// mirror — the server never re-serialises a pointer itself.
pub fn pointer_document(body: &Value) -> Result<Vec<u8>> {
    let scratch = NamedTempFile::new()?;
    write_commit_manifest(scratch.path(), body, false)?;
    Ok(std::fs::read(scratch.path())?)
}

/// Extracts the canonical `body_sha256` from pointer-document bytes produced
/// by [`pointer_document`].
pub fn identity_from_document(document: &[u8]) -> Result<String> {
    let document: Value = serde_json::from_slice(document).map_err(|error| {
        ArtifactStoreError::Integrity(format!(
            "engine-written pointer document failed to parse: {error}"
        ))
    })?;
    document
        .get("body_sha256")
        .and_then(Value::as_str)
        .filter(|digest| !digest.is_empty())
        .map(str::to_string)
        .ok_or_else(|| {
            ArtifactStoreError::Integrity(
                "engine-written pointer document carries no body_sha256".to_string(),
            )
        })
}

/// Returns the engine's canonical `body_sha256` for `body` — the digest a
/// `<key>.commit.json` pointer carrying this exact body records.
pub fn snapshot_id(body: &Value) -> Result<String> {
    identity_from_document(&pointer_document(body)?)
}

/// Returns the snapshot id of `body`'s fully redacted form.
///
/// Redaction ([`TransferPolicy::redacted`]) nulls the payload-bearing stores,
/// and redacting an already-redacted body is a no-op, so every push of one
/// engine commit — full, redacted, or partial — maps to one `logical_id`. A
/// corollary used by the sync classifier: a body is fully redacted iff its
/// `snapshot_id` equals its `logical_id`.
pub fn logical_id(body: &Value) -> Result<String> {
    snapshot_id(&TransferPolicy::redacted().redact_body(body))
}
