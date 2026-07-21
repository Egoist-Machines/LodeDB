//! Content identity for a committed generation body.
//!
//! Sync needs to ask "are these two committed generations the same?" across
//! machines and stores, where generation *numbers* are not unique version
//! tokens (two independent lineages can share one). The answer is the engine's
//! own canonical body digest:
//!
//! - [`snapshot_id`]: the `body_sha256` the engine records in
//!   `<key>.commit.json` for exactly this body. Two stores hold the same bytes
//!   iff their snapshot ids match; storage and authorization decisions key on
//!   it.
//! - [`logical_id`]: the snapshot id of the body's fully *redacted* form. A
//!   full push and a redacted push of one engine commit share a `logical_id`
//!   while their `snapshot_id`s differ, so lineage comparison ("is this the
//!   same commit?") runs on `logical_id` and never mistakes a redaction for a
//!   divergence.
//!
//! The digest is over the engine's canonical JSON, which this crate must never
//! reimplement: `lodedb-core` exports `commit_body_sha256` and
//! `render_commit_manifest`, so identities and pointer documents come straight
//! from the engine's own writer with no scratch-file round trip. That matters
//! because every sync/status classifies several identities per run and the
//! object-store backend validates a pointer document on every read.

use crate::error::{ArtifactStoreError, Result};
use crate::transfer_policy::TransferPolicy;
use lodedb_core::storage::commit_manifest::{
    commit_body_sha256, parse_commit_manifest, render_commit_manifest,
};
use serde_json::Value;

/// Serialises `body` into the engine's canonical pointer-document bytes,
/// the exact `{"body":…,"body_sha256":…,"schema_version":…}` file a
/// `<key>.commit.json` pointer carrying this body holds on disk.
///
/// Rendered by the engine's own `render_commit_manifest` (the text
/// `write_commit_manifest` persists), so the body checksum and schema envelope
/// are byte-identical to an engine-written pointer and `read_commit_manifest`
/// validates the result the same way. The managed transfer plane ships these
/// bytes to the control plane, which stores them verbatim for the
/// object-store pointer mirror; the server never re-serialises a pointer
/// itself.
pub fn pointer_document(body: &Value) -> Result<Vec<u8>> {
    Ok(render_commit_manifest(body)?.into_bytes())
}

/// Extracts the canonical `body_sha256` from pointer-document bytes produced
/// by [`pointer_document`], validating the document (schema version and body
/// checksum) through the engine's own parser.
pub fn identity_from_document(document: &[u8]) -> Result<String> {
    let manifest = parse_commit_manifest(document).map_err(|error| {
        ArtifactStoreError::Integrity(format!("pointer document failed validation: {error}"))
    })?;
    Ok(manifest.body_sha256)
}

/// Returns the engine's canonical `body_sha256` for `body`, the digest a
/// `<key>.commit.json` pointer carrying this exact body records.
pub fn snapshot_id(body: &Value) -> Result<String> {
    Ok(commit_body_sha256(body)?)
}

/// Returns the snapshot id of `body`'s fully redacted form.
///
/// Redaction ([`TransferPolicy::redacted`]) nulls the payload-bearing stores,
/// and redacting an already-redacted body is a no-op, so every push of one
/// engine commit (full, redacted, or partial) maps to one `logical_id`. A
/// corollary used by the sync classifier: a body is fully redacted iff its
/// `snapshot_id` equals its `logical_id`.
pub fn logical_id(body: &Value) -> Result<String> {
    snapshot_id(&TransferPolicy::redacted().redact_body(body))
}
