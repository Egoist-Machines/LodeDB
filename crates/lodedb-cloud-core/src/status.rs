//! Compare a local generation against a remote one.
//!
//! [`compare_generations`] answers "what would a push move, and are the two ends
//! already in sync?" purely from two inventories — no bytes are read. It composes
//! [`diff_inventories`](crate::diff_inventories), so the O(changed) upload set and
//! the base-vs-delta signal are computed the same way the transfer computes them.
//!
//! The comparison is direction-aware (push: local -> remote) and compares the two
//! inventories exactly as given. A caller applying a [`TransferPolicy`] should
//! build the local inventory from the *redacted* body, so `in_sync` reflects the
//! redacted push it intends to make — [`status_for_push`] does exactly that over
//! two stores. `in_sync` means a push would move nothing at
//! all: no artifacts to upload *and* the remote's committed body already equals the
//! body the push would publish. The body check matters because a push republishes
//! the pointer whenever the bodies differ even when no bytes move — e.g. a redacted
//! push against a previously-full remote uploads nothing yet must still swap the
//! pointer to drop the text reference, so it is correctly reported as not in sync.

use crate::artifact_store::ArtifactStore;
use crate::error::Result;
use crate::generation_inventory::{diff_inventories, inventory_from_body, GenerationInventory};
use crate::transfer_policy::TransferPolicy;

/// Metrics-only comparison of a local and a remote committed generation.
///
/// `local_*`/`remote_*` are `None` when that side holds no committed generation.
/// `artifacts_to_upload`/`bytes_to_upload`/`ships_base` describe a push from local
/// to remote (the O(changed) upload set). `in_sync` is true when a push would move
/// nothing — the remote already holds every artifact the local side would ship.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatusReport {
    pub index_key: String,
    pub local_generation: Option<u64>,
    pub remote_generation: Option<u64>,
    pub local_document_count: Option<u64>,
    pub remote_document_count: Option<u64>,
    pub local_chunk_count: Option<u64>,
    pub remote_chunk_count: Option<u64>,
    pub artifacts_to_upload: usize,
    pub bytes_to_upload: u64,
    pub ships_base: bool,
    pub in_sync: bool,
    /// Whether a *trusted* sync sidecar for this remote was found next to the
    /// local index (valid checksum, recorded against the same remote target).
    /// Always false from [`compare_generations`]/[`status_for_push`] — the
    /// sidecar lives on the local filesystem, so only the client edge
    /// ([`client_ops::status`](crate::client_ops::status)) can fill lineage.
    pub sidecar_present: bool,
    /// Whether a sidecar file was found but failed validation (torn or
    /// tampered) and was therefore ignored. Distinct from a sidecar recorded
    /// against a different remote, which is valid — just not trusted here.
    pub sidecar_corrupt: bool,
    /// The recorded base generation from the sidecar, when one was trusted.
    pub base_generation: Option<u64>,
    /// The three-pointer [`SyncClassification`](crate::sync_plan::SyncClassification)
    /// name (`in_sync`/`local_ahead`/…), when lineage was evaluated. `None`
    /// from the store-level comparison, which has no sidecar to consult.
    pub classification: Option<String>,
}

/// Reads both ends and compares them for a push of `index_key` under `policy`.
///
/// The local body is redacted by `policy` *before* inventorying, so the report
/// describes the push the caller would actually make (a redacted push neither
/// counts nor ships text/lexical artifacts, and `in_sync` compares against the
/// redacted body the push would publish). Read-only on both stores.
pub fn status_for_push(
    source: &dyn ArtifactStore,
    dest: &dyn ArtifactStore,
    index_key: &str,
    policy: TransferPolicy,
) -> Result<StatusReport> {
    let local_body = source
        .read_pointer(index_key)?
        .map(|body| policy.redact_body(&body));
    let local = inventory_from_body(index_key, local_body.as_ref())?;
    let remote_body = dest.read_pointer(index_key)?;
    let remote = inventory_from_body(index_key, remote_body.as_ref())?;
    Ok(compare_generations(
        index_key,
        local.as_ref(),
        remote.as_ref(),
    ))
}

/// Compares `local` against `remote` for a push (local -> remote).
///
/// `index_key` labels the report (the inventories carry it too, but either side
/// may be `None`). When `local` is `None` there is nothing to push, so the report
/// is empty and `in_sync` is true. When `local` is `Some`, the upload set is
/// `diff_inventories(local, remote)` and `in_sync` is true iff that set is empty
/// *and* the remote already holds the exact body the push would publish (so the
/// pointer would not be re-swapped).
pub fn compare_generations(
    index_key: &str,
    local: Option<&GenerationInventory>,
    remote: Option<&GenerationInventory>,
) -> StatusReport {
    let Some(local) = local else {
        // Nothing local to push: the report reflects only the remote side.
        return StatusReport {
            index_key: index_key.to_string(),
            local_generation: None,
            remote_generation: remote.map(|inventory| inventory.generation),
            local_document_count: None,
            remote_document_count: remote.map(|inventory| inventory.document_count),
            local_chunk_count: None,
            remote_chunk_count: remote.map(|inventory| inventory.chunk_count),
            artifacts_to_upload: 0,
            bytes_to_upload: 0,
            ships_base: false,
            in_sync: true,
            sidecar_present: false,
            sidecar_corrupt: false,
            base_generation: None,
            classification: None,
        };
    };

    let diff = diff_inventories(local, remote);
    // A push moves nothing only when there are no artifacts to upload AND the
    // remote already holds the exact body the push would publish — otherwise the
    // pointer is still re-swapped (e.g. a redaction that drops a store reference
    // without uploading any bytes).
    let body_matches_remote = remote.is_some_and(|remote| remote.root_body == local.root_body);
    StatusReport {
        index_key: index_key.to_string(),
        local_generation: Some(local.generation),
        remote_generation: remote.map(|inventory| inventory.generation),
        local_document_count: Some(local.document_count),
        remote_document_count: remote.map(|inventory| inventory.document_count),
        local_chunk_count: Some(local.chunk_count),
        remote_chunk_count: remote.map(|inventory| inventory.chunk_count),
        artifacts_to_upload: diff.to_upload.len(),
        bytes_to_upload: diff.upload_bytes,
        ships_base: diff.ships_base,
        in_sync: diff.to_upload.is_empty() && body_matches_remote,
        sidecar_present: false,
        sidecar_corrupt: false,
        base_generation: None,
        classification: None,
    }
}
