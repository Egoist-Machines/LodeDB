//! Export a committed generation between two artifact stores.
//!
//! Copies exactly one committed generation from one [`ArtifactStore`] to another,
//! then publishes the destination's root pointer. Because the source is read
//! through its committed
//! root pointer (never the `.wal` tail or the on-disk per-store manifests), only
//! committed state ships — an uncommitted WAL write or a torn half-commit is
//! excluded by construction.
//!
//! The copy is O(changed): the destination's existing inventory is diffed against
//! the source's, so only missing artifacts are uploaded. The destination verifies
//! each artifact's checksum on write, and the pointer swap is the single commit
//! point, so a crash mid-transfer leaves the destination on its previous
//! generation with some extra immutable blobs — never a torn generation.
//!
//! Both ends are artifact stores, so a transfer is symmetric: a restore is the
//! same call with the stores swapped — the remote/backup store as `source` and
//! the local directory as `dest` (a source that was pushed redacted already omits
//! text, so [`TransferPolicy::full`] on the way back restores it verbatim). Follow
//! a local restore with
//! [`verify_local_generation_opens`](crate::verify_local_generation_opens) to
//! confirm the engine can open it.
//!
//! Every transfer states its [`TransferPolicy`] explicitly: a redacted push
//! publishes a body whose text/lexical sub-manifests are nulled and skips those
//! artifacts, so only redacted state leaves the machine; a full policy ships the
//! generation verbatim. There is deliberately no default-policy convenience —
//! shipping payload-bearing stores must be a visible choice at the call site.

use crate::artifact_store::ArtifactStore;
use crate::error::{ArtifactStoreError, Result};
use crate::generation_inventory::{diff_inventories, inventory_from_body};
use crate::transfer_policy::TransferPolicy;

/// Metrics-only summary of one generation transfer.
///
/// Carries counts and bytes only (no ids, text, or vectors), so it is safe to log
/// or surface in control-plane telemetry. `artifacts_written` are the blobs
/// actually uploaded; `artifacts_skipped` were already present at the
/// destination; `pointer_published` is false when the destination already pointed
/// at this generation (an idempotent no-op transfer).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferResult {
    pub index_key: String,
    pub generation: u64,
    pub artifacts_written: usize,
    pub artifacts_skipped: usize,
    pub bytes_written: u64,
    pub pointer_published: bool,
}

/// Copies `index_key`'s committed generation from `source` to `dest` under
/// `policy`.
///
/// A redacted `policy` publishes a body whose excluded text/lexical sub-manifests
/// are nulled and ships none of their artifacts, so the destination generation
/// genuinely omits that payload (see [`TransferPolicy::redact_body`]); a
/// [`TransferPolicy::full`] policy ships the generation verbatim. Reads the
/// source's committed root, diffs it against whatever the destination already
/// holds, uploads only the missing artifacts (each checksum-verified by the
/// destination), and finally swaps the destination's root pointer — the one
/// commit point. Idempotent: re-running when the destination already holds the
/// exact body this transfer would publish uploads nothing and leaves the pointer
/// untouched.
///
/// Returns [`ArtifactStoreError::NotFound`] when the source has no committed
/// generation for `index_key`. The pointer swap may return
/// [`ArtifactStoreError::PointerConflict`] if a concurrent writer advanced the
/// destination between the read and the swap.
pub fn export_generation(
    source: &dyn ArtifactStore,
    dest: &dyn ArtifactStore,
    index_key: &str,
    policy: TransferPolicy,
) -> Result<TransferResult> {
    export_generation_with_body(source, dest, index_key, policy).map(|(result, _)| result)
}

/// [`export_generation`], additionally returning the committed body the
/// destination now points at (the policy-redacted source body).
///
/// The sync layer records that body's identity in the sidecar; returning it
/// from the transfer itself (rather than re-reading the source afterwards)
/// means the recorded base is exactly what was published, even if a concurrent
/// engine commit advances the source mid-call. Crate-internal: the public
/// result type stays metrics-only (safe to log), which a full body is not.
pub(crate) fn export_generation_with_body(
    source: &dyn ArtifactStore,
    dest: &dyn ArtifactStore,
    index_key: &str,
    policy: TransferPolicy,
) -> Result<(TransferResult, serde_json::Value)> {
    let dest_body = dest.read_pointer(index_key)?;
    export_generation_against(source, dest, index_key, policy, dest_body)
}

/// [`export_generation_with_body`] with the destination's committed body
/// supplied by the caller instead of read here.
///
/// The pointer swap preconditions on exactly `dest_body`, so a caller that
/// already read (and classified) the destination — the sync verb — makes its
/// transfer conditional on the state it decided on: a concurrent advance of
/// the destination between that read and this call surfaces as a
/// [`PointerConflict`](ArtifactStoreError::PointerConflict), with the
/// atomicity the destination backend's `compare_and_swap_pointer` provides
/// (a true conditional write on object stores; read-check-then-replace on a
/// directory, per that impl's documented single-writer contract).
pub(crate) fn export_generation_against(
    source: &dyn ArtifactStore,
    dest: &dyn ArtifactStore,
    index_key: &str,
    policy: TransferPolicy,
    dest_body: Option<serde_json::Value>,
) -> Result<(TransferResult, serde_json::Value)> {
    let raw_body = source.read_pointer(index_key)?.ok_or_else(|| {
        ArtifactStoreError::NotFound(format!(
            "no committed generation to export for index key {index_key:?}"
        ))
    })?;
    export_generation_pinned(source, dest, index_key, policy, raw_body, dest_body)
}

/// [`export_generation_against`] with the *source's* committed body supplied
/// too, instead of re-read here.
///
/// The sync verb classifies a specific pair of bodies; pinning the source
/// means the transfer ships exactly the classified snapshot — a concurrent
/// source change cannot swap in a body the classifier never saw. The
/// artifact bytes are still read live by name, but every name/checksum comes
/// from the pinned body and the destination re-hashes on write, so a mutated
/// source artifact fails the transfer rather than shipping unclassified
/// bytes.
pub(crate) fn export_generation_pinned(
    source: &dyn ArtifactStore,
    dest: &dyn ArtifactStore,
    index_key: &str,
    policy: TransferPolicy,
    raw_body: serde_json::Value,
    dest_body: Option<serde_json::Value>,
) -> Result<(TransferResult, serde_json::Value)> {
    let staged = stage_generation_pinned(source, dest, index_key, policy, raw_body, dest_body)?;
    publish_staged(dest, index_key, staged)
}

/// Everything a staged (artifacts-uploaded, pointer-untouched) transfer knows:
/// the body a publish would commit, the destination body the publish must
/// precondition on, and the upload metrics.
///
/// Splitting staging from publication lets local restores insert their
/// acceptance checks (candidate verify-open, journal reconstruction) between
/// the two, so a failed check leaves the destination on its previous committed
/// generation — the pointer swap stays the single commit point.
pub(crate) struct StagedExport {
    pub source_body: serde_json::Value,
    pub dest_body: Option<serde_json::Value>,
    pub generation: u64,
    pub artifacts_total: usize,
    pub artifacts_written: usize,
    pub bytes_written: u64,
}

/// Uploads the missing artifact set for a pinned transfer WITHOUT touching the
/// destination pointer.
///
/// The destination re-hashes each artifact on write, so a corrupt source
/// artifact fails here — before any pointer moves. Redaction happens before
/// inventorying, exactly as in [`export_generation_pinned`].
pub(crate) fn stage_generation_pinned(
    source: &dyn ArtifactStore,
    dest: &dyn ArtifactStore,
    index_key: &str,
    policy: TransferPolicy,
    raw_body: serde_json::Value,
    dest_body: Option<serde_json::Value>,
) -> Result<StagedExport> {
    // Redact before inventorying so excluded artifacts are neither listed nor
    // uploaded, and the published body matches exactly what shipped.
    let source_body = policy.redact_body(&raw_body);
    let source_inventory = inventory_from_body(index_key, Some(&source_body))?
        .expect("inventory is Some when the body is Some");

    let dest_inventory = inventory_from_body(index_key, dest_body.as_ref())?;
    let diff = diff_inventories(&source_inventory, dest_inventory.as_ref());

    for artifact in &diff.to_upload {
        // Streamed end to end: the destination hashes while it copies, so the
        // transfer's peak memory is a fixed buffer (or one multipart chunk on
        // object storage) — never the artifact, which for a vector base can
        // be gigabytes. The manifest-recorded size rides along as the
        // strategy hint (it is advisory; the digest gate stays authoritative).
        let mut reader = source.open_read(&artifact.name)?;
        dest.write_stream_if_absent(
            &artifact.name,
            &mut *reader,
            &artifact.sha256,
            artifact.size_bytes,
        )?;
    }

    Ok(StagedExport {
        source_body,
        dest_body,
        generation: source_inventory.generation,
        artifacts_total: source_inventory.artifacts.len(),
        artifacts_written: diff.to_upload.len(),
        bytes_written: diff.upload_bytes,
    })
}

/// Publishes a staged transfer's pointer — the single commit point.
///
/// Skips the swap only when the destination already holds this exact committed
/// *body* — comparing the full body, not just the generation integer, because
/// two independent lineages can share a generation number with different
/// content. The swap preconditions on the exact body staging read
/// (`dest_body`), so a concurrent change between read and swap is caught as a
/// PointerConflict.
pub(crate) fn publish_staged(
    dest: &dyn ArtifactStore,
    index_key: &str,
    staged: StagedExport,
) -> Result<(TransferResult, serde_json::Value)> {
    let StagedExport {
        source_body,
        dest_body,
        generation,
        artifacts_total,
        artifacts_written,
        bytes_written,
    } = staged;
    let pointer_published = dest_body.as_ref() != Some(&source_body);
    if pointer_published {
        dest.compare_and_swap_pointer(index_key, dest_body.as_ref(), &source_body)?;
    }

    let result = TransferResult {
        index_key: index_key.to_string(),
        generation,
        artifacts_written,
        artifacts_skipped: artifacts_total - artifacts_written,
        bytes_written,
        pointer_published,
    };
    Ok((result, source_body))
}
