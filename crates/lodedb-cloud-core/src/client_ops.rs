//! Client-edge operations: the six verbs a frontend invokes by target string.
//!
//! Frontends (the `orecloud` CLI via the Python binding today; `lodedb cloud` /
//! `LodeDBCloud` later) name each end of an operation with one string — a local
//! directory path or an `s3://bucket/prefix` URL. This module owns the
//! composition from those strings to the typed primitives (`export_generation`,
//! `verify_generation`, `status_for_push`, the sync classifier), so every
//! frontend shares one tested implementation and the FFI binding stays a pure
//! argument/result translator with no logic of its own.
//!
//! When the local end is a directory, transfers additionally maintain the sync
//! sidecar (`<key>.orecloud`, see [`sync_state`](crate::sync_state)): a
//! successful push or pull records the published/restored generation as the
//! new base, which is what lets [`sync`] classify later runs as fast-forwards
//! or divergences instead of blindly overwriting either end.
//!
//! Long-lived service code (e.g. a serving worker) should NOT route through
//! here: it constructs its stores once and calls the typed primitives directly,
//! rather than re-resolving target strings per operation.

use crate::artifact_store::ArtifactStore;
use crate::error::{ArtifactStoreError, Result};
use crate::generation_inventory::{
    inventory_from_body, list_index_keys, write_restored_journal_manifests,
};
use crate::manifest_transfer::{
    export_generation_pinned, export_generation_with_body, publish_staged,
    stage_generation_pinned, TransferResult,
};
use crate::snapshot_identity::{logical_id, snapshot_id};
use crate::status::{compare_generations, StatusReport};
use crate::store_target::artifact_store_from_target;
use crate::sync_plan::{classify, SnapRef, SyncClassification};
use crate::sync_state::{read_sync_state, write_sync_state, SidecarRead, SyncState};
use crate::transfer_policy::TransferPolicy;
use crate::verify::{verify_candidate_opens, verify_generation, OpenReport, VerifyReport};
use serde_json::Value;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// What a completed pull produced: the transfer itself plus the proof that the
/// restored copy opens through the engine (with its loaded counts).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PullOutcome {
    pub transfer: TransferResult,
    pub open: OpenReport,
}

/// An explicit override of the sync classification.
///
/// A force overrides the *classification* only — never the stores' integrity
/// invariants. In particular, divergent lineages that reuse an artifact file
/// name with different bytes (a same-epoch fork) still fail closed against
/// plain directory/`s3://` remotes; see [`sync`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncForce {
    /// No override: only fast-forwards run; `Diverged`/`Unknown` are errors.
    None,
    /// Push local over the remote regardless of classification (still raced
    /// through the pointer CAS, still checksum-verified).
    Push,
    /// Pull the remote over local regardless of classification (still
    /// verify-opened before the sidecar records it).
    Pull,
}

/// What one [`sync`] run decided and did.
///
/// `classification` is the pre-transfer three-pointer classification;
/// `action` is what actually ran (`"none"`, `"push"`, or `"pull"`), `forced`
/// whether a force flag overrode the classification. `transfer`/`open` are
/// present when that action moved data (`open` only for pulls).
/// `sidecar_corrupt` is a warning: a sidecar file was present but failed
/// validation, so the base was treated as absent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncOutcome {
    pub index_key: String,
    pub classification: String,
    pub action: String,
    pub forced: bool,
    pub transfer: Option<TransferResult>,
    pub open: Option<OpenReport>,
    pub sidecar_corrupt: bool,
}

/// Lists the index keys with a committed generation in the local directory
/// `dir`.
pub fn keys(dir: &str) -> Result<Vec<String>> {
    list_index_keys(Path::new(dir))
}

/// Pushes `index_key`'s committed generation from the local `dir` to `remote`
/// under `policy`.
///
/// When `dir` is a directory (not an object-store URL), a successful push also
/// records the published generation in the sync sidecar, making it the base
/// for later [`sync`]/[`status`] classification.
pub fn push(
    dir: &str,
    remote: &str,
    index_key: &str,
    policy: TransferPolicy,
) -> Result<TransferResult> {
    let source = artifact_store_from_target(dir)?;
    let dest = artifact_store_from_target(remote)?;
    let (transfer, published) = export_generation_with_body(&*source, &*dest, index_key, policy)?;
    record_base(dir, remote, index_key, &published)?;
    Ok(transfer)
}

/// Restores `index_key`'s committed generation from `remote` into the local
/// `dir`, then proves the engine can open the restored copy read-only — the
/// roadmap's "restore verifies before accepting" as one operation, so no
/// frontend can offer a pull that skips the check.
///
/// The pull ships the remote verbatim ([`TransferPolicy::full`]): a remote that
/// was pushed redacted already omits text, so there is nothing to filter on the
/// way back. A successful pull records the restored generation in the sync
/// sidecar (see [`push`]).
pub fn pull(remote: &str, dir: &str, index_key: &str) -> Result<PullOutcome> {
    if !is_local_dir(dir) {
        return Err(ArtifactStoreError::Backend(format!(
            "pull's local end must be a directory path, got {dir:?}"
        )));
    }
    let source = artifact_store_from_target(remote)?;
    let dest = artifact_store_from_target(dir)?;
    // Restores mutate the database directory, so they contend on the engine's
    // own single-writer lock: a live writer's in-memory state is based on the
    // pointer we are about to replace, and letting both proceed loses one
    // side's commit. The guard drops at the end of the restore.
    fs::create_dir_all(dir)?;
    let _writer_lock = acquire_writer_lock(dir)?;
    ensure_no_pending_wal(
        dir,
        index_key,
        "checkpoint them by opening the store once (or discard them with \
         `sync --force-pull`) before restoring over this directory",
    )?;
    let source_raw = source.read_pointer(index_key)?.ok_or_else(|| {
        ArtifactStoreError::NotFound(format!(
            "no committed generation to export for index key {index_key:?}"
        ))
    })?;
    let dest_body = dest.read_pointer(index_key)?;
    let (transfer, open, restored) = restore_staged(
        &*source,
        &*dest,
        dir,
        index_key,
        source_raw,
        dest_body,
        false,
    )?;
    record_base(dir, remote, index_key, &restored)?;
    Ok(PullOutcome { transfer, open })
}

/// The shared local-restore tail: stage the artifacts, prove the candidate
/// opens through the engine (from a scratch layout), and only then publish
/// the pointer — so every acceptance failure leaves the destination on its
/// previous committed generation. `discard_wal` truncates the destination
/// WAL right after the swap (force-pull semantics); callers must have
/// refused a pending WAL beforehand when not forcing. The journal manifests
/// are rebuilt strictly last (see the inline comment).
///
/// Known crash windows, both narrow and loud rather than corrupting: a crash
/// between the swap and the WAL truncation can leave discarded records
/// beside the new lineage (force-pull only), and a failure between the swap
/// and the journal rebuild leaves the restored copy readable but
/// fail-closed on its first O(changed) mutation. Closing them fully needs a
/// recoverable restore transaction (engine-side recovery on open) — a
/// follow-up, not this milestone.
///
/// Caller holds the directory writer lock.
fn restore_staged(
    source: &dyn ArtifactStore,
    dest: &dyn ArtifactStore,
    dir: &str,
    index_key: &str,
    source_raw: Value,
    dest_body: Option<Value>,
    discard_wal: bool,
) -> Result<(TransferResult, OpenReport, Value)> {
    let staged = stage_generation_pinned(
        source,
        dest,
        index_key,
        TransferPolicy::full(),
        source_raw,
        dest_body,
    )
    .map_err(explain_fork_collision)?;
    // Acceptance checks run against a scratch candidate BEFORE the pointer
    // moves: a checksum-consistent but semantically unopenable generation must
    // never become the committed destination. (The scratch carries its own
    // journal manifests, so the real destination stays untouched.)
    let open = verify_candidate_opens(Path::new(dir), index_key, &staged.source_body)?;
    let (transfer, restored) = publish_staged(dest, index_key, staged)?;
    if discard_wal {
        lodedb_core::storage::wal::truncate(&contained_wal_path(dir, index_key)?, false)?;
    }
    // Rebuild the engine's per-store delta-journal manifests (working state
    // the body doesn't pin): without them a restored copy opens read-only
    // fine but fails closed on its first O(changed) mutation. Strictly AFTER
    // the pointer swap: writing them first would, on a failed CAS, leave the
    // candidate's journals attached to some other writer's committed body.
    write_restored_journal_manifests(Path::new(dir), index_key, &restored)?;
    Ok((transfer, open, restored))
}

/// Takes the engine's exclusive single-writer lock on `dir` for the duration
/// of a local restore.
fn acquire_writer_lock(dir: &str) -> Result<lodedb_core::engine::DirWriterLock> {
    lodedb_core::engine::acquire_dir_writer_lock(Path::new(dir)).map_err(ArtifactStoreError::Core)
}

/// Refuses a restore when the destination WAL still holds acknowledged
/// operations: replaying them onto the pulled lineage — or silently dropping
/// them — would corrupt or lose acknowledged writes. `hint` names the caller's
/// resolution path.
pub(crate) fn ensure_no_pending_wal(dir: &str, index_key: &str, hint: &str) -> Result<()> {
    let ops = pending_wal_ops(dir, index_key)?;
    if ops > 0 {
        return Err(ArtifactStoreError::PendingWal {
            ops,
            hint: hint.to_string(),
        });
    }
    Ok(())
}

/// The WAL path for `index_key` under `dir`, confined to `dir`: the key can
/// arrive from CLI/remote input, so a `../` or absolute spelling must fail
/// closed here rather than name a WAL-shaped file outside the store root.
pub(crate) fn contained_wal_path(dir: &str, index_key: &str) -> Result<std::path::PathBuf> {
    crate::paths::resolve_within(
        Path::new(dir),
        &lodedb_core::storage::wal::wal_path(Path::new(dir), index_key),
    )
}

/// The number of valid, replayable records in `dir`'s WAL for `index_key`
/// (0 when the WAL is absent or empty). Public for the client edge: a
/// pull-direction sync consults it before downloading a single blob.
pub fn pending_wal_ops(dir: &str, index_key: &str) -> Result<usize> {
    let wal = contained_wal_path(dir, index_key)?;
    Ok(lodedb_core::storage::wal::scan_stats(&wal)?.op_count)
}

/// Compares the local `dir` against `remote` for a push of `index_key` under
/// `policy`. Read-only on both ends.
///
/// Beyond the byte-level comparison, the report carries lineage: whether a
/// sync sidecar is present, its recorded base generation, and the
/// three-pointer classification (`in_sync`/`local_ahead`/`remote_ahead`/
/// `diverged`/`republish`/`unknown`) a [`sync`] of the same arguments would
/// act on.
pub fn status(
    dir: &str,
    remote: &str,
    index_key: &str,
    policy: TransferPolicy,
) -> Result<StatusReport> {
    let source = artifact_store_from_target(dir)?;
    let dest = artifact_store_from_target(remote)?;
    // Read each pointer once and derive both the byte-level comparison and the
    // lineage classification from the same snapshot, so the two can't disagree.
    let local_body = source
        .read_pointer(index_key)?
        .map(|body| policy.redact_body(&body));
    let remote_body = dest.read_pointer(index_key)?;
    let local_inventory = inventory_from_body(index_key, local_body.as_ref())?;
    let remote_inventory = inventory_from_body(index_key, remote_body.as_ref())?;
    let mut report = compare_generations(
        index_key,
        local_inventory.as_ref(),
        remote_inventory.as_ref(),
    );

    let sidecar = read_sidecar_if_dir(dir, index_key)?;
    let base = trusted_base(&sidecar, remote);
    report.sidecar_present = base.is_some();
    report.sidecar_corrupt = sidecar.corrupt;
    report.base_generation = base.map(|base| base.generation);
    let local_ref = local_body.as_ref().map(snap_ref).transpose()?;
    let remote_ref = remote_body.as_ref().map(snap_ref).transpose()?;
    let classification = classify(local_ref.as_ref(), base, remote_ref.as_ref());
    report.classification = Some(classification.as_str().to_string());
    Ok(report)
}

/// Re-hashes every artifact `index_key`'s committed generation pins in `target`
/// (a local directory or object-store URL) against the manifest's checksums.
pub fn verify(target: &str, index_key: &str) -> Result<VerifyReport> {
    let store = artifact_store_from_target(target)?;
    verify_generation(&*store, index_key)
}

/// Synchronizes `index_key` between the local `dir` and `remote`: classifies
/// the three pointers (local under `policy`, the sidecar's recorded base, and
/// the remote), then runs at most one fast-forward transfer.
///
/// - `InSync` is a no-op; `LocalAhead`/`Republish` push; `RemoteAhead` pulls
///   (restore + verify-open, exactly like [`pull`]).
/// - `Diverged`/`Unknown` refuse with
///   [`ArtifactStoreError::SyncConflict`] — overwriting either end would
///   discard a commit, a decision only the caller can make via `force`.
/// - A forced push/pull skips the refusal but nothing else: the transfer still
///   goes through the pointer CAS and (for pulls) the verify-open. Force also
///   cannot override the stores' immutability invariant: two lineages that
///   diverged from one base and reuse the same artifact file names with
///   different bytes (a same-epoch fork) fail closed with a recovery hint —
///   resolving that shape against a dumb remote needs the managed
///   content-addressed layout of a later milestone.
///
/// Transfers are conditional on the exact remote/local state that was
/// classified: a concurrent advance of the destination between classification
/// and publish surfaces as a `PointerConflict` (retry the sync). How strong
/// that guarantee is depends on the destination backend: object stores use a
/// genuinely atomic conditional write, while a directory's pointer swap is
/// read-check-then-replace ([`LocalArtifactStore`](crate::LocalArtifactStore)
/// documents this) — so running sync concurrently with an *active engine
/// writer* on the same directory is out of contract, exactly as LodeDB's
/// single-writer model already requires. Any successful transfer records the
/// new base in the sidecar.
pub fn sync(
    dir: &str,
    remote: &str,
    index_key: &str,
    policy: TransferPolicy,
    force: SyncForce,
) -> Result<SyncOutcome> {
    if !is_local_dir(dir) {
        return Err(ArtifactStoreError::Backend(format!(
            "sync's local end must be a directory path, got {dir:?}"
        )));
    }
    let local_store = artifact_store_from_target(dir)?;
    let remote_store = artifact_store_from_target(remote)?;

    let sidecar = read_sync_state(Path::new(dir), index_key)?;
    let base = trusted_base(&sidecar, remote);
    // Classification sees the local generation as `policy` would publish it
    // (redacted), but the raw body is kept too: a pull's pointer CAS must
    // precondition on what the local pointer actually holds, not on the
    // redacted view.
    let local_raw = local_store.read_pointer(index_key)?;
    let local_body = local_raw.as_ref().map(|body| policy.redact_body(body));
    let remote_body = remote_store.read_pointer(index_key)?;
    let local_ref = local_body.as_ref().map(snap_ref).transpose()?;
    let remote_ref = remote_body.as_ref().map(snap_ref).transpose()?;
    let classification = classify(local_ref.as_ref(), base, remote_ref.as_ref());
    let had_trusted_base = base.is_some();

    let outcome = |action: &str,
                   forced: bool,
                   transfer: Option<TransferResult>,
                   open: Option<OpenReport>| SyncOutcome {
        index_key: index_key.to_string(),
        classification: classification.as_str().to_string(),
        action: action.to_string(),
        forced,
        transfer,
        open,
        sidecar_corrupt: sidecar.corrupt,
    };

    let (do_push, forced) = match force {
        SyncForce::Push => (true, true),
        SyncForce::Pull => (false, true),
        SyncForce::None => match classification {
            SyncClassification::InSync => {
                // No transfer — but if the two ends agree while the recorded
                // base is missing (fresh clone of an already-synced pair, a
                // remote switch to an identical mirror) or stale (a prior
                // transfer whose sidecar write failed), record the agreed
                // state so later runs classify as fast-forwards.
                let base_is_current = had_trusted_base
                    && base.map(|base| &base.snapshot_id)
                        == remote_ref.as_ref().map(|r| &r.snapshot_id);
                if !base_is_current {
                    if let Some(remote_body) = &remote_body {
                        record_base(dir, remote, index_key, remote_body)?;
                    }
                }
                return Ok(outcome("none", false, None, None));
            }
            SyncClassification::LocalAhead | SyncClassification::Republish => (true, false),
            SyncClassification::RemoteAhead => (false, false),
            SyncClassification::Diverged | SyncClassification::Unknown => {
                // The refusal carries the corruption diagnosis: a corrupt
                // sidecar is the likeliest reason a previously-synced pair
                // reads as unknown, and the CLI's warning path never runs on
                // an error.
                let mut hint = "re-run with --force-push to keep the local copy or --force-pull \
                                to keep the remote copy"
                    .to_string();
                if sidecar.corrupt {
                    hint.push_str(
                        " (note: the sync sidecar was present but corrupt and was ignored, so \
                         the recorded base could not be trusted)",
                    );
                }
                return Err(ArtifactStoreError::SyncConflict {
                    classification: classification.as_str().to_string(),
                    hint,
                });
            }
        },
    };

    if do_push {
        // Pinned to the exact pair classified above: the source body ships
        // verbatim (a concurrent local commit cannot swap in an unclassified
        // snapshot) and the CAS preconditions on the classified remote body
        // (a concurrent remote advance fails as a PointerConflict).
        let source_raw = local_raw.ok_or_else(|| {
            ArtifactStoreError::NotFound(format!(
                "no committed generation to push for index key {index_key:?}"
            ))
        })?;
        let (transfer, published) = export_generation_pinned(
            &*local_store,
            &*remote_store,
            index_key,
            policy,
            source_raw,
            remote_body,
        )
        .map_err(explain_fork_collision)?;
        record_base(dir, remote, index_key, &published)?;
        Ok(outcome("push", forced, Some(transfer), None))
    } else {
        // Pull mirrors `pull`: take the writer lock (a live engine writer and
        // a restore must never interleave), refuse or discard a pending WAL,
        // stage + candidate-verify + publish, then record the base. The CAS
        // preconditions on the raw local body read above — a concurrent
        // engine commit fails as a PointerConflict rather than being
        // clobbered.
        let source_raw = remote_body.ok_or_else(|| {
            ArtifactStoreError::NotFound(format!(
                "no committed generation to pull for index key {index_key:?}"
            ))
        })?;
        fs::create_dir_all(dir)?;
        let _writer_lock = acquire_writer_lock(dir)?;
        let discard_wal = forced;
        if !discard_wal {
            ensure_no_pending_wal(
                dir,
                index_key,
                "checkpoint them by opening the store once, or re-run with --force-pull \
                 to discard them along with the local lineage",
            )?;
        }
        let (transfer, open, restored) = restore_staged(
            &*remote_store,
            &*local_store,
            dir,
            index_key,
            source_raw,
            local_raw,
            discard_wal,
        )?;
        record_base(dir, remote, index_key, &restored)?;
        Ok(outcome("pull", forced, Some(transfer), Some(open)))
    }
}

/// Whether a target string names a local directory (no URL scheme).
fn is_local_dir(target: &str) -> bool {
    !target.contains("://")
}

/// Adds recovery guidance when a sync transfer hits the fork-collision limit.
///
/// Two lineages that diverged from one base commonly reuse the same engine
/// artifact names (`<key>.gen/g<epoch>.*`) with different bytes; the store's
/// immutability invariant then refuses the overwrite — deliberately, and force
/// flags do not override it (dumb path/`s3://` remotes store artifacts by
/// engine name until the managed content-addressed layout, which absorbs this,
/// arrives in a later milestone). Matching on the store's message text is
/// acceptable here because both sites live in this crate and the immutability
/// tests pin the wording.
pub(crate) fn explain_fork_collision(error: ArtifactStoreError) -> ArtifactStoreError {
    match error {
        ArtifactStoreError::Integrity(message)
            if message.contains("refusing to overwrite an immutable artifact") =>
        {
            ArtifactStoreError::Integrity(format!(
                "{message}. The two ends hold divergent lineages that reuse the same artifact \
                 file name — a fork this version cannot resolve against a plain \
                 directory/s3 remote, even with a force flag. Recover by pulling the side you \
                 want to keep into a fresh directory, or pushing it to a fresh remote prefix."
            ))
        }
        other => other,
    }
}

/// Builds the identity for a committed body: the id pair, the generation, and
/// a per-store identity for each payload store (a store is carried iff its
/// sub-manifest is non-null — the same test the engine uses to load it; the
/// identity is a digest of that sub-manifest, which names every artifact and
/// checksum the store pins).
pub(crate) fn snap_ref(body: &Value) -> Result<SnapRef> {
    let store_id = |store: &str| {
        body.get(store)
            .filter(|value| !value.is_null())
            .map(|value| crate::digest::sha256_hex(&serde_json::to_vec(value).unwrap_or_default()))
    };
    Ok(SnapRef {
        snapshot_id: snapshot_id(body)?,
        logical_id: logical_id(body)?,
        generation: body.get("generation").and_then(Value::as_u64).unwrap_or(0),
        text_id: store_id("tvtext"),
        lexical_id: store_id("tvlex"),
    })
}

/// A remote target's canonical identity for sidecar trust comparison.
///
/// Local directory targets normalize to an absolute path (resolving symlinks
/// and `.`/`..` like the store's own containment checks do), because a
/// relative spelling is cwd-dependent — the same string can name unrelated
/// directories from different working directories, and trusting on the raw
/// string would let one remote inherit another's history.
///
/// `s3://` targets additionally carry the effective endpoint: the URL alone
/// names a backend only through `AWS_ENDPOINT` (MinIO/R2 use the same
/// `s3://bucket/prefix` spelling against different services), so the endpoint
/// is part of the identity. Credentials and region are deliberately not —
/// rotating keys or fixing a region points at the *same* bucket. A
/// normalization failure or identity mismatch can only produce a false
/// *mismatch* — failing toward force, never toward a wrong fast-forward.
fn remote_identity(remote: &str) -> String {
    if is_local_dir(remote) {
        return crate::paths::canonical_identity(Path::new(remote));
    }
    let url = remote.trim_end_matches('/');
    if remote.starts_with("s3://") {
        if let Ok(endpoint) = std::env::var("AWS_ENDPOINT") {
            if !endpoint.is_empty() {
                return format!("{url}#endpoint={}", endpoint.trim_end_matches('/'));
            }
        }
    }
    url.to_string()
}

/// The sidecar base, but only when it was recorded against `remote` (compared
/// by [`remote_identity`]).
///
/// The base is a claim about what *this pair of ends* last agreed on; trusting
/// it against a different remote would classify an unrelated remote's content
/// as a fast-forward using another remote's history.
fn trusted_base<'a>(sidecar: &'a SidecarRead, remote: &str) -> Option<&'a SnapRef> {
    let identity = remote_identity(remote);
    sidecar
        .state
        .as_ref()
        .filter(|state| state.remote == identity)
        .map(|state| &state.base)
}

/// Reads the sidecar when the target is a directory; a URL target has none.
fn read_sidecar_if_dir(dir: &str, index_key: &str) -> Result<SidecarRead> {
    if is_local_dir(dir) {
        read_sync_state(Path::new(dir), index_key)
    } else {
        Ok(SidecarRead {
            state: None,
            corrupt: false,
        })
    }
}

/// Records `body` as the sidecar base after a successful transfer, when the
/// local end is a directory. URL-addressed local ends carry no sidecar.
fn record_base(dir: &str, remote: &str, index_key: &str, body: &Value) -> Result<()> {
    if !is_local_dir(dir) {
        return Ok(());
    }
    let state = SyncState {
        index_key: index_key.to_string(),
        remote: remote_identity(remote),
        base: snap_ref(body)?,
        updated_unix: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|elapsed| elapsed.as_secs())
            .unwrap_or(0),
    };
    write_sync_state(Path::new(dir), &state)
}

#[cfg(test)]
mod tests {
    use super::remote_identity;

    /// The same `s3://` spelling under different `AWS_ENDPOINT`s names
    /// different backends (MinIO/R2), so the endpoint is part of the
    /// identity. Env mutation is process-global; nothing else in this test
    /// binary reads `AWS_ENDPOINT`.
    #[test]
    fn s3_identity_includes_the_effective_endpoint() {
        std::env::remove_var("AWS_ENDPOINT");
        let plain = remote_identity("s3://bucket/prefix/");
        assert_eq!(plain, "s3://bucket/prefix");

        std::env::set_var("AWS_ENDPOINT", "http://minio.local:9000");
        let minio = remote_identity("s3://bucket/prefix");
        std::env::remove_var("AWS_ENDPOINT");

        assert_ne!(plain, minio);
        assert!(minio.contains("minio.local"));
        // Back to the default environment, the identity is stable again.
        assert_eq!(remote_identity("s3://bucket/prefix"), plain);
    }
}
