//! Client-side composition for the managed (`orecloud://`) transfer plane.
//!
//! Managed transfers split the client edge in two: the Python layer speaks
//! HTTP to the control plane (begin/commit push sessions, pull plans,
//! presigned or proxied blob bytes), while everything that touches the commit
//! format stays here: inventories, canonical identities, the engine-written
//! pointer document, sidecar trust, classification, and the restore path with
//! its verify-open. The seam is deliberate: Python never parses a manifest or
//! re-serialises a body, so the canonical-JSON contract has exactly one
//! implementation (the engine's), exactly as the dumb-target verbs already
//! guarantee.
//!
//! The remote's committed state arrives as the head *body* the control plane
//! returned (stored and served as JSON). Identities recomputed from that body
//! are stable across the trip because the engine's canonical form is
//! key-order-insensitive: `serde_json::Value` objects are sorted maps, so a
//! body that passed through the catalog re-canonicalises to the same digest.
//! The end-to-end tests pin this.
//!
//! Unlike dumb targets, the remote identity string used for sidecar trust is
//! supplied by the caller verbatim (`orecloud://org/environment/db#host=…`): the
//! Python layer owns the spelling of managed remotes, and the sidecar records
//! and compares exactly that string.

use crate::client_ops::{ensure_no_pending_wal, snap_ref};
use crate::error::{ArtifactStoreError, Result};
use crate::generation_inventory::{
    diff_inventories, inventory_from_body, write_restored_journal_manifests, ArtifactRef,
};
use crate::local_artifact_store::LocalArtifactStore;
use crate::manifest_transfer::{publish_staged, stage_generation_pinned, TransferResult};
use crate::snapshot_identity::{identity_from_document, pointer_document};
use crate::status::{compare_generations, StatusReport};
use crate::sync_plan::{classify, SnapRef};
use crate::sync_state::{read_sync_state, write_sync_state, SyncState};
use crate::transfer_policy::TransferPolicy;
use crate::verify::{verify_candidate_opens, OpenReport};
use crate::ArtifactStore;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// One side's identity and payload masks, as the wire contract speaks them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManagedSide {
    pub snapshot_id: String,
    pub logical_id: String,
    pub generation: u64,
    pub has_text: bool,
    pub has_lexical: bool,
}

/// The local half of a push: identity, the exact body/pointer document a push
/// would publish, and the full artifact set backing it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManagedLocal {
    pub side: ManagedSide,
    /// The policy-redacted committed body, what a push publishes.
    pub body: Value,
    /// The engine-canonical pointer document for `body` (UTF-8 JSON bytes,
    /// carried as a string). Shipped verbatim at commit so the control plane
    /// can maintain the object-store pointer mirror without re-serialising.
    pub pointer_document: String,
    /// Every artifact the body pins (the begin-push inventory).
    pub artifacts: Vec<ArtifactRef>,
}

/// Everything the Python edge needs to run one managed status/push/sync
/// decision: the byte-level status report (lineage fields filled), both ends'
/// identities, and the trusted base.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManagedPlan {
    pub report: StatusReport,
    pub local: Option<ManagedLocal>,
    pub remote: Option<ManagedSide>,
    pub base: Option<SnapRef>,
    /// Whether the trusted base already records the remote's current snapshot.
    /// When false after an `in_sync` classification, the caller should
    /// refresh the sidecar (mirrors `client_ops::sync`'s stale-base repair).
    pub base_is_current: bool,
    /// The RAW (unredacted) local pointer's snapshot id at classification
    /// time; `None` when the directory holds no committed generation. A
    /// pull-direction materialization pins on this so a local commit landing
    /// between classification and materialization refuses instead of being
    /// silently overwritten. This is the managed twin of the dumb sync carrying
    /// its classified `local_raw` into the pointer CAS.
    pub local_raw_snapshot_id: Option<String>,
}

/// What one managed pull produced (mirrors [`crate::PullOutcome`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManagedPullOutcome {
    pub transfer: TransferResult,
    pub open: OpenReport,
}

fn require_local_dir(dir: &str) -> Result<()> {
    if dir.contains("://") {
        return Err(ArtifactStoreError::Backend(format!(
            "a managed transfer's local end must be a directory path, got {dir:?}"
        )));
    }
    Ok(())
}

fn managed_side(body: &Value, reference: &SnapRef) -> ManagedSide {
    ManagedSide {
        snapshot_id: reference.snapshot_id.clone(),
        logical_id: reference.logical_id.clone(),
        generation: reference.generation,
        has_text: body.get("tvtext").is_some_and(|value| !value.is_null()),
        has_lexical: body.get("tvlex").is_some_and(|value| !value.is_null()),
    }
}

/// Builds the full decision context for one managed index: reads the local
/// committed generation (redacted by `policy`), computes identities and the
/// pointer document, inventories both ends, reads the sidecar (trusted only
/// when recorded against exactly `remote_id`), and classifies.
///
/// `remote_body` is the branch-head body the control plane returned (`None`
/// when the remote holds nothing). Pure local I/O, no network.
pub fn managed_plan(
    dir: &str,
    index_key: &str,
    remote_id: &str,
    remote_body: Option<Value>,
    policy: TransferPolicy,
) -> Result<ManagedPlan> {
    require_local_dir(dir)?;
    let local_store = LocalArtifactStore::new(dir, false);

    let local_raw = local_store.read_pointer(index_key)?;
    let local_raw_snapshot_id = local_raw
        .as_ref()
        .map(crate::snapshot_identity::snapshot_id)
        .transpose()?;
    let local_body = local_raw.map(|body| policy.redact_body(&body));
    let local_inventory = inventory_from_body(index_key, local_body.as_ref())?;
    let remote_inventory = inventory_from_body(index_key, remote_body.as_ref())?;
    let mut report = compare_generations(
        index_key,
        local_inventory.as_ref(),
        remote_inventory.as_ref(),
    );

    let sidecar = read_sync_state(Path::new(dir), index_key)?;
    let base = sidecar
        .state
        .as_ref()
        .filter(|state| state.remote == remote_id)
        .map(|state| state.base.clone());

    let local_ref = local_body.as_ref().map(snap_ref).transpose()?;
    let remote_ref = remote_body.as_ref().map(snap_ref).transpose()?;
    let classification = classify(local_ref.as_ref(), base.as_ref(), remote_ref.as_ref());

    report.sidecar_present = base.is_some();
    report.sidecar_corrupt = sidecar.corrupt;
    report.base_generation = base.as_ref().map(|base| base.generation);
    report.classification = Some(classification.as_str().to_string());

    let base_is_current = match (&base, &remote_ref) {
        (Some(base), Some(remote)) => base.snapshot_id == remote.snapshot_id,
        _ => false,
    };

    let local = match (local_body, local_ref, local_inventory) {
        (Some(body), Some(reference), Some(inventory)) => {
            let document = pointer_document(&body)?;
            // Cross-check the document against the recomputed identity: both
            // come from the same engine writer, so a mismatch means the body
            // changed between the two reads. Fail closed rather than begin a
            // push whose commit will contradict itself.
            let document_id = identity_from_document(&document)?;
            if document_id != reference.snapshot_id {
                return Err(ArtifactStoreError::Integrity(format!(
                    "pointer document identity {document_id} does not match the snapshot id \
                     {} computed from the same body",
                    reference.snapshot_id
                )));
            }
            let document = String::from_utf8(document).map_err(|error| {
                ArtifactStoreError::Integrity(format!(
                    "engine-written pointer document is not UTF-8: {error}"
                ))
            })?;
            Some(ManagedLocal {
                side: managed_side(&body, &reference),
                body,
                pointer_document: document,
                artifacts: inventory.artifacts,
            })
        }
        _ => None,
    };
    let remote = match (&remote_body, &remote_ref) {
        (Some(body), Some(reference)) => Some(managed_side(body, reference)),
        _ => None,
    };

    Ok(ManagedPlan {
        report,
        local,
        remote,
        base,
        base_is_current,
        local_raw_snapshot_id,
    })
}

/// Records `body` as the sidecar base for `remote_id` after a successful
/// managed transfer. `remote_id` is stored verbatim; the caller owns the
/// canonical spelling of managed remote identities.
pub fn managed_record_base(
    dir: &str,
    index_key: &str,
    remote_id: &str,
    body: &Value,
) -> Result<()> {
    require_local_dir(dir)?;
    let state = SyncState {
        index_key: index_key.to_string(),
        remote: remote_id.to_string(),
        base: snap_ref(body)?,
        updated_unix: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|elapsed| elapsed.as_secs())
            .unwrap_or(0),
    };
    write_sync_state(Path::new(dir), &state)
}

/// The artifacts a pull of `body` must download: everything the remote body
/// pins that the local directory does not already hold byte-identically.
pub fn managed_pull_requirements(
    dir: &str,
    index_key: &str,
    body: &Value,
) -> Result<Vec<ArtifactRef>> {
    require_local_dir(dir)?;
    let local_store = LocalArtifactStore::new(dir, false);
    let remote_inventory = inventory_from_body(index_key, Some(body))?
        .expect("inventory is Some when the body is Some");
    let local_body = local_store.read_pointer(index_key)?;
    let local_inventory = inventory_from_body(index_key, local_body.as_ref())?;
    Ok(diff_inventories(&remote_inventory, local_inventory.as_ref()).to_upload)
}

/// A read-only [`ArtifactStore`] over a staging directory of downloaded
/// blobs, addressed by content: `read_bytes(name)` resolves the engine
/// artifact name to its sha256 via the pull body's inventory and reads
/// `<staging>/<sha256>`. Write operations are unreachable by construction
/// (it is only ever a transfer *source*) and fail closed if reached.
struct StagingBlobStore {
    root: std::path::PathBuf,
    body: Value,
    sha_by_name: HashMap<String, String>,
}

impl StagingBlobStore {
    fn new(staging_dir: &Path, index_key: &str, body: Value) -> Result<Self> {
        let inventory = inventory_from_body(index_key, Some(&body))?
            .expect("inventory is Some when the body is Some");
        let sha_by_name = inventory
            .artifacts
            .into_iter()
            .map(|artifact| (artifact.name, artifact.sha256))
            .collect();
        Ok(Self {
            root: staging_dir.to_path_buf(),
            body,
            sha_by_name,
        })
    }

    fn staged_path(&self, name: &str) -> Result<std::path::PathBuf> {
        let sha = self.sha_by_name.get(name).ok_or_else(|| {
            ArtifactStoreError::NotFound(format!(
                "artifact {name:?} is not referenced by the pull body"
            ))
        })?;
        // The inventory already validated every digest as 64 lowercase hex
        // characters; re-assert here (defense in depth) so a digest can never
        // traverse out of the staging root even if a future inventory change
        // loosens that guarantee.
        crate::blob_layout::validate_sha256(sha)?;
        Ok(self.root.join(sha))
    }
}

impl ArtifactStore for StagingBlobStore {
    fn open_read<'a>(&'a self, name: &str) -> Result<Box<dyn std::io::Read + 'a>> {
        let path = self.staged_path(name)?;
        match std::fs::File::open(&path) {
            Ok(handle) => Ok(Box::new(handle)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                Err(ArtifactStoreError::NotFound(format!(
                    "blob for artifact {name:?} was not downloaded into the staging directory"
                )))
            }
            Err(error) => Err(ArtifactStoreError::Io(error)),
        }
    }

    fn write_stream_if_absent(
        &self,
        name: &str,
        _data: &mut dyn std::io::Read,
        _sha256: &str,
        _size_hint: u64,
    ) -> Result<()> {
        Err(ArtifactStoreError::Backend(format!(
            "staging blob store is read-only (attempted write of {name:?})"
        )))
    }

    fn contains(&self, name: &str) -> Result<bool> {
        Ok(self.staged_path(name)?.try_exists()?)
    }

    fn read_pointer(&self, _key: &str) -> Result<Option<Value>> {
        Ok(Some(self.body.clone()))
    }

    fn compare_and_swap_pointer(
        &self,
        _key: &str,
        _old_body: Option<&Value>,
        _new_body: &Value,
    ) -> Result<()> {
        Err(ArtifactStoreError::Backend(
            "staging blob store is read-only (attempted pointer swap)".to_string(),
        ))
    }
}

/// Materialises a managed pull: restores `body`'s generation into `dir` from
/// content-addressed blobs previously downloaded into `staging_dir`
/// (`<staging_dir>/<sha256>` per blob), proves the restored copy opens
/// through the engine BEFORE the pointer moves, and records the sidecar base
/// against `remote_id`.
///
/// The restore runs under the engine's single-writer lock (a live writer and
/// a restore must never interleave) and refuses when the destination WAL
/// still holds acknowledged operations. Pass `discard_pending_wal` (the
/// force-pull semantics) to truncate them along with the local lineage. Every
/// blob is re-hashed against the manifest checksum on write (a corrupt
/// download fails the pull before any pointer moves), the candidate is
/// verify-opened from a scratch layout before publication, and the local
/// pointer swap preconditions on the committed body read at the start (a
/// concurrent engine commit surfaces as a `PointerConflict` instead of being
/// clobbered).
pub fn managed_materialize(
    dir: &str,
    index_key: &str,
    remote_id: &str,
    body: Value,
    staging_dir: &str,
    discard_pending_wal: bool,
    expected_local_snapshot_id: Option<&str>,
) -> Result<ManagedPullOutcome> {
    require_local_dir(dir)?;
    let staging = StagingBlobStore::new(Path::new(staging_dir), index_key, body.clone())?;
    let local_store = LocalArtifactStore::new(dir, false);
    fs::create_dir_all(dir)?;
    let _writer_lock = lodedb_core::engine::acquire_dir_writer_lock(Path::new(dir))
        .map_err(ArtifactStoreError::Core)?;
    if !discard_pending_wal {
        ensure_no_pending_wal(
            dir,
            index_key,
            "checkpoint them by opening the store once, or re-run the sync with \
             --force-pull to discard them along with the local lineage",
        )?;
    }
    let local_raw = local_store.read_pointer(index_key)?;
    // Pin the materialization to the local state the caller CLASSIFIED, not
    // the state found now: a commit landing between classification and this
    // lock acquisition must refuse (re-run the sync to re-classify) rather
    // than be silently overwritten. `None` skips the pin (a plain pull's
    // decision IS the state read under this lock); the empty string pins to
    // "classified as absent" (a fresh clone), any other value to that exact
    // raw snapshot id.
    if let Some(expected) = expected_local_snapshot_id {
        let current = local_raw
            .as_ref()
            .map(crate::snapshot_identity::snapshot_id)
            .transpose()?;
        let unchanged = if expected.is_empty() {
            current.is_none()
        } else {
            current.as_deref() == Some(expected)
        };
        if !unchanged {
            return Err(ArtifactStoreError::SyncConflict {
                classification: "stale".to_string(),
                hint: "the local store committed a new generation after this sync \
                       classified it; re-run the sync to reconcile"
                    .to_string(),
            });
        }
    }
    // The managed remote absorbs same-name forks (blobs are content-
    // addressed there), but the LOCAL directory still stores artifacts by
    // engine name: force-pulling a diverged lineage that reuses a name with
    // different bytes fails closed on the immutability invariant, exactly
    // like the dumb-target verbs. Recover by pulling into a fresh
    // directory. The wrapper attaches that recovery hint.
    let staged = stage_generation_pinned(
        &staging,
        &local_store,
        index_key,
        TransferPolicy::full(),
        body,
        local_raw,
    )
    .map_err(crate::client_ops::explain_fork_collision)?;
    // Acceptance checks run against a scratch candidate BEFORE the pointer
    // moves (the scratch carries its own journal manifests, so the real
    // destination stays untouched until the swap).
    let open = verify_candidate_opens(Path::new(dir), index_key, &staged.source_body)?;
    let (transfer, restored) = publish_staged(&local_store, index_key, staged)?;
    if discard_pending_wal {
        lodedb_core::storage::wal::truncate(
            &crate::client_ops::contained_wal_path(dir, index_key)?,
            false,
        )?;
    }
    // Rebuild the journal manifests (working state the body doesn't pin) so
    // the restored copy is writable, not just readable; the cloud writer
    // opens hydrated copies through this path. Strictly AFTER the swap:
    // writing them first would, on a failed CAS, leave the candidate's
    // journals attached to some other writer's committed body.
    write_restored_journal_manifests(Path::new(dir), index_key, &restored)?;
    managed_record_base(dir, index_key, remote_id, &restored)?;
    Ok(ManagedPullOutcome { transfer, open })
}
