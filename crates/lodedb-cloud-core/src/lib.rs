//! Native Rust core for OreCloud.
//!
//! OreCloud's durable transfer/backup operates on LodeDB's open, generation-
//! addressed commit format. Rather than reimplement (or import the hollowing-out
//! Python implementation of) that format, this crate links `lodedb-core`'s
//! `storage` module directly, so the cloud and the embedded engine share ONE
//! commit-format implementation: schema, on-disk layout, canonical-JSON
//! checksum, and every sub-manifest (`json`/`tvim`/`tvtext`/`tvlex`/`tvmv`/`tvann`/`tvvf`).
//!
//! It provides an [`ArtifactStore`] abstraction with a filesystem default
//! ([`LocalArtifactStore`]), a read-only generation inventory
//! ([`inventory_committed_generation`]/[`diff_inventories`]), committed-generation
//! transfer that ships only committed state ([`export_generation`], filtered by an
//! explicit [`TransferPolicy`]; a restore is the same call with the stores
//! swapped), plus
//! [`verify_generation`] and [`compare_generations`] for integrity checks and
//! push status. This is the Rust *data plane*; auth/catalog/control-plane concerns
//! live in a separate (non-Rust) service.

pub mod artifact_store;
pub mod blob_layout;
pub mod client_ops;
mod digest;
pub mod error;
pub mod generation_inventory;
pub mod local_artifact_store;
pub mod managed;
pub mod manifest_transfer;
pub mod object_artifact_store;
mod paths;
pub mod snapshot_identity;
pub mod status;
pub mod store_target;
pub mod sync_plan;
pub mod sync_state;
pub mod transfer_policy;
pub mod verify;

pub use artifact_store::ArtifactStore;
pub use blob_layout::{blob_name, parse_blob_name};
pub use client_ops::{PullOutcome, SyncForce, SyncOutcome};
pub use error::{ArtifactStoreError, Result};
pub use generation_inventory::{
    diff_inventories, inventory_committed_generation, inventory_from_body, list_index_keys,
    ArtifactRef, GenerationInventory, InventoryDiff,
};
pub use local_artifact_store::LocalArtifactStore;
pub use managed::{
    managed_materialize, managed_plan, managed_pull_requirements, managed_record_base,
    ManagedLocal, ManagedPlan, ManagedPullOutcome, ManagedSide,
};
pub use manifest_transfer::{export_generation, TransferResult};
pub use object_artifact_store::ObjectArtifactStore;
pub use snapshot_identity::{logical_id, snapshot_id};
pub use status::{compare_generations, status_for_push, StatusReport};
pub use store_target::artifact_store_from_target;
pub use sync_plan::{classify, SnapRef, SyncClassification};
pub use sync_state::{read_sync_state, write_sync_state, SidecarRead, SyncState};
pub use transfer_policy::TransferPolicy;
pub use verify::{verify_generation, verify_local_generation_opens, OpenReport, VerifyReport};

/// Returns the commit-manifest schema version understood by the linked
/// `lodedb-core`.
///
/// OreCloud pins to this value and fails closed on a mismatch: a store written
/// by a newer engine schema must never be silently mis-transferred. Reading it
/// straight from the linked core (rather than hard-coding `1`) means the pin
/// tracks the exact engine OreCloud builds against.
pub fn linked_core_schema_version() -> i64 {
    lodedb_core::storage::commit_manifest::COMMIT_MANIFEST_SCHEMA_VERSION
}

#[cfg(test)]
mod tests {
    use super::linked_core_schema_version;

    /// Proves the git dependency on `lodedb-core` resolves, compiles, and that
    /// the engine's public commit-format surface is reachable from OreCloud.
    #[test]
    fn links_lodedb_core_commit_manifest() {
        assert_eq!(linked_core_schema_version(), 1);
    }
}
