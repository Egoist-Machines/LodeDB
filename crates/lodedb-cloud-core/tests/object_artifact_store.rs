//! Tests for `ObjectArtifactStore` over `object_store`'s in-memory backend:
//! immutable byte I/O, conditional-write pointer CAS, per-tenant isolation, and a
//! full export/restore round trip through an "object store".

mod common;

use common::*;
use lodedb_core::storage::{
    write_generation_commit, GenerationCommitInput, GenerationWriteOptions,
};
use object_store::memory::InMemory;
use object_store::ObjectStore;
use lodedb_cloud_core::{
    export_generation, verify_generation, verify_local_generation_opens, ArtifactStore,
    ArtifactStoreError, LocalArtifactStore, ObjectArtifactStore, TransferPolicy,
};
use serde_json::{json, Value};
use std::sync::Arc;

/// A base-only json sub-manifest for a pointer body (no artifacts need exist for
/// pointer-CAS tests).
fn json_sub(base_name: &str) -> Value {
    json!({
        "base": { "file_name": base_name, "sha256": sha_hex(b"x"), "file_bytes": 0 },
        "deltas": [],
    })
}

#[test]
fn write_and_read_round_trips() {
    let backend: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let store = ObjectArtifactStore::new(backend, "tenant").unwrap();
    let data = b"generation-artifact-bytes";
    store
        .write_bytes_if_absent("idx.gen/g0.json", data, &sha_hex(data))
        .unwrap();
    assert_eq!(store.read_bytes("idx.gen/g0.json").unwrap(), data.to_vec());
}

#[test]
fn read_missing_artifact_is_not_found() {
    let backend: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let store = ObjectArtifactStore::new(backend, "tenant").unwrap();
    let err = store.read_bytes("idx.gen/absent.json").unwrap_err();
    assert!(matches!(err, ArtifactStoreError::NotFound(_)));
}

#[test]
fn checksum_mismatch_is_rejected() {
    let backend: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let store = ObjectArtifactStore::new(backend, "tenant").unwrap();
    let err = store
        .write_bytes_if_absent("idx.gen/g0.json", b"data", &sha_hex(b"other"))
        .unwrap_err();
    assert!(matches!(err, ArtifactStoreError::Integrity(_)));
}

#[test]
fn immutable_rewrite_is_noop_or_conflict() {
    let backend: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let store = ObjectArtifactStore::new(backend, "tenant").unwrap();
    let data = b"immutable";
    store
        .write_bytes_if_absent("idx.gen/g0.json", data, &sha_hex(data))
        .unwrap();
    // Identical bytes: idempotent no-op.
    store
        .write_bytes_if_absent("idx.gen/g0.json", data, &sha_hex(data))
        .unwrap();
    // Different bytes under the same name: refused, original preserved.
    let err = store
        .write_bytes_if_absent("idx.gen/g0.json", b"different", &sha_hex(b"different"))
        .unwrap_err();
    assert!(matches!(err, ArtifactStoreError::Integrity(_)));
    assert_eq!(store.read_bytes("idx.gen/g0.json").unwrap(), data.to_vec());
}

#[test]
fn pointer_cas_creates_updates_and_conflicts() {
    let backend: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let store = ObjectArtifactStore::new(backend, "tenant").unwrap();
    assert!(store.read_pointer("idx").unwrap().is_none());

    // Create (expect-absent) publishes generation 1.
    let gen1 = commit_body("idx", 1, 0, json_sub("g0.json"));
    store.compare_and_swap_pointer("idx", None, &gen1).unwrap();

    // Expect-absent now conflicts (the pointer exists).
    let gen2 = commit_body("idx", 2, 0, json_sub("g0.json"));
    let err = store
        .compare_and_swap_pointer("idx", None, &gen2)
        .unwrap_err();
    assert!(matches!(err, ArtifactStoreError::PointerConflict { .. }));

    // Matching the exact committed body advances the pointer to 2.
    store
        .compare_and_swap_pointer("idx", Some(&gen1), &gen2)
        .unwrap();
    assert_eq!(
        store
            .read_pointer("idx")
            .unwrap()
            .unwrap()
            .get("generation")
            .and_then(Value::as_u64),
        Some(2)
    );

    // A stale expected body (a would-be concurrent writer still holding gen 1)
    // conflicts — the pointer is now gen 2.
    let gen3 = commit_body("idx", 3, 0, json_sub("g0.json"));
    let err = store
        .compare_and_swap_pointer("idx", Some(&gen1), &gen3)
        .unwrap_err();
    assert!(matches!(err, ArtifactStoreError::PointerConflict { .. }));
}

#[test]
fn read_pointer_rejects_a_corrupt_document() {
    // A pointer object whose bytes are not a valid commit document must fail
    // closed, not parse as an empty/zero generation.
    let backend: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let store = ObjectArtifactStore::new(backend, "tenant").unwrap();
    store
        .write_bytes_if_absent("idx.commit.json", b"not-json", &sha_hex(b"not-json"))
        .unwrap();
    assert!(store.read_pointer("idx").is_err());
}

#[test]
fn tenants_are_isolated_by_prefix() {
    // Two stores share one bucket under different tenant prefixes.
    let backend: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let tenant_a = ObjectArtifactStore::new(backend.clone(), "tenant-a").unwrap();
    let tenant_b = ObjectArtifactStore::new(backend, "tenant-b").unwrap();

    let data = b"tenant-a-secret";
    tenant_a
        .write_bytes_if_absent("idx.gen/g0.json", data, &sha_hex(data))
        .unwrap();
    tenant_a
        .compare_and_swap_pointer("idx", None, &commit_body("idx", 1, 0, json_sub("g0.json")))
        .unwrap();

    // Tenant B, knowing the exact name and checksum, still cannot reach A's blob
    // or pointer — the prefix namespaces content addressing per tenant.
    assert!(matches!(
        tenant_b.read_bytes("idx.gen/g0.json").unwrap_err(),
        ArtifactStoreError::NotFound(_)
    ));
    assert!(tenant_b.read_pointer("idx").unwrap().is_none());
}

#[test]
fn exports_and_restores_through_an_object_store() {
    const KEY: &str = "dd44ee55ff6600112233445566778899aabbccddeeff00112233445566778899";
    let src = tempfile::tempdir().unwrap();
    write_generation_commit(
        src.path(),
        GenerationCommitInput {
            index_key: KEY,
            generation: 1,
            applied_lsn: 0,
            base_epoch: 1,
            state: &engine_state(KEY),
            tvim: None,
            raw_text: None,
            lexical_tokens: None,
            multivec: None,
            ann: None,
            tvvf_manifest: None,
            compress_text: false,
        },
        GenerationWriteOptions::default(),
    )
    .unwrap();

    // Push: local directory -> object store.
    let source = LocalArtifactStore::new(src.path(), false);
    let backend: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let remote = ObjectArtifactStore::new(backend, "tenant-x").unwrap();
    let pushed = export_generation(&source, &remote, KEY, TransferPolicy::full()).unwrap();
    assert!(pushed.pointer_published);
    assert!(pushed.artifacts_written > 0);

    // Idempotent re-push moves nothing.
    let again = export_generation(&source, &remote, KEY, TransferPolicy::full()).unwrap();
    assert_eq!(again.artifacts_written, 0);
    assert!(!again.pointer_published);

    // Every artifact in the object store re-hashes to its recorded checksum.
    let verified = verify_generation(&remote, KEY).unwrap();
    assert_eq!(verified.generation, 1);
    assert!(verified.artifacts_verified > 0);

    // Pull: object store -> a fresh local directory that opens read-only.
    let restored = tempfile::tempdir().unwrap();
    let dest = LocalArtifactStore::new(restored.path(), false);
    export_generation(&remote, &dest, KEY, TransferPolicy::full()).unwrap();
    let report = verify_local_generation_opens(restored.path(), KEY).unwrap();
    assert_eq!(report.index_key, KEY);
}

/// Deterministic pseudo-random bytes big enough to cross the multipart
/// threshold (8 MiB) with a non-chunk-aligned tail.
fn large_payload() -> Vec<u8> {
    let len = 9 * 1024 * 1024 + 12345;
    (0..len).map(|i| (i * 31 + i / 251) as u8).collect()
}

#[test]
fn large_artifacts_stream_through_multipart_byte_identically() {
    let backend: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let store = ObjectArtifactStore::new(backend, "tenant").unwrap();
    let data = large_payload();
    let digest = sha_hex(&data);

    let mut reader: &[u8] = &data;
    store
        .write_stream_if_absent("idx.gen/g0.tvim", &mut reader, &digest, 0)
        .unwrap();

    // Read back through the streaming bridge and compare byte for byte.
    let mut restored = Vec::new();
    std::io::Read::read_to_end(
        &mut *store.open_read("idx.gen/g0.tvim").unwrap(),
        &mut restored,
    )
    .unwrap();
    assert_eq!(restored, data);

    // Idempotent large re-push: same name, same bytes is a no-op.
    let mut reader: &[u8] = &data;
    store
        .write_stream_if_absent("idx.gen/g0.tvim", &mut reader, &digest, 0)
        .unwrap();

    // Same name, different large bytes refuses (immutability).
    let mut other = data.clone();
    other[0] ^= 0xff;
    let mut reader: &[u8] = &other;
    let err = store
        .write_stream_if_absent("idx.gen/g0.tvim", &mut reader, &sha_hex(&other), 0)
        .unwrap_err();
    assert!(matches!(err, ArtifactStoreError::Integrity(_)));
}

#[test]
fn large_upload_with_a_wrong_digest_stores_nothing() {
    // The digest is computed while streaming and checked before the multipart
    // completes, so a corrupt stream aborts without publishing the object.
    let backend: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let store = ObjectArtifactStore::new(backend, "tenant").unwrap();
    let data = large_payload();

    let mut reader: &[u8] = &data;
    let err = store
        .write_stream_if_absent("idx.gen/g0.tvim", &mut reader, &sha_hex(b"other"), 0)
        .unwrap_err();
    assert!(matches!(err, ArtifactStoreError::Integrity(_)));
    assert!(!store.contains("idx.gen/g0.tvim").unwrap());
}

#[test]
fn large_upload_claims_atomically_and_leaves_no_scratch_object() {
    // The multipart path streams to a unique scratch key and claims the final
    // name with a conditional copy; after success the scratch is deleted and
    // only the tenant-prefixed artifact remains.
    let backend = Arc::new(InMemory::new());
    let store = ObjectArtifactStore::new(backend.clone(), "tenant").unwrap();
    let data = large_payload();
    let mut reader: &[u8] = &data;
    store
        .write_stream_if_absent("idx.gen/g0.tvim", &mut reader, &sha_hex(&data), 0)
        .unwrap();

    use futures::TryStreamExt;
    let listed: Vec<_> =
        futures::executor::block_on(backend.list(None).try_collect::<Vec<_>>()).unwrap();
    let keys: Vec<String> = listed
        .iter()
        .map(|meta| meta.location.to_string())
        .collect();
    assert_eq!(keys, vec!["tenant/idx.gen/g0.tvim".to_string()], "{keys:?}");
}
