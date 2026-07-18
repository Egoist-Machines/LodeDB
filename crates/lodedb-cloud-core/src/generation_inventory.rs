//! Read-only inventory of the artifacts a committed generation references.
//!
//! Given a committed root manifest, this enumerates every artifact it pins —
//! each per-store base plus its delta segments — as [`ArtifactRef`] records
//! carrying name, checksum, size, kind, base epoch, and base-vs-delta. It reads
//! only the committed root (via `read_commit_manifest`); it never stats the
//! files, reads the `.wal` tail, or reads the on-disk per-store `manifest.json`.
//! So the inventory describes exactly one consistent committed generation.
//!
//! [`diff_inventories`] compares a local inventory against a remote one and
//! reports which artifacts are missing remotely, distinguishing "delta segments
//! onto an existing base epoch" from "a whole new base epoch" — the signal a
//! push needs to stay O(changed).

use crate::error::{ArtifactStoreError, Result};
use crate::paths::resolve_within;
use lodedb_core::storage::commit_manifest::{
    commit_manifest_path, read_commit_manifest, COMMIT_MANIFEST_SUFFIX,
};
use lodedb_core::storage::lexical_store::LEXICAL_INDEX_DELTA_DIR_SUFFIX;
use lodedb_core::storage::multivec_store::MULTIVEC_DELTA_DIR_SUFFIX;
use lodedb_core::storage::state_journal::STATE_JOURNAL_DIR_SUFFIX;
use lodedb_core::storage::text_store::DOCUMENT_TEXT_DELTA_DIR_SUFFIX;
use lodedb_core::storage::tvim_delta::TVIM_DELTA_DIR_SUFFIX;
use lodedb_core::storage::tvvf_store::TVVF_DELTA_DIR_SUFFIX;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// The generation directory suffix (`<key>.gen`), mirroring
/// `commit_manifest::generation_dir`. Artifact names are store-relative and
/// begin with this directory, so we build the relative prefix here rather than
/// deriving an absolute path.
const GEN_DIR_SUFFIX: &str = ".gen";

/// Each per-store sub-manifest key in the root body, paired with the directory
/// suffix its delta segments live under (the delta dir is the base file name +
/// suffix, e.g. `g7.json` -> `g7.json.json-delta/`). The key doubles as the base
/// file extension, since the engine derives base paths as `g<epoch>.<kind>`. A
/// store is inventoried iff its sub-manifest is non-null; `tvmv` (multi-vector /
/// late-interaction) is included — omitting it would silently drop those artifacts
/// from a backup. `tvann` (the persisted ANN cluster partition) is base-only —
/// the engine treats a missing/corrupt `.tvann` as a cache miss and rebuilds —
/// but it still ships: a body referencing a base we never uploaded would fail
/// byte-verification on pull, and shipping it saves the restored/hydrated copy
/// a corpus-sized re-cluster. Its delta suffix follows the engine's uniform
/// `<base>.<kind>-delta` derivation (no deltas are ever recorded today).
const STORE_KINDS: &[(&str, &str)] = &[
    ("json", STATE_JOURNAL_DIR_SUFFIX),
    ("tvim", TVIM_DELTA_DIR_SUFFIX),
    ("tvtext", DOCUMENT_TEXT_DELTA_DIR_SUFFIX),
    ("tvlex", LEXICAL_INDEX_DELTA_DIR_SUFFIX),
    ("tvmv", MULTIVEC_DELTA_DIR_SUFFIX),
    ("tvann", ".tvann-delta"),
    // `tvvf` (the rescore original-vector sidecar, engine 1.3.2+) is a journaled
    // {base, deltas} store like tvim: vector payload, never text-gated, and the
    // engine refuses to open a rescore store without it — so it ships always.
    ("tvvf", TVVF_DELTA_DIR_SUFFIX),
];

/// One artifact (a base or a delta segment) referenced by a committed generation.
///
/// `name` is the store-relative path (e.g. `idx.gen/g7.json`); `kind` is the
/// owning store (`json`/`tvim`/`tvtext`/`tvlex`/`tvmv`/`tvann`/`tvvf`); `epoch` is the base
/// epoch the artifact lives under; `is_base` is true for the base snapshot and
/// false for a delta segment appended onto it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactRef {
    pub name: String,
    pub sha256: String,
    pub size_bytes: u64,
    pub kind: String,
    pub epoch: u64,
    pub is_base: bool,
}

/// The full set of artifacts pinned by one committed generation.
///
/// `root_body` is the committed manifest body verbatim, suitable as the payload
/// for a destination's `compare_and_swap_pointer` when shipping this generation
/// elsewhere.
#[derive(Debug, Clone)]
pub struct GenerationInventory {
    pub index_key: String,
    pub generation: u64,
    pub base_epoch: u64,
    pub document_count: u64,
    pub chunk_count: u64,
    pub root_body: Value,
    pub artifacts: Vec<ArtifactRef>,
}

/// What a local generation has that a remote one does not.
///
/// `ships_base` is true when the transfer must upload a base artifact — a new
/// base epoch (cold build or compaction) *or* a replacement base under an epoch
/// the remote already holds at a different checksum (two divergent lineages at
/// the same epoch number). It is false only when every uploaded artifact is a
/// delta segment onto a base the remote already holds byte-for-byte (the
/// O(changed) common case). Callers choosing delta-only vs full-base handling
/// must branch on this, not on the epoch number alone.
#[derive(Debug, Clone)]
pub struct InventoryDiff {
    pub to_upload: Vec<ArtifactRef>,
    pub ships_base: bool,
    pub upload_bytes: u64,
}

/// Builds an inventory from an already-read commit-manifest body.
///
/// Returns `None` when `body` is `None` (no committed generation). This is the
/// store-agnostic core: a caller holding a body from any [`ArtifactStore`] (not
/// just the local filesystem) builds the inventory the same way.
///
/// [`ArtifactStore`]: crate::ArtifactStore
pub fn inventory_from_body(
    index_key: &str,
    body: Option<&Value>,
) -> Result<Option<GenerationInventory>> {
    let Some(body) = body else {
        return Ok(None);
    };
    // The pointer's file-name key must match the body's own `index_key`, and the
    // engine additionally rejects an empty/missing `index_key` as corrupt. Requiring
    // exact equality covers both: a mismatch would inventory one directory while
    // `load_store` reads another, and an empty key would publish a body the engine
    // refuses to open. Only reachable via a tampered/corrupt pointer (the body
    // checksum is already validated), so fail closed.
    let body_key = body
        .get("index_key")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if body_key != index_key {
        return Err(ArtifactStoreError::Integrity(format!(
            "commit manifest body index_key {body_key:?} does not match requested key \
             {index_key:?}"
        )));
    }
    let base_epoch = body_u64(body, "base_epoch");
    // Fail closed on a store this table does not know: a future engine that
    // adds a sub-manifest (as `tvann` was added) must not have its artifacts
    // silently dropped from a backup — an inventory that understates what the
    // body pins ships a generation whose blobs were never uploaded. Store
    // sub-manifests are recognizable by shape (an object carrying a journaled
    // `base`), which no scalar body field (`generation`, `native_dim`, …) has.
    if let Some(object) = body.as_object() {
        for (key, value) in object {
            let looks_like_store = value.is_object() && value.get("base").is_some();
            if looks_like_store && !STORE_KINDS.iter().any(|(kind, _)| kind == key) {
                return Err(ArtifactStoreError::Integrity(format!(
                    "commit body carries an unknown store sub-manifest {key:?}; this \
                     OreCloud build does not know how to inventory it — upgrade OreCloud \
                     before transferring this generation"
                )));
            }
        }
    }
    let mut artifacts: Vec<ArtifactRef> = Vec::new();
    // Inventory each store whose sub-manifest is non-null, mirroring exactly how the
    // engine decides to load a store (`store_manifest(kind).is_some()`). tvim
    // included: the engine reads a non-null tvim manifest regardless of the body's
    // informational `tvim_present` flag, so gating on that flag could drop a base
    // the restored generation needs.
    for (kind, dir_suffix) in STORE_KINDS {
        artifacts.extend(refs_for_store(
            index_key,
            kind,
            body.get(*kind),
            dir_suffix,
            base_epoch,
        )?);
    }
    Ok(Some(GenerationInventory {
        index_key: index_key.to_string(),
        generation: body_u64(body, "generation"),
        base_epoch,
        document_count: body_u64(body, "document_count"),
        chunk_count: body_u64(body, "chunk_count"),
        root_body: body.clone(),
        artifacts,
    }))
}

/// Reads the live committed root for `index_key` and inventories it.
///
/// Returns `None` when no `<key>.commit.json` is present (an uncommitted or
/// legacy-only store). Read-only: it touches only the committed root pointer.
/// The key is confined to `persistence_dir` via [`resolve_within`], so an
/// `index_key` from CLI/remote input cannot escape the store root.
pub fn inventory_committed_generation(
    persistence_dir: &Path,
    index_key: &str,
) -> Result<Option<GenerationInventory>> {
    let pointer = resolve_within(
        persistence_dir,
        &commit_manifest_path(persistence_dir, index_key),
    )?;
    let body = read_commit_manifest(&pointer)?.map(|manifest| manifest.body);
    inventory_from_body(index_key, body.as_ref())
}

/// Lists the index keys with a committed root manifest under `persistence_dir`.
///
/// Lets callers enumerate what to back up without reaching into engine
/// internals. Returns a sorted list; empty when the directory is absent.
pub fn list_index_keys(persistence_dir: &Path) -> Result<Vec<String>> {
    if !persistence_dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut keys = Vec::new();
    for entry in fs::read_dir(persistence_dir)? {
        let entry = entry?;
        if let Some(name) = entry.file_name().to_str() {
            if let Some(key) = name.strip_suffix(COMMIT_MANIFEST_SUFFIX) {
                keys.push(key.to_string());
            }
        }
    }
    keys.sort();
    Ok(keys)
}

/// Returns the artifacts present in `local` but missing/differing in `remote`.
///
/// An artifact is "present" remotely only when the remote names it with the same
/// sha256, so a divergent checksum is treated as needing upload. `remote` is
/// `None` when the destination holds no committed generation yet.
pub fn diff_inventories(
    local: &GenerationInventory,
    remote: Option<&GenerationInventory>,
) -> InventoryDiff {
    let remote_by_name: HashMap<&str, &str> = remote
        .map(|inventory| {
            inventory
                .artifacts
                .iter()
                .map(|artifact| (artifact.name.as_str(), artifact.sha256.as_str()))
                .collect()
        })
        .unwrap_or_default();
    let to_upload: Vec<ArtifactRef> = local
        .artifacts
        .iter()
        .filter(|artifact| {
            remote_by_name.get(artifact.name.as_str()).copied() != Some(artifact.sha256.as_str())
        })
        .cloned()
        .collect();
    // Ship a "full base" whenever a base artifact is in the upload set: a missing
    // remote, a different base epoch, or the same epoch number whose base differs
    // by checksum (divergent lineages) all surface here as a base ref to upload.
    let ships_base = remote.is_none() || to_upload.iter().any(|artifact| artifact.is_base);
    let upload_bytes = to_upload.iter().map(|artifact| artifact.size_bytes).sum();
    InventoryDiff {
        to_upload,
        ships_base,
        upload_bytes,
    }
}

/// Expands one per-store sub-manifest into its base + delta artifact refs.
///
/// Pulls sha256 and byte size straight from the manifest entries (no file stat).
/// A sub-manifest that is absent or empty yields nothing; one carrying the legacy
/// pre-journal marker (`present` with no journaled `base`) fails closed rather
/// than silently omit its artifacts from a backup — rewrite the generation to
/// migrate it to the journaled layout first.
fn refs_for_store(
    index_key: &str,
    kind: &str,
    sub_manifest: Option<&Value>,
    dir_suffix: &str,
    base_epoch: u64,
) -> Result<Vec<ArtifactRef>> {
    // Distinguish an absent store (missing key or explicit null -> no artifacts)
    // from a present-but-malformed one. The engine's `store_manifest` treats any
    // non-null value as present, so a non-null, non-object manifest is corrupt and
    // must fail closed rather than be silently skipped — skipping would publish a
    // body the engine then refuses to open.
    let sub_manifest = match sub_manifest {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(value) => value.as_object().ok_or_else(|| {
            ArtifactStoreError::Integrity(format!(
                "{kind} sub-manifest is non-null but not a JSON object"
            ))
        })?,
    };
    let Some(base) = sub_manifest.get("base").and_then(Value::as_object) else {
        if truthy_map(sub_manifest, "present") {
            return Err(ArtifactStoreError::Integrity(format!(
                "{kind} sub-manifest uses the legacy pre-journal layout, which the generation \
                 inventory does not support; rewrite the generation first"
            )));
        }
        // A non-null sub-manifest object must carry a journaled base: the engine
        // reads `<key>.gen/g<base_epoch>.<kind>` when opening this store. Treating a
        // base-less object as "no artifacts" would silently drop that base from a
        // backup and leave the restored generation unopenable. An absent store is
        // null (handled above), never a base-less object, so fail closed here.
        return Err(ArtifactStoreError::Integrity(format!(
            "{kind} sub-manifest is a non-null object without a base; expected a journaled \
             {{base, deltas}} manifest"
        )));
    };

    let gen_dir = format!("{index_key}{GEN_DIR_SUFFIX}");
    // The engine derives the base path from base_epoch + store kind
    // (`base_json_path` etc. yield `g<epoch>.<kind>`), NOT from the recorded
    // file_name. Use the derived name as authoritative so the inventory names
    // exactly the file the engine will open, and reject a manifest whose recorded
    // base file_name disagrees (a tampered or inconsistent pointer) rather than
    // copying the wrong file under a name the engine cannot find.
    let base_name = format!("g{base_epoch}.{kind}");
    let recorded = str_field(base, "file_name");
    if !recorded.is_empty() && recorded != base_name {
        return Err(ArtifactStoreError::Integrity(format!(
            "{kind} base file_name {recorded:?} does not match the engine-derived name \
             {base_name:?}"
        )));
    }
    let mut refs = vec![ArtifactRef {
        name: format!("{gen_dir}/{base_name}"),
        sha256: str_field(base, "sha256"),
        size_bytes: u64_field(base, "file_bytes"),
        kind: kind.to_string(),
        epoch: base_epoch,
        is_base: true,
    }];

    let delta_dir = format!("{base_name}{dir_suffix}");
    if let Some(deltas) = sub_manifest.get("deltas").and_then(Value::as_array) {
        for delta in deltas {
            let Some(delta) = delta.as_object() else {
                // The engine only ever records object delta entries and rejects
                // anything else on replay; skipping a malformed entry would build
                // an inventory that disagrees with what the engine will load, so
                // fail closed instead.
                return Err(ArtifactStoreError::Integrity(format!(
                    "{kind} sub-manifest has a non-object delta entry"
                )));
            };
            let delta_name = str_field(delta, "file_name");
            ensure_plain_file_name(kind, &delta_name)?;
            refs.push(ArtifactRef {
                name: format!("{gen_dir}/{delta_dir}/{delta_name}"),
                sha256: str_field(delta, "sha256"),
                size_bytes: u64_field(delta, "file_bytes"),
                kind: kind.to_string(),
                epoch: base_epoch,
                is_base: false,
            });
        }
    }
    Ok(refs)
}

/// Reconstructs each journaled store's on-disk delta-journal `manifest.json`
/// after a restore.
///
/// The engine writes one journal manifest per store (in
/// `<base>.<kind>-delta/manifest.json`) whose content IS the store's
/// sub-manifest in the commit body — but the journal file itself is engine
/// working state, not an artifact the body pins, so transfers never ship it.
/// The read path doesn't need it (it reads the committed root), but the
/// engine's O(changed) mutation path appends journal deltas through it and
/// fails closed when it is missing — which would make a restored directory
/// readable but not writable. Restores call this to rebuild every journal
/// manifest verbatim from the body, so a pulled/hydrated copy behaves
/// exactly like an engine-authored one. `tvann` is excluded: the ANN sidecar
/// is base-only and the engine keeps no journal for it.
pub(crate) fn write_restored_journal_manifests(
    persistence_dir: &Path,
    index_key: &str,
    body: &Value,
) -> Result<()> {
    let base_epoch = body_u64(body, "base_epoch");
    for (kind, dir_suffix) in STORE_KINDS {
        if *kind == "tvann" {
            continue;
        }
        let Some(sub_manifest) = body.get(*kind).filter(|value| !value.is_null()) else {
            continue;
        };
        let base_name = format!("g{base_epoch}.{kind}");
        let journal_dir = persistence_dir
            .join(format!("{index_key}{GEN_DIR_SUFFIX}"))
            .join(format!("{base_name}{dir_suffix}"));
        fs::create_dir_all(&journal_dir)?;
        let path = journal_dir.join("manifest.json");
        let rendered = serde_json::to_string_pretty(sub_manifest).map_err(|error| {
            ArtifactStoreError::Integrity(format!(
                "{kind} sub-manifest failed to serialize for the journal manifest: {error}"
            ))
        })?;
        // Atomic replace, like every other pointer/manifest write here.
        let scratch = journal_dir.join(".manifest.json.tmp");
        fs::write(&scratch, rendered.as_bytes())?;
        fs::rename(&scratch, &path)?;
    }
    Ok(())
}


/// Rejects a manifest `file_name` that is not a plain path component.
///
/// Engine artifact names are always single components (`g7.json`,
/// `delta-00000000.jsd`). A name containing a path separator or a `.`/`..` segment
/// would, when joined onto `<key>.gen/`, point outside that directory — still
/// inside the store root (`resolve_within` confines it there), but under another
/// index's key — letting a tampered source manifest plant a file during restore.
/// Fails closed; every legitimate manifest passes.
fn ensure_plain_file_name(kind: &str, file_name: &str) -> Result<()> {
    let is_unsafe = file_name.is_empty()
        || file_name.contains('/')
        || file_name.contains('\\')
        || file_name == "."
        || file_name == "..";
    if is_unsafe {
        return Err(ArtifactStoreError::Integrity(format!(
            "{kind} sub-manifest has an unsafe artifact file name {file_name:?}; \
             expected a plain file name"
        )));
    }
    Ok(())
}

/// Reads an unsigned integer field from a manifest body, defaulting to 0 for a
/// missing/non-integer value (matching the engine's own manifest accessors).
fn body_u64(body: &Value, key: &str) -> u64 {
    body.get(key).and_then(Value::as_u64).unwrap_or(0)
}

/// Whether a sub-manifest object flag is present and true.
fn truthy_map(object: &Map<String, Value>, key: &str) -> bool {
    object.get(key).and_then(Value::as_bool).unwrap_or(false)
}

/// Reads a string field from a manifest object, defaulting to empty.
fn str_field(object: &Map<String, Value>, key: &str) -> String {
    object
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// Reads an unsigned integer field from a manifest object, defaulting to 0.
fn u64_field(object: &Map<String, Value>, key: &str) -> u64 {
    object.get(key).and_then(Value::as_u64).unwrap_or(0)
}
