//! Persisted ANN cluster-index sidecar (`.tvann`).
//!
//! Stores the deterministic cluster *assignment* (per-cluster chunk-id postings)
//! so a reopen can skip the expensive k-means and recompute centroids in one pass
//! from the vectors already reconstructed from `.tvim`. Deliberately stores no
//! centroids or vector-derived data: only chunk ids and cluster membership, in
//! keeping with the payload boundary the other sidecars hold to (`.tvlex` stores
//! tokens only, `.tvtext` only opt-in text). k-means' final centroids are exactly
//! the means of the final assignment, so recomputing from the persisted postings
//! reproduces the identical clustering.
//!
//! This version is base-only: the base is (re)written whenever the TurboVec base
//! is, and O(changed) delta commits carry the base manifest forward unchanged.
//! The image is a pure acceleration cache (the exact scan is the authority), so a
//! stale, missing, or corrupt sidecar is never a correctness problem: the caller
//! validates it against the live TurboVec calibration fingerprint and the live
//! chunk set, and rebuilds on any mismatch. The base file mirrors the other JSON
//! sidecars (a checksummed body) and its manifest is embedded in the commit
//! manifest, like `tvim`.

use std::path::Path;

use serde_json::Value;

use crate::storage::util::{
    body_sha256, corrupt, get_i64, get_str, read_json, sha256_file_hex, value_object,
    verify_file_sha256, write_py_json, CoreResult,
};

/// On-disk schema version for the `.tvann` base.
pub const TVANN_INDEX_SCHEMA_VERSION: i64 = 1;

/// One index's persisted cluster assignment, loaded from a `.tvann` base.
#[derive(Debug, Clone)]
pub struct LoadedAnn {
    pub algorithm: String,
    pub dim: usize,
    pub calibration_fingerprint: u64,
    /// Sorted chunk ids per cluster. Centroids are recomputed from these plus the
    /// reconstructed vectors on load, so no vector data is persisted.
    pub postings: Vec<Vec<String>>,
}

/// The cluster assignment to persist as a `.tvann` base.
#[derive(Debug, Clone, Copy)]
pub struct AnnBaseInput<'a> {
    pub algorithm: &'a str,
    pub dim: usize,
    pub calibration_fingerprint: u64,
    pub postings: &'a [Vec<String>],
}

/// Writes the `.tvann` base and returns its manifest for the commit manifest.
pub fn record_base(base_path: &Path, ann: AnnBaseInput<'_>, fsync: bool) -> CoreResult<Value> {
    let body = serde_json::json!({
        "schema_version": TVANN_INDEX_SCHEMA_VERSION,
        "algorithm": ann.algorithm,
        "dim": ann.dim,
        "calibration_fingerprint": ann.calibration_fingerprint,
        "postings": ann.postings,
    });
    let payload = serde_json::json!({
        "schema_version": TVANN_INDEX_SCHEMA_VERSION,
        "body_sha256": body_sha256(&body)?,
        "body": body,
    });
    write_py_json(base_path, &payload, fsync)?;
    let file_bytes = base_path
        .metadata()
        .map_err(|error| corrupt(format!("tvann base metadata failed: {error}")))?
        .len();
    Ok(serde_json::json!({
        "schema_version": TVANN_INDEX_SCHEMA_VERSION,
        "base": {
            "file_name": base_path.file_name().unwrap_or_default().to_string_lossy(),
            "sha256": sha256_file_hex(base_path)?,
            "file_bytes": file_bytes,
            "cluster_count": ann.postings.len(),
            "calibration_fingerprint": ann.calibration_fingerprint,
        },
    }))
}

/// Verifies the `.tvann` base file matches its manifest checksum and schema.
pub fn validate(base_path: &Path, manifest: Option<&Value>) -> CoreResult<()> {
    let Some(manifest) = manifest else {
        return Ok(());
    };
    let manifest = value_object(manifest, "tvann index manifest")?;
    if get_i64(manifest, "schema_version", -1) != TVANN_INDEX_SCHEMA_VERSION {
        return Err(corrupt("unsupported tvann index manifest schema version"));
    }
    if let Some(base) = manifest.get("base").and_then(Value::as_object) {
        verify_file_sha256(base_path, get_str(base, "sha256"), "tvann index base")?;
    }
    Ok(())
}

/// Reads and parses the `.tvann` base into a [`LoadedAnn`], or `None` when the
/// manifest (and thus the sidecar) is absent.
pub fn load(base_path: &Path, manifest: Option<&Value>) -> CoreResult<Option<LoadedAnn>> {
    if manifest.is_none() || !base_path.is_file() {
        return Ok(None);
    }
    let payload = read_json(base_path, "tvann index base")?;
    let payload = value_object(&payload, "tvann index base")?;
    if get_i64(payload, "schema_version", -1) != TVANN_INDEX_SCHEMA_VERSION {
        return Err(corrupt("unsupported tvann index base schema version"));
    }
    let body = payload
        .get("body")
        .ok_or_else(|| corrupt("tvann index base body is missing"))?;
    if body_sha256(body)? != get_str(payload, "body_sha256") {
        return Err(corrupt("tvann index base failed checksum"));
    }
    let body = value_object(body, "tvann index base body")?;
    let algorithm = get_str(body, "algorithm").to_string();
    let dim = usize::try_from(get_i64(body, "dim", 0))
        .map_err(|_| corrupt("tvann index base has a negative dim"))?;
    let calibration_fingerprint = body
        .get("calibration_fingerprint")
        .and_then(Value::as_u64)
        .ok_or_else(|| corrupt("tvann index base is missing calibration_fingerprint"))?;
    let postings = parse_postings(body.get("postings"))?;
    Ok(Some(LoadedAnn {
        algorithm,
        dim,
        calibration_fingerprint,
        postings,
    }))
}

fn parse_postings(value: Option<&Value>) -> CoreResult<Vec<Vec<String>>> {
    let clusters = value
        .and_then(Value::as_array)
        .ok_or_else(|| corrupt("tvann index postings must be an array of clusters"))?;
    clusters
        .iter()
        .map(|cluster| {
            let ids = cluster
                .as_array()
                .ok_or_else(|| corrupt("tvann index cluster must be an array of chunk ids"))?;
            ids.iter()
                .map(|id| {
                    id.as_str()
                        .map(str::to_string)
                        .ok_or_else(|| corrupt("tvann index chunk id must be a string"))
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{load, record_base, validate, AnnBaseInput};

    fn unique_base(name: &str) -> std::path::PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!(
            "lodedb_tvann_{name}_{}_{}_{nanos}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed),
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("g0.tvann")
    }

    #[test]
    fn base_round_trips_through_record_and_load() {
        let base = unique_base("round_trip");
        let postings = vec![
            vec!["a".to_string(), "b".to_string()],
            vec!["c".to_string()],
        ];
        let manifest = record_base(
            &base,
            AnnBaseInput {
                algorithm: "cluster",
                dim: 2,
                calibration_fingerprint: 42,
                postings: &postings,
            },
            false,
        )
        .unwrap();
        validate(&base, Some(&manifest)).unwrap();
        let loaded = load(&base, Some(&manifest)).unwrap().unwrap();
        assert_eq!(loaded.algorithm, "cluster");
        assert_eq!(loaded.dim, 2);
        assert_eq!(loaded.calibration_fingerprint, 42);
        assert_eq!(loaded.postings, postings);
    }

    #[test]
    fn absent_manifest_loads_as_none() {
        let base = unique_base("absent");
        assert!(load(&base, None).unwrap().is_none());
        validate(&base, None).unwrap();
    }

    #[test]
    fn corrupt_base_fails_checksum() {
        let base = unique_base("corrupt");
        let postings = vec![vec!["a".to_string()]];
        let manifest = record_base(
            &base,
            AnnBaseInput {
                algorithm: "cluster",
                dim: 1,
                calibration_fingerprint: 7,
                postings: &postings,
            },
            false,
        )
        .unwrap();
        std::fs::write(&base, b"{\"schema_version\":1,\"body_sha256\":\"x\",\"body\":{}}").unwrap();
        assert!(validate(&base, Some(&manifest)).is_err());
    }
}
