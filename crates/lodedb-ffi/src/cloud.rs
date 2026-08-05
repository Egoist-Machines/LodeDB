//! C ABI for the managed-cloud staging and verification surface.
//!
//! This module only translates JSON requests and results. HTTP, credentials,
//! and downloading content-addressed blobs belong to the caller; manifest
//! interpretation, hashing, verification, materialization, and classification
//! remain in `lodedb-cloud-core`.

use lodedb_cloud_core::generation_inventory::ArtifactRef;
use lodedb_cloud_core::{
    managed, ArtifactStoreError, ManagedPlan, ManagedSide, OpenReport, StatusReport,
    TransferPolicy, TransferResult,
};
use lodedb_core::{CoreError, CoreErrorCode};
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::{
    ffi_result, read_json_view, require_out, write_owned_json, LodeError, LodeOwnedString,
    LodeStringView,
};

#[derive(Deserialize)]
struct ManagedPlanReq {
    dir: String,
    index_key: String,
    remote_id: String,
    remote_body: Option<String>,
    include_text: bool,
    include_lexical: bool,
}

#[derive(Deserialize)]
struct PullRequirementsReq {
    dir: String,
    index_key: String,
    body: String,
}

#[derive(Deserialize)]
struct MaterializeReq {
    dir: String,
    index_key: String,
    remote_id: String,
    body: String,
    staging_dir: String,
    discard_pending_wal: bool,
    expected_local_snapshot_id: Option<String>,
}

#[derive(Deserialize)]
struct RecordBaseReq {
    dir: String,
    index_key: String,
    remote_id: String,
    body: String,
}

fn policy(include_text: bool, include_lexical: bool) -> TransferPolicy {
    TransferPolicy {
        include_text,
        include_lexical,
    }
}

fn parse_body(label: &str, body_json: &str) -> Result<Value, CoreError> {
    serde_json::from_str(body_json).map_err(|error| {
        CoreError::new(
            CoreErrorCode::InvalidArgument,
            format!("{label} is not valid JSON: {error}"),
        )
    })
}

/// Maps the transfer core's stable failure categories to the existing ABI
/// status codes. Refusals are `PLAN_STALE`, with structured JSON in the error
/// message so a caller can branch without matching prose.
fn cloud_err(error: ArtifactStoreError) -> CoreError {
    match error {
        ArtifactStoreError::NotFound(message) => CoreError::new(CoreErrorCode::NotFound, message),
        ArtifactStoreError::Integrity(message) => {
            CoreError::new(CoreErrorCode::CorruptStore, message)
        }
        // Core errors keep their own category: writer-lock contention must reach
        // Swift as invalid-argument, not a corruption diagnosis.
        ArtifactStoreError::Core(error) => error,
        ArtifactStoreError::SyncConflict {
            classification,
            hint,
        } => plan_stale_error(&classification, &format!("sync refused: {hint}")),
        ArtifactStoreError::PendingWal { ops, hint } => plan_stale_error(
            "pending_wal",
            &format!("destination has {ops} pending WAL operation(s): {hint}"),
        ),
        ArtifactStoreError::PointerConflict {
            key,
            expected,
            found,
        } => plan_stale_error(
            "stale",
            &format!(
                "pointer {key:?} changed during materialization: expected generation {expected:?}, found {found:?}"
            ),
        ),
        ArtifactStoreError::Io(error) => CoreError::new(CoreErrorCode::Internal, error.to_string()),
        ArtifactStoreError::Backend(message) => CoreError::new(CoreErrorCode::Internal, message),
    }
}

fn plan_stale_error(classification: &str, message: &str) -> CoreError {
    let payload = json!({
        "classification": classification,
        "message": message,
    });
    let payload = serde_json::to_string(&payload)
        .unwrap_or_else(|_| "{\"classification\":\"stale\"}".to_string());
    CoreError::new(CoreErrorCode::PlanStale, payload)
}

fn artifact_json(artifact: &ArtifactRef) -> Value {
    json!({
        "name": artifact.name,
        "sha256": artifact.sha256,
        "size_bytes": artifact.size_bytes,
        "kind": artifact.kind,
        "epoch": artifact.epoch,
        "is_base": artifact.is_base,
    })
}

fn side_json(side: &ManagedSide) -> Value {
    json!({
        "snapshot_id": side.snapshot_id,
        "logical_id": side.logical_id,
        "generation": side.generation,
        "has_text": side.has_text,
        "has_lexical": side.has_lexical,
    })
}

fn status_json(report: &StatusReport) -> Map<String, Value> {
    let mut value = Map::new();
    value.insert("index_key".to_string(), json!(report.index_key));
    value.insert(
        "local_generation".to_string(),
        json!(report.local_generation),
    );
    value.insert(
        "remote_generation".to_string(),
        json!(report.remote_generation),
    );
    value.insert(
        "local_document_count".to_string(),
        json!(report.local_document_count),
    );
    value.insert(
        "remote_document_count".to_string(),
        json!(report.remote_document_count),
    );
    value.insert(
        "local_chunk_count".to_string(),
        json!(report.local_chunk_count),
    );
    value.insert(
        "remote_chunk_count".to_string(),
        json!(report.remote_chunk_count),
    );
    value.insert(
        "artifacts_to_upload".to_string(),
        json!(report.artifacts_to_upload),
    );
    value.insert("bytes_to_upload".to_string(), json!(report.bytes_to_upload));
    value.insert("ships_base".to_string(), json!(report.ships_base));
    value.insert("in_sync".to_string(), json!(report.in_sync));
    value.insert("sidecar_present".to_string(), json!(report.sidecar_present));
    value.insert("sidecar_corrupt".to_string(), json!(report.sidecar_corrupt));
    value.insert("base_generation".to_string(), json!(report.base_generation));
    value.insert("classification".to_string(), json!(report.classification));
    value
}

fn plan_json(plan: &ManagedPlan) -> Result<Value, CoreError> {
    let mut value = status_json(&plan.report);
    let local = match &plan.local {
        Some(local) => {
            let mut rendered = side_json(&local.side).as_object().cloned().ok_or_else(|| {
                CoreError::new(
                    CoreErrorCode::Internal,
                    "managed side did not serialize to JSON",
                )
            })?;
            let body_json = serde_json::to_string(&local.body).map_err(|error| {
                CoreError::new(
                    CoreErrorCode::Internal,
                    format!("failed to serialize local managed body: {error}"),
                )
            })?;
            rendered.insert(
                "legacy_redacted_id".to_string(),
                json!(local.legacy_redacted_id),
            );
            rendered.insert("body_json".to_string(), json!(body_json));
            rendered.insert(
                "pointer_document".to_string(),
                json!(local.pointer_document),
            );
            rendered.insert(
                "artifacts".to_string(),
                Value::Array(local.artifacts.iter().map(artifact_json).collect()),
            );
            Value::Object(rendered)
        }
        None => Value::Null,
    };
    value.insert("local".to_string(), local);
    value.insert(
        "remote".to_string(),
        plan.remote.as_ref().map_or(Value::Null, side_json),
    );
    value.insert(
        "base".to_string(),
        plan.base.as_ref().map_or(Value::Null, |base| {
            json!({
                "snapshot_id": base.snapshot_id,
                "logical_id": base.logical_id,
                "generation": base.generation,
            })
        }),
    );
    value.insert("base_is_current".to_string(), json!(plan.base_is_current));
    value.insert(
        "local_raw_snapshot_id".to_string(),
        json!(plan.local_raw_snapshot_id),
    );
    Ok(Value::Object(value))
}

fn transfer_json(transfer: &TransferResult) -> Value {
    json!({
        "index_key": transfer.index_key,
        "generation": transfer.generation,
        "artifacts_written": transfer.artifacts_written,
        "artifacts_skipped": transfer.artifacts_skipped,
        "bytes_written": transfer.bytes_written,
        "pointer_published": transfer.pointer_published,
    })
}

fn open_json(open: &OpenReport) -> Value {
    json!({
        "document_count": open.document_count,
        "chunk_count": open.chunk_count,
    })
}

/// Builds a managed plan from one JSON request.
///
/// # Safety
/// `request`, `out`, and `error` must be valid for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn lodedb_cloud_managed_plan_json(
    request: LodeStringView,
    out: *mut *mut LodeOwnedString,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        require_out(out)?;
        let req = read_json_view::<ManagedPlanReq>(request)?;
        let remote_body = req
            .remote_body
            .as_deref()
            .map(|body| parse_body("remote_body", body))
            .transpose()?;
        let transfer_policy = policy(req.include_text, req.include_lexical);
        let plan = managed::managed_plan(
            &req.dir,
            &req.index_key,
            &req.remote_id,
            remote_body,
            transfer_policy,
        )
        .map_err(cloud_err)?;
        // Parity with the Python push wrapper: a text-bearing graph topology cannot
        // be redacted, so a text-excluding plan must refuse instead of handing the
        // caller an artifact list that would upload raw text without consent.
        if let Some(local) = &plan.local {
            transfer_policy
                .refuse_unredactable(&local.body)
                .map_err(cloud_err)?;
        }
        write_owned_json(out, &plan_json(&plan)?)
    })
}

/// Lists the content-addressed blobs a managed pull must download.
///
/// # Safety
/// `request`, `out`, and `error` must be valid for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn lodedb_cloud_managed_pull_requirements_json(
    request: LodeStringView,
    out: *mut *mut LodeOwnedString,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        require_out(out)?;
        let req = read_json_view::<PullRequirementsReq>(request)?;
        let body = parse_body("body", &req.body)?;
        let artifacts = managed::managed_pull_requirements(&req.dir, &req.index_key, &body)
            .map_err(cloud_err)?;
        write_owned_json(
            out,
            &json!({
                "artifacts": artifacts.iter().map(artifact_json).collect::<Vec<_>>(),
            }),
        )
    })
}

/// Verifies and materializes staged managed-pull blobs into a local store.
///
/// # Safety
/// `request`, `out`, and `error` must be valid for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn lodedb_cloud_managed_materialize_json(
    request: LodeStringView,
    out: *mut *mut LodeOwnedString,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        require_out(out)?;
        let req = read_json_view::<MaterializeReq>(request)?;
        let body = parse_body("body", &req.body)?;
        let outcome = managed::managed_materialize(
            &req.dir,
            &req.index_key,
            &req.remote_id,
            body,
            &req.staging_dir,
            req.discard_pending_wal,
            req.expected_local_snapshot_id.as_deref(),
        )
        .map_err(cloud_err)?;
        write_owned_json(
            out,
            &json!({
                "transfer": transfer_json(&outcome.transfer),
                "open": open_json(&outcome.open),
            }),
        )
    })
}

/// Records a managed remote body as the trusted sidecar base.
///
/// # Safety
/// `request`, `out`, and `error` must be valid for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn lodedb_cloud_managed_record_base_json(
    request: LodeStringView,
    out: *mut *mut LodeOwnedString,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        require_out(out)?;
        let req = read_json_view::<RecordBaseReq>(request)?;
        let body = parse_body("body", &req.body)?;
        managed::managed_record_base(&req.dir, &req.index_key, &req.remote_id, &body)
            .map_err(cloud_err)?;
        write_owned_json(out, &json!({}))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::c_char;
    use std::ptr;

    fn view(text: &str) -> LodeStringView {
        LodeStringView {
            size: std::mem::size_of::<LodeStringView>() as u32,
            version: crate::ABI_VERSION,
            data: text.as_ptr().cast::<c_char>(),
            len: text.len(),
        }
    }

    #[test]
    fn managed_plan_rejects_a_null_output_pointer() {
        let mut error = ptr::null_mut();
        let status = unsafe {
            lodedb_cloud_managed_plan_json(
                view(
                    r#"{"dir":"/tmp","index_key":"idx","remote_id":"orecloud://test","include_text":false,"include_lexical":false}"#,
                ),
                ptr::null_mut(),
                &mut error,
            )
        };
        assert_eq!(status, CoreErrorCode::InvalidArgument.ffi_status_code());
        assert!(!error.is_null());
        unsafe { crate::lodedb_error_free(error) };
    }

    #[test]
    fn managed_requirements_rejects_bad_json() {
        let mut out = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status =
            unsafe { lodedb_cloud_managed_pull_requirements_json(view("{"), &mut out, &mut error) };
        assert_eq!(status, CoreErrorCode::InvalidArgument.ffi_status_code());
        assert!(out.is_null());
        assert!(!error.is_null());
        unsafe { crate::lodedb_error_free(error) };
    }
}
