//! `lodedb._turbovec.cloud`: the Python face of the managed-cloud transfer plane.
//!
//! A pure translator over `lodedb_cloud_core::client_ops` — every function here
//! converts Python arguments in, calls the corresponding (Rust-tested) client
//! operation, and converts the report out as a plain dict. No logic lives at
//! this seam. The dict fields are mapped by hand, deliberately: a struct field
//! rename then fails to compile here instead of silently changing a dict key
//! under the CLI.
//!
//! Errors map onto stdlib exceptions: a missing artifact or generation raises
//! `FileNotFoundError`, a filesystem failure raises `OSError`, and every
//! integrity/conflict/backend failure raises `RuntimeError` with the library's
//! own diagnostic message.
//!
//! The GIL is released around every operation (`detach`) because a push
//! or pull can spend seconds-to-minutes in network and disk I/O.

use lodedb_cloud_core::generation_inventory::ArtifactRef;
use lodedb_cloud_core::{client_ops, managed};
use lodedb_cloud_core::{
    ArtifactStoreError, StatusReport, SyncForce, SyncOutcome, TransferPolicy, TransferResult,
    VerifyReport,
};
use pyo3::exceptions::{PyFileNotFoundError, PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

pyo3::create_exception!(
    cloud,
    SyncConflictError,
    PyRuntimeError,
    "A sync/restore refused because resolving it needs an explicit decision \
     (diverged/unknown lineage, or a destination WAL holding acknowledged \
     writes) — re-run with a force flag or checkpoint first. Subclasses \
     RuntimeError, so pre-existing handlers keep working."
);

/// Maps a library error onto the matching stdlib Python exception.
///
/// `SyncConflict` and `PendingWal` get the dedicated `SyncConflictError` (a
/// RuntimeError subclass): both are refusals the caller resolves with a force
/// flag or a checkpoint, and the CLI maps them to its "refused" exit class
/// instead of "unexpected".
fn to_py_err(error: ArtifactStoreError) -> PyErr {
    match &error {
        ArtifactStoreError::NotFound(_) => PyFileNotFoundError::new_err(error.to_string()),
        ArtifactStoreError::Io(_) => PyOSError::new_err(error.to_string()),
        ArtifactStoreError::SyncConflict { .. } | ArtifactStoreError::PendingWal { .. } => {
            SyncConflictError::new_err(error.to_string())
        }
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}

/// Builds a `TransferPolicy` from the two CLI-level opt-in flags.
fn policy(include_text: bool, include_lexical: bool) -> TransferPolicy {
    TransferPolicy {
        include_text,
        include_lexical,
    }
}

/// Renders a `TransferResult` as a dict.
fn transfer_dict<'py>(py: Python<'py>, result: &TransferResult) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("index_key", &result.index_key)?;
    dict.set_item("generation", result.generation)?;
    dict.set_item("artifacts_written", result.artifacts_written)?;
    dict.set_item("artifacts_skipped", result.artifacts_skipped)?;
    dict.set_item("bytes_written", result.bytes_written)?;
    dict.set_item("pointer_published", result.pointer_published)?;
    Ok(dict)
}

/// Renders a `StatusReport` as a dict (absent sides become `None`).
fn status_dict<'py>(py: Python<'py>, report: &StatusReport) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("index_key", &report.index_key)?;
    dict.set_item("local_generation", report.local_generation)?;
    dict.set_item("remote_generation", report.remote_generation)?;
    dict.set_item("local_document_count", report.local_document_count)?;
    dict.set_item("remote_document_count", report.remote_document_count)?;
    dict.set_item("local_chunk_count", report.local_chunk_count)?;
    dict.set_item("remote_chunk_count", report.remote_chunk_count)?;
    dict.set_item("artifacts_to_upload", report.artifacts_to_upload)?;
    dict.set_item("bytes_to_upload", report.bytes_to_upload)?;
    dict.set_item("ships_base", report.ships_base)?;
    dict.set_item("in_sync", report.in_sync)?;
    dict.set_item("sidecar_present", report.sidecar_present)?;
    dict.set_item("sidecar_corrupt", report.sidecar_corrupt)?;
    dict.set_item("base_generation", report.base_generation)?;
    dict.set_item("classification", report.classification.as_deref())?;
    Ok(dict)
}

/// Renders a `SyncOutcome` as one flat dict: the decision fields, then the
/// transfer metrics when data moved, then the opened counts for a pull.
fn sync_dict<'py>(py: Python<'py>, outcome: &SyncOutcome) -> PyResult<Bound<'py, PyDict>> {
    let dict = match &outcome.transfer {
        Some(transfer) => transfer_dict(py, transfer)?,
        None => {
            let dict = PyDict::new(py);
            dict.set_item("index_key", &outcome.index_key)?;
            dict
        }
    };
    dict.set_item("classification", &outcome.classification)?;
    dict.set_item("action", &outcome.action)?;
    dict.set_item("forced", outcome.forced)?;
    dict.set_item("sidecar_corrupt", outcome.sidecar_corrupt)?;
    if let Some(open) = &outcome.open {
        dict.set_item("document_count", open.document_count)?;
        dict.set_item("chunk_count", open.chunk_count)?;
    }
    Ok(dict)
}

/// Renders a `VerifyReport` as a dict.
fn verify_dict<'py>(py: Python<'py>, report: &VerifyReport) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("index_key", &report.index_key)?;
    dict.set_item("generation", report.generation)?;
    dict.set_item("artifacts_verified", report.artifacts_verified)?;
    dict.set_item("bytes_verified", report.bytes_verified)?;
    Ok(dict)
}

/// Lists the index keys with a committed generation in a local LodeDB directory.
#[pyfunction]
fn keys(py: Python<'_>, dir: &str) -> PyResult<Vec<String>> {
    py.detach(|| client_ops::keys(dir)).map_err(to_py_err)
}

/// Compares `dir`'s committed generation against `remote` for a push under the
/// given policy flags. Read-only on both ends.
#[pyfunction]
#[pyo3(signature = (dir, remote, key, include_text = false, include_lexical = false))]
fn status<'py>(
    py: Python<'py>,
    dir: &str,
    remote: &str,
    key: &str,
    include_text: bool,
    include_lexical: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let report = py
        .detach(|| client_ops::status(dir, remote, key, policy(include_text, include_lexical)))
        .map_err(to_py_err)?;
    status_dict(py, &report)
}

/// Pushes `dir`'s committed generation to `remote`. Redacted by default: the
/// payload-bearing text/lexical stores ship only with the explicit opt-ins.
#[pyfunction]
#[pyo3(signature = (dir, remote, key, include_text = false, include_lexical = false))]
fn push<'py>(
    py: Python<'py>,
    dir: &str,
    remote: &str,
    key: &str,
    include_text: bool,
    include_lexical: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let result = py
        .detach(|| client_ops::push(dir, remote, key, policy(include_text, include_lexical)))
        .map_err(to_py_err)?;
    transfer_dict(py, &result)
}

/// Restores `key`'s committed generation from `remote` into the local `dir`.
/// The restore verifies the copy opens through the engine before reporting
/// success (see `client_ops::pull`); the returned dict is the transfer report
/// plus the opened `document_count`/`chunk_count`.
#[pyfunction]
fn pull<'py>(py: Python<'py>, remote: &str, dir: &str, key: &str) -> PyResult<Bound<'py, PyDict>> {
    let outcome = py
        .detach(|| client_ops::pull(remote, dir, key))
        .map_err(to_py_err)?;
    let dict = transfer_dict(py, &outcome.transfer)?;
    dict.set_item("document_count", outcome.open.document_count)?;
    dict.set_item("chunk_count", outcome.open.chunk_count)?;
    Ok(dict)
}

/// Synchronizes `key` between the local `dir` and `remote`: classifies local
/// vs the recorded sidecar base vs remote, then runs at most one fast-forward
/// transfer. Diverged/unknown lineage raises `RuntimeError` unless exactly one
/// of `force_push`/`force_pull` overrides it; setting both is a `ValueError`.
#[pyfunction]
#[pyo3(signature = (
    dir, remote, key,
    include_text = false, include_lexical = false,
    force_push = false, force_pull = false
))]
#[allow(clippy::too_many_arguments)]
fn sync<'py>(
    py: Python<'py>,
    dir: &str,
    remote: &str,
    key: &str,
    include_text: bool,
    include_lexical: bool,
    force_push: bool,
    force_pull: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let force = match (force_push, force_pull) {
        (true, true) => {
            return Err(PyValueError::new_err(
                "force_push and force_pull are mutually exclusive",
            ))
        }
        (true, false) => SyncForce::Push,
        (false, true) => SyncForce::Pull,
        (false, false) => SyncForce::None,
    };
    let outcome = py
        .detach(|| {
            client_ops::sync(
                dir,
                remote,
                key,
                policy(include_text, include_lexical),
                force,
            )
        })
        .map_err(to_py_err)?;
    sync_dict(py, &outcome)
}

/// Re-hashes every artifact `key`'s committed generation pins in `target` (a
/// local directory or object-store URL) against the manifest's checksums.
#[pyfunction]
fn verify<'py>(py: Python<'py>, target: &str, key: &str) -> PyResult<Bound<'py, PyDict>> {
    let report = py
        .detach(|| client_ops::verify(target, key))
        .map_err(to_py_err)?;
    verify_dict(py, &report)
}

/// Parses a JSON body string handed across the FFI boundary.
fn parse_body(label: &str, body_json: &str) -> PyResult<serde_json::Value> {
    serde_json::from_str(body_json)
        .map_err(|error| PyValueError::new_err(format!("{label} is not valid JSON: {error}")))
}

/// Renders one inventory artifact as a dict.
fn artifact_dict<'py>(py: Python<'py>, artifact: &ArtifactRef) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("name", &artifact.name)?;
    dict.set_item("sha256", &artifact.sha256)?;
    dict.set_item("size_bytes", artifact.size_bytes)?;
    dict.set_item("kind", &artifact.kind)?;
    dict.set_item("is_base", artifact.is_base)?;
    Ok(dict)
}

/// Renders a list of inventory artifacts.
fn artifact_list<'py>(py: Python<'py>, artifacts: &[ArtifactRef]) -> PyResult<Bound<'py, PyList>> {
    let items = artifacts
        .iter()
        .map(|artifact| artifact_dict(py, artifact))
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, items)
}

/// Renders one side's identity/payload masks as a dict.
fn side_dict<'py>(py: Python<'py>, side: &managed::ManagedSide) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("snapshot_id", &side.snapshot_id)?;
    dict.set_item("logical_id", &side.logical_id)?;
    dict.set_item("generation", side.generation)?;
    dict.set_item("has_text", side.has_text)?;
    dict.set_item("has_lexical", side.has_lexical)?;
    Ok(dict)
}

/// Builds the decision context for one managed (`orecloud://`) index: local
/// identity + pointer document + artifact inventory, the remote head parsed
/// from `remote_body_json`, the sidecar base trusted against `remote_id`
/// (compared verbatim), and the three-pointer classification. Read-only.
#[pyfunction]
#[pyo3(signature = (dir, key, remote_id, remote_body_json = None, include_text = false, include_lexical = false))]
fn managed_plan<'py>(
    py: Python<'py>,
    dir: &str,
    key: &str,
    remote_id: &str,
    remote_body_json: Option<&str>,
    include_text: bool,
    include_lexical: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let remote_body = remote_body_json
        .map(|json| parse_body("remote body", json))
        .transpose()?;
    let plan = py
        .detach(|| {
            managed::managed_plan(
                dir,
                key,
                remote_id,
                remote_body,
                policy(include_text, include_lexical),
            )
        })
        .map_err(to_py_err)?;

    let dict = status_dict(py, &plan.report)?;
    match &plan.local {
        Some(local) => {
            let local_dict = side_dict(py, &local.side)?;
            let body_json = serde_json::to_string(&local.body).map_err(|error| {
                PyRuntimeError::new_err(format!("failed to serialize local body: {error}"))
            })?;
            local_dict.set_item("body_json", body_json)?;
            local_dict.set_item("pointer_document", &local.pointer_document)?;
            local_dict.set_item("artifacts", artifact_list(py, &local.artifacts)?)?;
            dict.set_item("local", local_dict)?;
        }
        None => dict.set_item("local", py.None())?,
    }
    match &plan.remote {
        Some(remote) => dict.set_item("remote", side_dict(py, remote)?)?,
        None => dict.set_item("remote", py.None())?,
    }
    match &plan.base {
        Some(base) => {
            let base_dict = PyDict::new(py);
            base_dict.set_item("snapshot_id", &base.snapshot_id)?;
            base_dict.set_item("logical_id", &base.logical_id)?;
            base_dict.set_item("generation", base.generation)?;
            dict.set_item("base", base_dict)?;
        }
        None => dict.set_item("base", py.None())?,
    }
    dict.set_item("base_is_current", plan.base_is_current)?;
    dict.set_item(
        "local_raw_snapshot_id",
        plan.local_raw_snapshot_id.as_deref(),
    )?;
    Ok(dict)
}

/// The number of valid, replayable records in `dir`'s WAL for `key` (0 when
/// the WAL is absent or empty). A pull-direction sync consults this BEFORE
/// downloading blobs; deliberately not part of `managed_plan`, so push/status
/// planning never pays an O(WAL bytes) scan.
#[pyfunction]
fn local_wal_ops(py: Python<'_>, dir: &str, key: &str) -> PyResult<usize> {
    py.detach(|| client_ops::pending_wal_ops(dir, key))
        .map_err(to_py_err)
}

/// Records `body_json` as the sidecar base for `remote_id` (stored verbatim)
/// after a successful managed transfer.
#[pyfunction]
fn managed_record_base(
    py: Python<'_>,
    dir: &str,
    key: &str,
    remote_id: &str,
    body_json: &str,
) -> PyResult<()> {
    let body = parse_body("body", body_json)?;
    py.detach(|| managed::managed_record_base(dir, key, remote_id, &body))
        .map_err(to_py_err)
}

/// The artifacts a pull of `body_json` must download into staging: everything
/// the remote body pins that `dir` does not already hold byte-identically.
#[pyfunction]
fn managed_pull_requirements<'py>(
    py: Python<'py>,
    dir: &str,
    key: &str,
    body_json: &str,
) -> PyResult<Bound<'py, PyList>> {
    let body = parse_body("body", body_json)?;
    let artifacts = py
        .detach(|| managed::managed_pull_requirements(dir, key, &body))
        .map_err(to_py_err)?;
    artifact_list(py, &artifacts)
}

/// Restores a managed pull from `staging_dir` (holding `<sha256>` blob files)
/// into `dir`, verifies the candidate opens through the engine BEFORE the
/// pointer moves, and records the sidecar base against `remote_id`. Runs
/// under the engine's single-writer lock; refuses when the destination WAL
/// holds acknowledged operations unless `discard_pending_wal` (the
/// force-pull semantics) truncates them. `expected_local_snapshot_id` pins
/// the local state the caller classified (`""` = classified as absent;
/// `None` = no pin, the plain-pull semantics): a local commit landing after
/// classification refuses with `SyncConflictError` instead of being
/// overwritten. Returns the transfer report plus the opened document/chunk
/// counts.
#[pyfunction]
#[pyo3(signature = (
    dir, key, remote_id, body_json, staging_dir,
    discard_pending_wal = false, expected_local_snapshot_id = None
))]
#[allow(clippy::too_many_arguments)]
fn managed_materialize<'py>(
    py: Python<'py>,
    dir: &str,
    key: &str,
    remote_id: &str,
    body_json: &str,
    staging_dir: &str,
    discard_pending_wal: bool,
    expected_local_snapshot_id: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let body = parse_body("body", body_json)?;
    let outcome = py
        .detach(|| {
            managed::managed_materialize(
                dir,
                key,
                remote_id,
                body,
                staging_dir,
                discard_pending_wal,
                expected_local_snapshot_id,
            )
        })
        .map_err(to_py_err)?;
    let dict = transfer_dict(py, &outcome.transfer)?;
    dict.set_item("document_count", outcome.open.document_count)?;
    dict.set_item("chunk_count", outcome.open.chunk_count)?;
    Ok(dict)
}

/// Registers the transfer plane as the `cloud` submodule of `_turbovec`.
///
/// Reached as an attribute (`from lodedb._turbovec import cloud` /
/// `_turbovec.cloud.push(...)`), never as a dotted import — extension
/// submodules are not importable through the module finder.
pub(crate) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = PyModule::new(parent.py(), "cloud")?;
    module.add(
        "SyncConflictError",
        parent.py().get_type::<SyncConflictError>(),
    )?;
    module.add_function(wrap_pyfunction!(keys, &module)?)?;
    module.add_function(wrap_pyfunction!(status, &module)?)?;
    module.add_function(wrap_pyfunction!(push, &module)?)?;
    module.add_function(wrap_pyfunction!(pull, &module)?)?;
    module.add_function(wrap_pyfunction!(sync, &module)?)?;
    module.add_function(wrap_pyfunction!(verify, &module)?)?;
    module.add_function(wrap_pyfunction!(local_wal_ops, &module)?)?;
    module.add_function(wrap_pyfunction!(managed_plan, &module)?)?;
    module.add_function(wrap_pyfunction!(managed_record_base, &module)?)?;
    module.add_function(wrap_pyfunction!(managed_pull_requirements, &module)?)?;
    module.add_function(wrap_pyfunction!(managed_materialize, &module)?)?;
    parent.add_submodule(&module)?;
    Ok(())
}
