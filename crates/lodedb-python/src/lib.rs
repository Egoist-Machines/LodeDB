//! Hidden PyO3 bindings for the native LodeDB core.
//!
//! This crate is intentionally not wired into the Python public API yet. Later
//! migration milestones can expose shadow-mode helpers without changing
//! `import lodedb` or importing heavy embedding dependencies.

use lodedb_core::{
    CoreDocument, CoreError, CoreErrorCode, CoreIndexConfig, CoreMutationResult, CoreOpenOptions,
    CoreQuery, CoreRoutePolicy, CoreSearchResults, CoreSecurityOptions, CoreStats,
    CoreVectorDocument,
};
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Returns the native-core crate version.
#[pyfunction]
fn native_core_version() -> &'static str {
    lodedb_core::CORE_VERSION
}

/// Returns the persisted-store schema version understood by the native core.
#[pyfunction]
fn storage_schema_version() -> u32 {
    lodedb_core::STORAGE_SCHEMA_VERSION
}

/// Builds and serializes a `CoreDocument`.
#[pyfunction]
fn core_document_to_json(
    document_id: String,
    text: String,
    metadata: std::collections::BTreeMap<String, String>,
) -> PyResult<String> {
    to_json(&CoreDocument {
        document_id,
        text,
        metadata,
    })
}

/// Deserializes then serializes a named core type.
#[pyfunction]
fn round_trip_core_json(type_name: &str, json: &str) -> PyResult<String> {
    match type_name {
        "CoreDocument" => round_trip::<CoreDocument>(json),
        "CoreIndexConfig" => round_trip::<CoreIndexConfig>(json),
        "CoreMutationResult" => round_trip::<CoreMutationResult>(json),
        "CoreOpenOptions" => round_trip::<CoreOpenOptions>(json),
        "CoreQuery" => round_trip::<CoreQuery>(json),
        "CoreRoutePolicy" => round_trip::<CoreRoutePolicy>(json),
        "CoreSearchResults" => round_trip::<CoreSearchResults>(json),
        "CoreSecurityOptions" => round_trip::<CoreSecurityOptions>(json),
        "CoreStats" => round_trip::<CoreStats>(json),
        "CoreVectorDocument" => round_trip::<CoreVectorDocument>(json),
        _ => Err(core_error_to_py(CoreError::new(
            CoreErrorCode::InvalidArgument,
            "unknown core type",
        ))),
    }
}

fn round_trip<T>(json: &str) -> PyResult<String>
where
    T: serde::Serialize + for<'de> serde::Deserialize<'de>,
{
    let value: T = serde_json::from_str(json).map_err(|error| {
        core_error_to_py(CoreError::new(
            CoreErrorCode::InvalidArgument,
            format!("invalid JSON payload: {error}"),
        ))
    })?;
    to_json(&value)
}

fn to_json<T: serde::Serialize>(value: &T) -> PyResult<String> {
    serde_json::to_string(value).map_err(|error| {
        core_error_to_py(CoreError::new(
            CoreErrorCode::Internal,
            format!("failed to serialize core type: {error}"),
        ))
    })
}

fn core_error_to_py(error: CoreError) -> PyErr {
    match error.code() {
        CoreErrorCode::InvalidArgument | CoreErrorCode::PlanStale => {
            PyValueError::new_err(error.to_string())
        }
        CoreErrorCode::NotFound => PyKeyError::new_err(error.to_string()),
        CoreErrorCode::CorruptStore | CoreErrorCode::Unsupported | CoreErrorCode::Internal => {
            PyRuntimeError::new_err(error.to_string())
        }
    }
}

/// Hidden native-core Python module.
#[pymodule]
fn _native_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", lodedb_core::CORE_VERSION)?;
    module.add_function(wrap_pyfunction!(native_core_version, module)?)?;
    module.add_function(wrap_pyfunction!(storage_schema_version, module)?)?;
    module.add_function(wrap_pyfunction!(core_document_to_json, module)?)?;
    module.add_function(wrap_pyfunction!(round_trip_core_json, module)?)?;
    Ok(())
}
