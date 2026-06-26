//! Hidden PyO3 bindings for the native LodeDB core.
//!
//! This crate is intentionally not wired into the Python public API yet. Later
//! migration milestones can expose shadow-mode helpers without changing
//! `import lodedb` or importing heavy embedding dependencies.

use lodedb_core::{
    engine::{CoreEngine as RustCoreEngine, IngestPlan, QueryPlan},
    CoreDocument, CoreError, CoreErrorCode, CoreIndexConfig, CoreIndexCreateOptions,
    CoreMutationResult, CoreOpenOptions, CoreQuery, CoreRoutePolicy, CoreSearchResults,
    CoreSecurityOptions, CoreStats, CoreVectorDocument,
};
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use serde::de::DeserializeOwned;
use serde_json::Value;

/// Private Python-owned native engine handle.
///
/// Methods intentionally accept and return JSON strings so this hidden module
/// can share the exact native-core wire contracts without exposing a second
/// public Python object model during the migration.
#[pyclass(name = "CoreEngine", unsendable)]
struct PyCoreEngine {
    inner: RustCoreEngine,
}

#[pymethods]
impl PyCoreEngine {
    #[new]
    fn new() -> Self {
        Self {
            inner: RustCoreEngine::new_in_memory(),
        }
    }

    #[staticmethod]
    fn open(options_json: &str) -> PyResult<Self> {
        let options = from_json::<CoreOpenOptions>(options_json)?;
        Ok(Self {
            inner: RustCoreEngine::open(options).map_err(core_error_to_py)?,
        })
    }

    #[staticmethod]
    fn open_readonly(path: String, options_json: &str) -> PyResult<Self> {
        let options = from_json::<CoreOpenOptions>(options_json)?;
        Ok(Self {
            inner: RustCoreEngine::open_readonly(path, options).map_err(core_error_to_py)?,
        })
    }

    fn create_index(
        &mut self,
        index_id: String,
        vector_dim: usize,
        bit_width: usize,
    ) -> PyResult<()> {
        self.inner
            .create_index(index_id, vector_dim, bit_width)
            .map_err(core_error_to_py)
    }

    fn create_index_with_options(&mut self, options_json: &str) -> PyResult<()> {
        let options = from_json::<CoreIndexCreateOptions>(options_json)?;
        self.inner
            .create_index_with_options(options)
            .map_err(core_error_to_py)
    }

    fn upsert_vectors(&mut self, index_id: &str, documents_json: &str) -> PyResult<String> {
        let documents = from_json::<Vec<CoreVectorDocument>>(documents_json)?;
        to_json(
            &self
                .inner
                .upsert_vectors(index_id, &documents)
                .map_err(core_error_to_py)?,
        )
    }

    fn delete_documents(&mut self, index_id: &str, document_ids_json: &str) -> PyResult<String> {
        let document_ids = from_json::<Vec<String>>(document_ids_json)?;
        to_json(
            &self
                .inner
                .delete_documents(index_id, &document_ids)
                .map_err(core_error_to_py)?,
        )
    }

    fn update_document_payload(
        &mut self,
        index_id: &str,
        document_id: &str,
        metadata_json: Option<&str>,
        text_json: Option<&str>,
    ) -> PyResult<String> {
        let metadata: Option<std::collections::BTreeMap<String, String>> =
            metadata_json.map(from_json).transpose()?;
        let text: Option<Option<String>> = text_json.map(from_json).transpose()?;
        to_json(
            &self
                .inner
                .update_document_payload(index_id, document_id, metadata, text)
                .map_err(core_error_to_py)?,
        )
    }

    fn query_vector(
        &self,
        index_id: &str,
        query_vector_json: &str,
        top_k: usize,
        filter_json: Option<&str>,
    ) -> PyResult<String> {
        let query_vector = from_json::<Vec<f32>>(query_vector_json)?;
        let filter = optional_value(filter_json)?;
        to_json(
            &self
                .inner
                .query_vector(index_id, &query_vector, top_k, filter.as_ref())
                .map_err(core_error_to_py)?,
        )
    }

    fn query_vectors_batch(
        &self,
        index_id: &str,
        query_vectors_json: &str,
        top_k: usize,
        filter_json: Option<&str>,
    ) -> PyResult<String> {
        let query_vectors = from_json::<Vec<Vec<f32>>>(query_vectors_json)?;
        let filter = optional_value(filter_json)?;
        to_json(
            &self
                .inner
                .query_vectors_batch(index_id, &query_vectors, top_k, filter.as_ref())
                .map_err(core_error_to_py)?,
        )
    }

    fn prepare_text_upsert(
        &mut self,
        index_id: &str,
        documents_json: &str,
        store_text: bool,
        index_text: bool,
        chunk_character_limit: usize,
    ) -> PyResult<String> {
        let documents = from_json::<Vec<CoreDocument>>(documents_json)?;
        to_json(
            &self
                .inner
                .prepare_text_upsert(
                    index_id,
                    &documents,
                    store_text,
                    index_text,
                    chunk_character_limit,
                )
                .map_err(core_error_to_py)?,
        )
    }

    fn apply_text_upsert(
        &mut self,
        plan_json: &str,
        embeddings_json: &str,
        embedding_time_ms: f64,
    ) -> PyResult<String> {
        let plan = from_json::<IngestPlan>(plan_json)?;
        let embeddings = from_json::<Vec<Vec<f32>>>(embeddings_json)?;
        to_json(
            &self
                .inner
                .apply_text_upsert(&plan, &embeddings, embedding_time_ms)
                .map_err(core_error_to_py)?,
        )
    }

    fn prepare_query_text(&self, query: &str, mode: &str) -> PyResult<String> {
        to_json(
            &self
                .inner
                .prepare_query_text(query, mode)
                .map_err(core_error_to_py)?,
        )
    }

    fn search_embedded_text(
        &self,
        index_id: &str,
        query_plan_json: &str,
        query_embedding_json: Option<&str>,
        top_k: usize,
        filter_json: Option<&str>,
    ) -> PyResult<String> {
        let query_plan = from_json::<QueryPlan>(query_plan_json)?;
        let query_embedding: Option<Vec<f32>> = query_embedding_json.map(from_json).transpose()?;
        let filter = optional_value(filter_json)?;
        to_json(
            &self
                .inner
                .search_embedded_text(
                    index_id,
                    &query_plan,
                    query_embedding.as_deref(),
                    top_k,
                    filter.as_ref(),
                )
                .map_err(core_error_to_py)?,
        )
    }

    fn stats(&self, index_id: &str) -> PyResult<String> {
        to_json(&self.inner.stats(index_id).map_err(core_error_to_py)?)
    }

    fn document_token_lists(&self, index_id: &str) -> PyResult<String> {
        to_json(
            &self
                .inner
                .document_token_lists(index_id)
                .map_err(core_error_to_py)?,
        )
    }

    fn persist(&mut self) -> PyResult<()> {
        self.inner.persist().map_err(core_error_to_py)
    }

    fn close(&mut self) -> PyResult<()> {
        self.inner.close().map_err(core_error_to_py)
    }
}

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
    let value: T = from_json(json)?;
    to_json(&value)
}

fn from_json<T>(json: &str) -> PyResult<T>
where
    T: DeserializeOwned,
{
    serde_json::from_str(json).map_err(|error| {
        core_error_to_py(CoreError::new(
            CoreErrorCode::InvalidArgument,
            format!("invalid JSON payload: {error}"),
        ))
    })
}

fn to_json<T: serde::Serialize>(value: &T) -> PyResult<String> {
    serde_json::to_string(value).map_err(|error| {
        core_error_to_py(CoreError::new(
            CoreErrorCode::Internal,
            format!("failed to serialize core type: {error}"),
        ))
    })
}

fn optional_value(json: Option<&str>) -> PyResult<Option<Value>> {
    json.map(from_json).transpose()
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
    module.add_class::<PyCoreEngine>()?;
    module.add_function(wrap_pyfunction!(native_core_version, module)?)?;
    module.add_function(wrap_pyfunction!(storage_schema_version, module)?)?;
    module.add_function(wrap_pyfunction!(core_document_to_json, module)?)?;
    module.add_function(wrap_pyfunction!(round_trip_core_json, module)?)?;
    Ok(())
}
