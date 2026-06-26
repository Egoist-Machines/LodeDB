//! Shared native-core wire contracts.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::error::{CoreError, CoreErrorCode};
use crate::version::{CORE_VERSION, STORAGE_SCHEMA_VERSION};

/// Metadata stored by the native core.
///
/// Values are strings by design: SDK layers coerce user metadata before it
/// enters the engine, and the Rust core should preserve those exact strings for
/// filtering, storage, and redacted result assembly.
pub type CoreMetadata = BTreeMap<String, String>;

/// Minimal API-version record exposed before the full engine lands.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoreApiVersion {
    /// Native-core crate version.
    pub core_version: String,
    /// Persisted-store schema version.
    pub storage_schema_version: u32,
}

impl Default for CoreApiVersion {
    fn default() -> Self {
        Self {
            core_version: CORE_VERSION.to_string(),
            storage_schema_version: STORAGE_SCHEMA_VERSION,
        }
    }
}

/// Persistent open options shared across bindings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoreOpenOptions {
    pub path: String,
    pub read_only: bool,
    pub durability: String,
    pub commit_mode: String,
    pub store_text: bool,
    pub index_text: bool,
}

/// Local-first security and telemetry options.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoreSecurityOptions {
    pub bind_host: String,
    pub route_profile: String,
    pub telemetry_mode: String,
    pub allow_raw_result_text: bool,
}

/// Route policy selected by the binding layer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoreRoutePolicy {
    pub profile: String,
    pub label: String,
    pub model: String,
    pub provider: String,
    pub task: String,
    pub native_dim: usize,
    pub index_backend: String,
    pub turbovec_bit_width: Option<u8>,
}

/// Stored index shape and schema identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoreIndexConfig {
    pub index_id: String,
    pub name: String,
    pub model: String,
    pub native_dim: usize,
    pub storage_schema_version: u32,
    pub route_policy: CoreRoutePolicy,
}

impl CoreIndexConfig {
    /// Validates that an embedding model and vector dimensionality match this index.
    pub fn validate_embedding_shape(
        &self,
        model: &str,
        vector_dim: usize,
    ) -> Result<(), CoreError> {
        if self.model != model {
            return Err(CoreError::new(
                CoreErrorCode::InvalidArgument,
                "embedding model does not match index config",
            ));
        }
        if self.native_dim != vector_dim {
            return Err(CoreError::new(
                CoreErrorCode::InvalidArgument,
                "embedding dimension does not match index config",
            ));
        }
        Ok(())
    }
}

/// Native index creation options including persisted storage identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoreIndexCreateOptions {
    pub index_id: String,
    pub index_key: String,
    pub client_id_hash: String,
    pub name: String,
    pub model: String,
    pub provider: String,
    pub task: String,
    pub route_profile: String,
    pub storage_profile: String,
    pub vector_dim: usize,
    pub bit_width: usize,
}

impl CoreIndexCreateOptions {
    /// Returns the native-core default metadata used by the simple create path.
    pub fn native_default(
        index_id: impl Into<String>,
        vector_dim: usize,
        bit_width: usize,
    ) -> Self {
        let index_id = index_id.into();
        Self {
            index_key: index_id.clone(),
            client_id_hash: index_id.clone(),
            index_id,
            name: "lodedb-local".to_string(),
            model: "native-core".to_string(),
            provider: "native".to_string(),
            task: "native-core".to_string(),
            route_profile: "native-core".to_string(),
            storage_profile: "native-core".to_string(),
            vector_dim,
            bit_width,
        }
    }
}

/// Text document supplied by a binding layer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoreDocument {
    pub document_id: String,
    pub text: String,
    pub metadata: CoreMetadata,
}

/// Vector-in document supplied by a binding layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoreVectorDocument {
    pub document_id: String,
    pub vector: Vec<f32>,
    pub metadata: CoreMetadata,
    pub text: Option<String>,
}

impl CoreVectorDocument {
    /// Validates this vector against the configured model identity and dimension.
    pub fn validate_for_index(
        &self,
        config: &CoreIndexConfig,
        model: &str,
    ) -> Result<(), CoreError> {
        config.validate_embedding_shape(model, self.vector.len())
    }
}

/// Query supplied by a binding layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoreQuery {
    pub text: String,
    pub top_k: usize,
    pub filter: Option<serde_json::Value>,
    pub include: Vec<String>,
    pub mode: String,
    pub embedding: Option<Vec<f32>>,
}

/// One assembled search hit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoreSearchHit {
    pub document_id: String,
    pub chunk_id: String,
    pub score: f32,
    pub metadata: CoreMetadata,
}

/// Search response for one query.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoreSearchResults {
    pub hits: Vec<CoreSearchHit>,
    pub total_considered: usize,
}

/// Mutation response shared by text and vector ingest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoreMutationResult {
    pub documents_upserted: usize,
    pub documents_deleted: usize,
    pub chunks_upserted: usize,
    pub chunks_deleted: usize,
    pub generation: u64,
}

/// Metrics-only engine stats.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoreStats {
    pub document_count: usize,
    pub chunk_count: usize,
    pub generation: u64,
    pub storage_schema_version: u32,
    pub native_core_enabled: bool,
    pub native_core_version: String,
}

impl Default for CoreStats {
    fn default() -> Self {
        Self {
            document_count: 0,
            chunk_count: 0,
            generation: 0,
            storage_schema_version: STORAGE_SCHEMA_VERSION,
            native_core_enabled: false,
            native_core_version: CORE_VERSION.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CoreApiVersion, CoreDocument, CoreIndexConfig, CoreIndexCreateOptions, CoreMetadata,
        CoreMutationResult, CoreOpenOptions, CoreQuery, CoreRoutePolicy, CoreSearchHit,
        CoreSearchResults, CoreSecurityOptions, CoreStats, CoreVectorDocument,
    };
    use crate::error::CoreErrorCode;

    fn metadata() -> CoreMetadata {
        CoreMetadata::from([
            ("kind".to_string(), "note".to_string()),
            ("year".to_string(), "2024".to_string()),
        ])
    }

    fn route_policy() -> CoreRoutePolicy {
        CoreRoutePolicy {
            profile: "local".to_string(),
            label: "Local".to_string(),
            model: "hash_fixture".to_string(),
            provider: "local".to_string(),
            task: "text".to_string(),
            native_dim: 8,
            index_backend: "turbovec_direct".to_string(),
            turbovec_bit_width: Some(4),
        }
    }

    fn index_config() -> CoreIndexConfig {
        CoreIndexConfig {
            index_id: "default".to_string(),
            name: "Default index".to_string(),
            model: "hash_fixture".to_string(),
            native_dim: 8,
            storage_schema_version: 1,
            route_policy: route_policy(),
        }
    }

    fn assert_round_trip<T>(value: &T)
    where
        T: Serialize + for<'de> Deserialize<'de> + PartialEq + std::fmt::Debug,
    {
        let encoded = serde_json::to_string(value).expect("serialize");
        let decoded: T = serde_json::from_str(&encoded).expect("deserialize");
        assert_eq!(&decoded, value);
    }

    use serde::{Deserialize, Serialize};

    #[test]
    fn domain_types_round_trip_json() {
        assert_round_trip(&CoreApiVersion::default());
        assert_round_trip(&CoreOpenOptions {
            path: "/tmp/lodedb".to_string(),
            read_only: false,
            durability: "fast".to_string(),
            commit_mode: "wal".to_string(),
            store_text: true,
            index_text: false,
        });
        assert_round_trip(&CoreSecurityOptions {
            bind_host: "127.0.0.1".to_string(),
            route_profile: "local".to_string(),
            telemetry_mode: "metrics_only".to_string(),
            allow_raw_result_text: false,
        });
        assert_round_trip(&route_policy());
        assert_round_trip(&index_config());
        assert_round_trip(&CoreIndexCreateOptions::native_default("default", 8, 4));
        assert_round_trip(&CoreDocument {
            document_id: "doc-alpha".to_string(),
            text: "alpha document".to_string(),
            metadata: metadata(),
        });
        assert_round_trip(&CoreVectorDocument {
            document_id: "vec-alpha".to_string(),
            vector: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            metadata: metadata(),
            text: Some("retained text".to_string()),
        });
        assert_round_trip(&CoreQuery {
            text: "alpha".to_string(),
            top_k: 3,
            filter: Some(serde_json::json!({"year": {"$gte": 2024}})),
            include: vec!["metadata".to_string()],
            mode: "hybrid".to_string(),
            embedding: Some(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        });
        assert_round_trip(&CoreSearchResults {
            hits: vec![CoreSearchHit {
                document_id: "doc-alpha".to_string(),
                chunk_id: "doc-alpha:abc".to_string(),
                score: 1.0,
                metadata: metadata(),
            }],
            total_considered: 1,
        });
        assert_round_trip(&CoreMutationResult {
            documents_upserted: 1,
            documents_deleted: 0,
            chunks_upserted: 1,
            chunks_deleted: 0,
            generation: 2,
        });
        assert_round_trip(&CoreStats::default());
    }

    #[test]
    fn metadata_values_are_strings() {
        let document = CoreDocument {
            document_id: "doc-alpha".to_string(),
            text: "alpha".to_string(),
            metadata: CoreMetadata::from([("year".to_string(), "2024".to_string())]),
        };
        let encoded = serde_json::to_string(&document).expect("serialize document");
        assert!(encoded.contains("\"year\":\"2024\""));
    }

    #[test]
    fn index_config_enforces_model_and_dimension() {
        let config = index_config();
        assert!(config.validate_embedding_shape("hash_fixture", 8).is_ok());

        let wrong_model = config
            .validate_embedding_shape("other", 8)
            .expect_err("model mismatch");
        assert_eq!(wrong_model.code(), CoreErrorCode::InvalidArgument);

        let wrong_dim = config
            .validate_embedding_shape("hash_fixture", 7)
            .expect_err("dimension mismatch");
        assert_eq!(wrong_dim.code(), CoreErrorCode::InvalidArgument);
    }
}
