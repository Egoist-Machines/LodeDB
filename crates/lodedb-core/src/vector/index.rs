//! Shared vector-index data types.

use serde::{Deserialize, Serialize};

/// One chunk row owned by the native vector index.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoreVectorChunk {
    pub chunk_id: String,
    pub document_id: String,
    pub embedding: Vec<f32>,
}

impl CoreVectorChunk {
    pub fn new(
        chunk_id: impl Into<String>,
        document_id: impl Into<String>,
        embedding: Vec<f32>,
    ) -> Self {
        Self {
            chunk_id: chunk_id.into(),
            document_id: document_id.into(),
            embedding,
        }
    }
}

/// One vector search hit with resolved chunk/document metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorSearchHit {
    pub chunk_id: String,
    pub document_id: String,
    pub stable_id: u64,
    pub score: f32,
}

/// Safe backend metadata; contains counts and configuration only.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorBackendMetadata {
    pub compact_backend: String,
    pub native_backend: String,
    pub native_used: bool,
    pub dim: usize,
    pub bit_width: usize,
    pub generation: u64,
    pub vector_count: usize,
}

/// Metrics-only `.tvim` write report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorIndexWriteMetrics {
    pub compact_backend: String,
    pub snapshot_bytes: u64,
    pub persist_ms: f64,
    pub raw_payload_text_present: bool,
}
