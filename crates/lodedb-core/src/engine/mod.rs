//! In-memory native core engine.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{CoreError, CoreErrorCode};
use crate::filter::{build_field_indexes, coerce_sdk_filter, resolve_filter};
use crate::types::{
    CoreMetadata, CoreMutationResult, CoreSearchHit, CoreSearchResults, CoreVectorDocument,
};
use crate::version::{CORE_VERSION, STORAGE_SCHEMA_VERSION};

/// In-memory vector-only native core engine.
#[derive(Debug, Default)]
pub struct CoreEngine {
    indexes: BTreeMap<String, VectorOnlyIndex>,
}

impl CoreEngine {
    /// Creates an empty in-memory engine. No files are read or written.
    pub fn new_in_memory() -> Self {
        Self::default()
    }

    /// Creates a vector-only index.
    pub fn create_index(
        &mut self,
        index_id: impl Into<String>,
        vector_dim: usize,
        bit_width: usize,
    ) -> Result<(), CoreError> {
        validate_index_shape(vector_dim, bit_width)?;
        let index_id = index_id.into();
        if self.indexes.contains_key(&index_id) {
            return invalid("index already exists");
        }
        self.indexes.insert(
            index_id.clone(),
            VectorOnlyIndex::new(index_id, vector_dim, bit_width),
        );
        Ok(())
    }

    /// Upserts vector documents into an existing index.
    pub fn upsert_vectors(
        &mut self,
        index_id: &str,
        documents: &[CoreVectorDocument],
    ) -> Result<CoreMutationResult, CoreError> {
        let index = self.index_mut(index_id)?;
        let mut changed = 0usize;
        for document in documents {
            if document.document_id.trim().is_empty() {
                return invalid("document_id is required");
            }
            if document.vector.len() != index.vector_dim {
                return invalid("vector dimension does not match index");
            }
            index.documents.insert(
                document.document_id.clone(),
                VectorRecord {
                    vector: document.vector.clone(),
                    metadata: document.metadata.clone(),
                    text: document.text.clone(),
                },
            );
            changed += 1;
        }
        if changed > 0 {
            index.generation += 1;
        }
        Ok(CoreMutationResult {
            documents_upserted: changed,
            documents_deleted: 0,
            chunks_upserted: changed,
            chunks_deleted: 0,
            generation: index.generation,
        })
    }

    /// Deletes documents from an index.
    pub fn delete_documents(
        &mut self,
        index_id: &str,
        document_ids: &[String],
    ) -> Result<CoreMutationResult, CoreError> {
        let index = self.index_mut(index_id)?;
        let mut deleted = 0usize;
        let mut seen = BTreeSet::new();
        for document_id in document_ids {
            if document_id.trim().is_empty() {
                return invalid("document_id is required");
            }
            if seen.insert(document_id.clone()) && index.documents.remove(document_id).is_some() {
                deleted += 1;
            }
        }
        if deleted > 0 {
            index.delete_count += deleted;
            index.deleted_chunk_count += deleted;
            index.generation += 1;
        }
        Ok(CoreMutationResult {
            documents_upserted: 0,
            documents_deleted: deleted,
            chunks_upserted: 0,
            chunks_deleted: deleted,
            generation: index.generation,
        })
    }

    /// Updates metadata/text payload for an existing vector document.
    pub fn update_document_payload(
        &mut self,
        index_id: &str,
        document_id: &str,
        metadata: Option<CoreMetadata>,
        text: Option<Option<String>>,
    ) -> Result<CoreMutationResult, CoreError> {
        let index = self.index_mut(index_id)?;
        let Some(record) = index.documents.get_mut(document_id) else {
            return invalid("document not found");
        };
        if let Some(metadata) = metadata {
            record.metadata = metadata;
        }
        if let Some(text) = text {
            record.text = text;
        }
        index.generation += 1;
        Ok(CoreMutationResult {
            documents_upserted: 1,
            documents_deleted: 0,
            chunks_upserted: 0,
            chunks_deleted: 0,
            generation: index.generation,
        })
    }

    /// Queries one vector.
    pub fn query_vector(
        &self,
        index_id: &str,
        query_vector: &[f32],
        top_k: usize,
        filter: Option<&Value>,
    ) -> Result<CoreSearchResults, CoreError> {
        let index = self.index(index_id)?;
        if top_k == 0 {
            return invalid("top_k must be positive");
        }
        if query_vector.len() != index.vector_dim {
            return invalid("query dimension does not match index");
        }
        let candidates = index.resolve_filter(filter)?;
        let total_considered = candidates.len();
        let mut hits = candidates
            .into_iter()
            .filter_map(|document_id| {
                let record = index.documents.get(&document_id)?;
                Some(CoreSearchHit {
                    document_id: document_id.clone(),
                    chunk_id: document_id,
                    score: dot(query_vector, &record.vector),
                    metadata: record.metadata.clone(),
                })
            })
            .collect::<Vec<_>>();
        hits.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.document_id.cmp(&right.document_id))
        });
        hits.truncate(top_k);
        Ok(CoreSearchResults {
            hits,
            total_considered,
        })
    }

    /// Queries a batch of vectors with one shared filter.
    pub fn query_vectors_batch(
        &self,
        index_id: &str,
        query_vectors: &[Vec<f32>],
        top_k: usize,
        filter: Option<&Value>,
    ) -> Result<Vec<CoreSearchResults>, CoreError> {
        query_vectors
            .iter()
            .map(|query| self.query_vector(index_id, query, top_k, filter))
            .collect()
    }

    /// Returns metrics-only stats for an index.
    pub fn stats(&self, index_id: &str) -> Result<CoreEngineStats, CoreError> {
        let index = self.index(index_id)?;
        Ok(CoreEngineStats {
            index_id: index.index_id.clone(),
            document_count: index.documents.len(),
            chunk_count: index.documents.len(),
            embedded_chunk_count: index.documents.len(),
            delete_count: index.delete_count,
            deleted_chunk_count: index.deleted_chunk_count,
            generation: index.generation,
            storage_schema_version: STORAGE_SCHEMA_VERSION,
            native_core_enabled: true,
            native_core_version: CORE_VERSION.to_string(),
            vector_dim: index.vector_dim,
            bit_width: index.bit_width,
            raw_payload_text_present: false,
        })
    }

    fn index(&self, index_id: &str) -> Result<&VectorOnlyIndex, CoreError> {
        self.indexes
            .get(index_id)
            .ok_or_else(|| invalid_err("index not found"))
    }

    fn index_mut(&mut self, index_id: &str) -> Result<&mut VectorOnlyIndex, CoreError> {
        self.indexes
            .get_mut(index_id)
            .ok_or_else(|| invalid_err("index not found"))
    }
}

#[derive(Debug, Clone)]
struct VectorOnlyIndex {
    index_id: String,
    vector_dim: usize,
    bit_width: usize,
    documents: BTreeMap<String, VectorRecord>,
    generation: u64,
    delete_count: usize,
    deleted_chunk_count: usize,
}

impl VectorOnlyIndex {
    fn new(index_id: String, vector_dim: usize, bit_width: usize) -> Self {
        Self {
            index_id,
            vector_dim,
            bit_width,
            documents: BTreeMap::new(),
            generation: 0,
            delete_count: 0,
            deleted_chunk_count: 0,
        }
    }

    fn resolve_filter(&self, filter: Option<&Value>) -> Result<BTreeSet<String>, CoreError> {
        let all_docs = self.documents.keys().cloned().collect::<BTreeSet<_>>();
        let Some(filter) = filter else {
            return Ok(all_docs);
        };
        let object = filter
            .as_object()
            .ok_or_else(|| invalid_err("filter must be an object"))?;
        let structured = object
            .keys()
            .all(|key| key == "metadata" || key == "document_ids")
            && object
                .keys()
                .any(|key| key == "metadata" || key == "document_ids");
        let mut candidates = all_docs;
        let metadata_filter = if structured {
            if let Some(document_ids) = object.get("document_ids") {
                let ids = document_ids
                    .as_array()
                    .ok_or_else(|| invalid_err("document_ids filter must be a list"))?
                    .iter()
                    .map(|value| value.as_str().unwrap_or("").to_string())
                    .collect::<BTreeSet<_>>();
                candidates = candidates.intersection(&ids).cloned().collect();
            }
            object.get("metadata")
        } else {
            Some(filter)
        };
        if let Some(metadata_filter) = metadata_filter {
            let metadata_by_id = self
                .documents
                .iter()
                .map(|(document_id, record)| (document_id.clone(), record.metadata.clone()))
                .collect::<BTreeMap<_, _>>();
            let (fields, all_docs) = build_field_indexes(&metadata_by_id);
            let coerced = coerce_sdk_filter(metadata_filter)?;
            let metadata_docs = resolve_filter(&coerced, &fields, &all_docs)?;
            candidates = candidates.intersection(&metadata_docs).cloned().collect();
        }
        Ok(candidates)
    }
}

#[derive(Debug, Clone)]
struct VectorRecord {
    vector: Vec<f32>,
    metadata: CoreMetadata,
    text: Option<String>,
}

/// Metrics-only stats for the in-memory engine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoreEngineStats {
    pub index_id: String,
    pub document_count: usize,
    pub chunk_count: usize,
    pub embedded_chunk_count: usize,
    pub delete_count: usize,
    pub deleted_chunk_count: usize,
    pub generation: u64,
    pub storage_schema_version: u32,
    pub native_core_enabled: bool,
    pub native_core_version: String,
    pub vector_dim: usize,
    pub bit_width: usize,
    pub raw_payload_text_present: bool,
}

fn validate_index_shape(vector_dim: usize, bit_width: usize) -> Result<(), CoreError> {
    if vector_dim == 0 || vector_dim > 65_536 {
        return invalid("vector_dim must be between 1 and 65536");
    }
    if !matches!(bit_width, 2 | 4) {
        return invalid("bit_width must be 2 or 4");
    }
    Ok(())
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn invalid<T>(message: impl Into<String>) -> Result<T, CoreError> {
    Err(invalid_err(message))
}

fn invalid_err(message: impl Into<String>) -> CoreError {
    CoreError::new(CoreErrorCode::InvalidArgument, message)
}
