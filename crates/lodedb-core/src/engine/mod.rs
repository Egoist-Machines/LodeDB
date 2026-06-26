//! In-memory native core engine.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{CoreError, CoreErrorCode};
use crate::filter::{build_field_indexes, coerce_sdk_filter, resolve_filter};
use crate::lexical::tokenize;
use crate::text::chunk::{chunk_id_for_hash, chunk_text};
use crate::text::hash::normalized_chunk_hash;
use crate::types::{
    CoreDocument, CoreMetadata, CoreMutationResult, CoreSearchHit, CoreSearchResults,
    CoreVectorDocument,
};
use crate::version::{CORE_VERSION, STORAGE_SCHEMA_VERSION};

/// In-memory vector-only native core engine.
#[derive(Debug, Default)]
pub struct CoreEngine {
    indexes: BTreeMap<String, VectorOnlyIndex>,
    next_plan_id: u64,
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
                DocumentRecord {
                    metadata: document.metadata.clone(),
                    text: document.text.clone(),
                    token_lists: Vec::new(),
                    chunks: vec![ChunkRecord {
                        chunk_id: document.document_id.clone(),
                        vector: document.vector.clone(),
                    }],
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
        let mut deleted_chunks = 0usize;
        let mut seen = BTreeSet::new();
        for document_id in document_ids {
            if document_id.trim().is_empty() {
                return invalid("document_id is required");
            }
            if seen.insert(document_id.clone()) {
                let Some(record) = index.documents.remove(document_id) else {
                    continue;
                };
                deleted += 1;
                deleted_chunks += record.chunks.len();
            }
        }
        if deleted > 0 {
            index.delete_count += deleted;
            index.deleted_chunk_count += deleted_chunks;
            index.generation += 1;
        }
        Ok(CoreMutationResult {
            documents_upserted: 0,
            documents_deleted: deleted,
            chunks_upserted: 0,
            chunks_deleted: deleted_chunks,
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

    /// Plans a text upsert while embeddings stay in the binding layer.
    pub fn prepare_text_upsert(
        &mut self,
        index_id: &str,
        documents: &[CoreDocument],
        store_text: bool,
        index_text: bool,
        chunk_character_limit: usize,
    ) -> Result<IngestPlan, CoreError> {
        let base_generation = self.index(index_id)?.generation;
        let existing_chunks = self.index(index_id)?.chunk_vectors_by_id();
        let mut prepared_documents = Vec::with_capacity(documents.len());
        let mut chunks_to_embed = Vec::new();

        for document in documents {
            if document.document_id.trim().is_empty() {
                return invalid("document_id is required");
            }
            if document.text.trim().is_empty() {
                return invalid("document text is required");
            }
            let pieces = chunk_text(&document.text, chunk_character_limit)?;
            let mut occurrences: BTreeMap<String, usize> = BTreeMap::new();
            let mut chunks = Vec::with_capacity(pieces.len());
            for piece in pieces {
                let chunk_hash = normalized_chunk_hash(&piece);
                let occurrence = *occurrences.get(&chunk_hash).unwrap_or(&0);
                occurrences.insert(chunk_hash.clone(), occurrence + 1);
                let chunk_id = chunk_id_for_hash(&document.document_id, &chunk_hash, occurrence);
                let tokens = if index_text {
                    tokenize(&piece)
                } else {
                    Vec::new()
                };
                let needs_embedding = !existing_chunks.contains_key(&chunk_id);
                if needs_embedding {
                    chunks_to_embed.push(PlanEmbeddingChunk {
                        document_id: document.document_id.clone(),
                        chunk_id: chunk_id.clone(),
                        text: piece.clone(),
                    });
                }
                chunks.push(PlanDocumentChunk {
                    chunk_id,
                    text: piece,
                    tokens,
                    needs_embedding,
                });
            }
            prepared_documents.push(PlanDocument {
                document_id: document.document_id.clone(),
                metadata: document.metadata.clone(),
                text: if store_text {
                    Some(document.text.clone())
                } else {
                    None
                },
                chunks,
            });
        }

        let plan_id = self.next_plan_id;
        self.next_plan_id += 1;
        Ok(IngestPlan {
            plan_id,
            index_id: index_id.to_string(),
            base_generation,
            documents: prepared_documents,
            chunks_to_embed,
            store_text,
            index_text,
        })
    }

    /// Applies a text upsert plan with binding-provided embeddings.
    pub fn apply_text_upsert(
        &mut self,
        plan: &IngestPlan,
        embeddings: &[Vec<f32>],
        embedding_time_ms: f64,
    ) -> Result<TextApplyResult, CoreError> {
        let index = self.index_mut(&plan.index_id)?;
        if index.generation != plan.base_generation {
            return Err(CoreError::new(
                CoreErrorCode::PlanStale,
                "ingest plan is stale",
            ));
        }
        if embeddings.len() != plan.chunks_to_embed.len() {
            return invalid("embedding count does not match ingest plan");
        }
        for embedding in embeddings {
            if embedding.len() != index.vector_dim {
                return invalid("embedding dimension does not match index");
            }
        }

        let mut new_embeddings = plan
            .chunks_to_embed
            .iter()
            .zip(embeddings)
            .map(|(chunk, embedding)| (chunk.chunk_id.clone(), embedding.clone()))
            .collect::<BTreeMap<_, _>>();
        let existing_chunks = index.chunk_vectors_by_id();
        let mut chunks_upserted = 0usize;
        let mut reused_chunks = 0usize;
        for document in &plan.documents {
            let mut chunks = Vec::with_capacity(document.chunks.len());
            let token_lists = document
                .chunks
                .iter()
                .map(|chunk| chunk.tokens.clone())
                .collect::<Vec<_>>();
            for chunk in &document.chunks {
                let vector = if let Some(embedding) = new_embeddings.remove(&chunk.chunk_id) {
                    chunks_upserted += 1;
                    embedding
                } else if let Some(existing) = existing_chunks.get(&chunk.chunk_id) {
                    reused_chunks += 1;
                    existing.clone()
                } else {
                    return invalid("ingest plan references a missing reusable chunk");
                };
                chunks.push(ChunkRecord {
                    chunk_id: chunk.chunk_id.clone(),
                    vector,
                });
            }
            index.documents.insert(
                document.document_id.clone(),
                DocumentRecord {
                    metadata: document.metadata.clone(),
                    text: document.text.clone(),
                    token_lists,
                    chunks,
                },
            );
        }
        if !plan.documents.is_empty() {
            index.generation += 1;
        }
        Ok(TextApplyResult {
            mutation: CoreMutationResult {
                documents_upserted: plan.documents.len(),
                documents_deleted: 0,
                chunks_upserted,
                chunks_deleted: 0,
                generation: index.generation,
            },
            embedded_chunks: chunks_upserted,
            reused_chunks,
            embedding_time_ms,
        })
    }

    /// Prepares a text query; embeddings remain a binding responsibility.
    pub fn prepare_query_text(&self, query: &str, mode: &str) -> Result<QueryPlan, CoreError> {
        if query.trim().is_empty() {
            return invalid("query must be a non-empty string");
        }
        if !matches!(mode, "vector" | "hybrid" | "lexical") {
            return invalid("unsupported query mode");
        }
        Ok(QueryPlan {
            query: query.to_string(),
            mode: mode.to_string(),
            query_tokens: tokenize(query),
            requires_embedding: matches!(mode, "vector" | "hybrid"),
        })
    }

    /// Executes a prepared text query with a binding-provided query embedding.
    pub fn search_embedded_text(
        &self,
        index_id: &str,
        query_plan: &QueryPlan,
        query_embedding: Option<&[f32]>,
        top_k: usize,
        filter: Option<&Value>,
    ) -> Result<CoreSearchResults, CoreError> {
        if query_plan.requires_embedding {
            let embedding = query_embedding
                .ok_or_else(|| invalid_err("query embedding is required for this mode"))?;
            return self.query_vector(index_id, embedding, top_k, filter);
        }
        invalid("lexical-only text search is not implemented in the prepare/apply protocol yet")
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
            .flat_map(|document_id| {
                let Some(record) = index.documents.get(&document_id) else {
                    return Vec::new();
                };
                record
                    .chunks
                    .iter()
                    .map(|chunk| CoreSearchHit {
                        document_id: document_id.clone(),
                        chunk_id: chunk.chunk_id.clone(),
                        score: dot(query_vector, &chunk.vector),
                        metadata: record.metadata.clone(),
                    })
                    .collect::<Vec<_>>()
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
            chunk_count: index.chunk_count(),
            embedded_chunk_count: index.chunk_count(),
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

    /// Returns captured per-document, per-chunk lexical tokens.
    pub fn document_token_lists(
        &self,
        index_id: &str,
    ) -> Result<BTreeMap<String, Vec<Vec<String>>>, CoreError> {
        let index = self.index(index_id)?;
        Ok(index
            .documents
            .iter()
            .map(|(document_id, record)| (document_id.clone(), record.token_lists.clone()))
            .collect())
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
    documents: BTreeMap<String, DocumentRecord>,
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

    fn chunk_count(&self) -> usize {
        self.documents
            .values()
            .map(|record| record.chunks.len())
            .sum()
    }

    fn chunk_vectors_by_id(&self) -> BTreeMap<String, Vec<f32>> {
        let mut chunks = BTreeMap::new();
        for record in self.documents.values() {
            for chunk in &record.chunks {
                chunks.insert(chunk.chunk_id.clone(), chunk.vector.clone());
            }
        }
        chunks
    }
}

#[derive(Debug, Clone)]
struct DocumentRecord {
    metadata: CoreMetadata,
    text: Option<String>,
    token_lists: Vec<Vec<String>>,
    chunks: Vec<ChunkRecord>,
}

#[derive(Debug, Clone)]
struct ChunkRecord {
    chunk_id: String,
    vector: Vec<f32>,
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

/// Text ingest plan returned to a binding before embeddings are computed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IngestPlan {
    pub plan_id: u64,
    pub index_id: String,
    pub base_generation: u64,
    pub documents: Vec<PlanDocument>,
    pub chunks_to_embed: Vec<PlanEmbeddingChunk>,
    pub store_text: bool,
    pub index_text: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanDocument {
    pub document_id: String,
    pub metadata: CoreMetadata,
    pub text: Option<String>,
    pub chunks: Vec<PlanDocumentChunk>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanDocumentChunk {
    pub chunk_id: String,
    pub text: String,
    pub tokens: Vec<String>,
    pub needs_embedding: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanEmbeddingChunk {
    pub document_id: String,
    pub chunk_id: String,
    pub text: String,
}

/// Result returned after applying binding-provided embeddings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextApplyResult {
    pub mutation: CoreMutationResult,
    pub embedded_chunks: usize,
    pub reused_chunks: usize,
    pub embedding_time_ms: f64,
}

/// Prepared text query returned to a binding before query embedding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryPlan {
    pub query: String,
    pub mode: String,
    pub query_tokens: Vec<String>,
    pub requires_embedding: bool,
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
