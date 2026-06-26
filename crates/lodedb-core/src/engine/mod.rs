//! Native core engine.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{CoreError, CoreErrorCode};
use crate::filter::{build_field_indexes, coerce_sdk_filter, resolve_filter};
use crate::lexical::tokenize;
use crate::text::chunk::{chunk_id_for_hash, chunk_text};
use crate::text::hash::normalized_chunk_hash;
use crate::types::{
    CoreDocument, CoreMetadata, CoreMutationResult, CoreOpenOptions, CoreSearchHit,
    CoreSearchResults, CoreVectorDocument,
};
use crate::version::{CORE_VERSION, STORAGE_SCHEMA_VERSION};

/// In-memory vector-only native core engine.
#[derive(Debug, Default)]
pub struct CoreEngine {
    indexes: BTreeMap<String, VectorOnlyIndex>,
    next_plan_id: u64,
    persistence: Option<PersistenceState>,
}

impl CoreEngine {
    /// Creates an empty in-memory engine. No files are read or written.
    pub fn new_in_memory() -> Self {
        Self::default()
    }

    /// Opens a writable persistent engine and replays WAL tails when requested.
    pub fn open(options: CoreOpenOptions) -> Result<Self, CoreError> {
        let path = PathBuf::from(&options.path);
        fs::create_dir_all(&path).map_err(core_io_error)?;
        let lock = PersistentLock::acquire(&path)?;
        let mut engine = Self {
            indexes: BTreeMap::new(),
            next_plan_id: 0,
            persistence: Some(PersistenceState {
                path,
                read_only: false,
                fsync: options.durability == "fsync",
                store_text: options.store_text,
                index_text: options.index_text,
                _lock: Some(lock),
            }),
        };
        engine.load_persisted_indexes(options.commit_mode == "wal")?;
        Ok(engine)
    }

    /// Opens a lock-free read-only generation snapshot. WAL tails are ignored.
    pub fn open_readonly(
        path: impl AsRef<Path>,
        mut options: CoreOpenOptions,
    ) -> Result<Self, CoreError> {
        options.path = path.as_ref().to_string_lossy().to_string();
        options.read_only = true;
        let mut engine = Self {
            indexes: BTreeMap::new(),
            next_plan_id: 0,
            persistence: Some(PersistenceState {
                path: path.as_ref().to_path_buf(),
                read_only: true,
                fsync: options.durability == "fsync",
                store_text: options.store_text,
                index_text: options.index_text,
                _lock: None,
            }),
        };
        engine.load_persisted_indexes(false)?;
        Ok(engine)
    }

    /// Creates a vector-only index.
    pub fn create_index(
        &mut self,
        index_id: impl Into<String>,
        vector_dim: usize,
        bit_width: usize,
    ) -> Result<(), CoreError> {
        self.require_writable()?;
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
        self.require_writable()?;
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
        self.require_writable()?;
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
        self.require_writable()?;
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
        self.require_writable()?;
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
        self.require_writable()?;
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

    /// Persists every open index through generation-mode storage.
    pub fn persist(&mut self) -> Result<(), CoreError> {
        let Some(persistence) = &self.persistence else {
            return Ok(());
        };
        if persistence.read_only {
            return invalid("read-only engine cannot persist");
        }
        for index in self.indexes.values() {
            let state = state_payload_for_index(index);
            let raw_text = if persistence.store_text {
                index
                    .documents
                    .iter()
                    .filter_map(|(document_id, record)| {
                        record
                            .text
                            .as_ref()
                            .map(|text| (document_id.clone(), text.clone()))
                    })
                    .collect::<BTreeMap<_, _>>()
            } else {
                BTreeMap::new()
            };
            let lexical_tokens = if persistence.index_text {
                index
                    .documents
                    .iter()
                    .filter(|(_, record)| !record.token_lists.is_empty())
                    .map(|(document_id, record)| (document_id.clone(), record.token_lists.clone()))
                    .collect::<BTreeMap<_, _>>()
            } else {
                BTreeMap::new()
            };
            crate::storage::write_generation_commit(
                &persistence.path,
                crate::storage::GenerationCommitInput {
                    index_key: &index.index_id,
                    generation: index.generation.max(1),
                    base_epoch: index.generation.max(1),
                    state: &state,
                    tvim: None,
                    raw_text: Some(&raw_text),
                    lexical_tokens: Some(&lexical_tokens),
                },
                crate::storage::GenerationWriteOptions {
                    fsync: persistence.fsync,
                    retained_epochs: 4,
                },
            )?;
        }
        Ok(())
    }

    /// Persists a writable engine and releases its writer lock.
    pub fn close(&mut self) -> Result<(), CoreError> {
        if matches!(
            self.persistence.as_ref(),
            Some(persistence) if !persistence.read_only
        ) {
            self.persist()?;
        }
        self.persistence = None;
        Ok(())
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

    fn require_writable(&self) -> Result<(), CoreError> {
        if matches!(
            self.persistence.as_ref(),
            Some(persistence) if persistence.read_only
        ) {
            return invalid("engine is open read-only");
        }
        Ok(())
    }

    fn load_persisted_indexes(&mut self, replay_wal: bool) -> Result<(), CoreError> {
        let Some(persistence) = &self.persistence else {
            return Ok(());
        };
        if !persistence.path.is_dir() {
            return Ok(());
        }
        let mut index_keys = Vec::new();
        for entry in fs::read_dir(&persistence.path).map_err(core_io_error)? {
            let entry = entry.map_err(core_io_error)?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            if let Some(index_key) = name.strip_suffix(".commit.json") {
                index_keys.push(index_key.to_string());
            }
        }
        for index_key in index_keys {
            let mut loaded = crate::storage::load_store(
                &persistence.path,
                &index_key,
                crate::storage::LoadOptions {
                    read_only: persistence.read_only,
                    read_wal: replay_wal && !persistence.read_only,
                },
            )?;
            if replay_wal && !persistence.read_only {
                let records = crate::storage::wal::read_records(&crate::storage::wal::wal_path(
                    &persistence.path,
                    &index_key,
                ))?;
                if !records.is_empty() {
                    crate::storage::wal::replay_records_onto_store(&mut loaded, &records, 8192)?;
                    crate::storage::wal::checkpoint_store(
                        &persistence.path,
                        &loaded,
                        loaded.generation + 1,
                        persistence.fsync,
                    )?;
                    loaded.generation += 1;
                }
            }
            let index = index_from_loaded_store(&loaded)?;
            self.indexes.insert(index.index_id.clone(), index);
        }
        Ok(())
    }
}

#[derive(Debug)]
struct PersistenceState {
    path: PathBuf,
    read_only: bool,
    fsync: bool,
    store_text: bool,
    index_text: bool,
    _lock: Option<PersistentLock>,
}

#[derive(Debug)]
struct PersistentLock {
    path: PathBuf,
    _file: File,
}

impl PersistentLock {
    fn acquire(path: &Path) -> Result<Self, CoreError> {
        let lock_path = path.join(".lodedb.native.lock");
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&lock_path)
            .map_err(|error| {
                CoreError::new(
                    CoreErrorCode::InvalidArgument,
                    format!("could not acquire native writer lock: {error}"),
                )
            })?;
        Ok(Self {
            path: lock_path,
            _file: file,
        })
    }
}

impl Drop for PersistentLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn index_from_loaded_store(
    loaded: &crate::storage::LoadedStore,
) -> Result<VectorOnlyIndex, CoreError> {
    let state = loaded
        .state
        .as_object()
        .ok_or_else(|| invalid_err("loaded state must be a JSON object"))?;
    let index_id = state
        .get("index_id")
        .and_then(Value::as_str)
        .unwrap_or(&loaded.index_key)
        .to_string();
    let vector_dim = state.get("native_dim").and_then(Value::as_u64).unwrap_or(1) as usize;
    let bit_width = state
        .get("turbovec_bit_width")
        .and_then(Value::as_u64)
        .unwrap_or(4) as usize;
    validate_index_shape(vector_dim, bit_width)?;
    let metadata_by_id = state
        .get("document_metadata")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let chunk_ids_by_doc = state
        .get("document_chunk_ids")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let mut documents = BTreeMap::new();
    for document_id in state
        .get("document_hashes")
        .and_then(Value::as_object)
        .map(|object| object.keys().cloned().collect::<Vec<_>>())
        .unwrap_or_default()
    {
        let metadata = metadata_by_id
            .get(&document_id)
            .and_then(Value::as_object)
            .map(|object| {
                object
                    .iter()
                    .map(|(key, value)| (key.clone(), value.as_str().unwrap_or("").to_string()))
                    .collect()
            })
            .unwrap_or_default();
        let chunks = chunk_ids_by_doc
            .get(&document_id)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(|chunk_id| ChunkRecord {
                chunk_id: chunk_id.to_string(),
                vector: vec![0.0; vector_dim],
            })
            .collect();
        documents.insert(
            document_id.clone(),
            DocumentRecord {
                metadata,
                text: loaded.raw_text.get(&document_id).cloned(),
                token_lists: loaded
                    .lexical_tokens
                    .get(&document_id)
                    .cloned()
                    .unwrap_or_default(),
                chunks,
            },
        );
    }
    Ok(VectorOnlyIndex {
        index_id,
        vector_dim,
        bit_width,
        documents,
        generation: loaded.generation,
        delete_count: state
            .get("delete_count")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize,
        deleted_chunk_count: state
            .get("deleted_chunk_count")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize,
    })
}

fn state_payload_for_index(index: &VectorOnlyIndex) -> Value {
    let mut chunks = Vec::new();
    let mut document_hashes = serde_json::Map::new();
    let mut document_chunk_ids = serde_json::Map::new();
    let mut document_metadata = serde_json::Map::new();
    for (document_id, record) in &index.documents {
        let content_hash = record
            .text
            .as_ref()
            .map(|text| crate::text::hash::sha256_text(text))
            .unwrap_or_else(|| crate::text::hash::sha256_text(document_id));
        document_hashes.insert(document_id.clone(), Value::String(content_hash));
        document_chunk_ids.insert(
            document_id.clone(),
            Value::Array(
                record
                    .chunks
                    .iter()
                    .map(|chunk| Value::String(chunk.chunk_id.clone()))
                    .collect(),
            ),
        );
        document_metadata.insert(
            document_id.clone(),
            Value::Object(
                record
                    .metadata
                    .iter()
                    .map(|(key, value)| (key.clone(), Value::String(value.clone())))
                    .collect(),
            ),
        );
        for chunk in &record.chunks {
            chunks.push(serde_json::json!({
                "chunk_id": chunk.chunk_id,
                "content_hash": crate::text::hash::sha256_text(&chunk.chunk_id),
                "document_id": document_id,
            }));
        }
    }
    chunks.sort_by(|left, right| {
        left.get("chunk_id")
            .and_then(Value::as_str)
            .cmp(&right.get("chunk_id").and_then(Value::as_str))
    });
    serde_json::json!({
        "cache_reuse_count": 0,
        "chunks": chunks,
        "client_id_hash": index.index_id,
        "columnar_generation": index.generation.max(1),
        "created_at": "1970-01-01T00:00:00+00:00",
        "delete_count": index.delete_count,
        "deleted_chunk_count": index.deleted_chunk_count,
        "document_chunk_ids": document_chunk_ids,
        "document_hashes": document_hashes,
        "document_metadata": document_metadata,
        "embedded_chunk_count": index.chunk_count(),
        "fallback_count": 0,
        "fallback_reasons": {},
        "index_id": index.index_id,
        "index_key": index.index_id,
        "metadata": {},
        "model": "native-core",
        "name": "lodedb-local",
        "native_dim": index.vector_dim,
        "provider": "native",
        "query_count": 0,
        "route_profile": "native-core",
        "schema_version": 1,
        "status": "ready",
        "storage_profile": "native-core",
        "task": "native-core",
        "turbovec_bit_width": index.bit_width,
        "updated_at": "1970-01-01T00:00:00+00:00"
    })
}

fn core_io_error(error: std::io::Error) -> CoreError {
    CoreError::new(
        CoreErrorCode::Internal,
        format!("persistent engine I/O failed: {error}"),
    )
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
