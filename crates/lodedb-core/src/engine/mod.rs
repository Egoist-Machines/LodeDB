//! Native core engine.

use std::cell::{Ref, RefCell};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use turbovec::IdMapIndex;

use crate::error::{CoreError, CoreErrorCode};
use crate::filter::doc_set::DocSet;
use crate::filter::{build_field_indexes, coerce_sdk_filter, resolve_filter, FieldIndex};
use crate::lexical::rrf::RRF_C;
use crate::lexical::{tokenize, Bm25Index};
use crate::text::chunk::{chunk_id_for_hash, chunk_text};
use crate::text::hash::normalized_chunk_hash;
use crate::types::{
    CoreDocument, CoreIndexCreateOptions, CoreMetadata, CoreMutationResult, CoreOpenOptions,
    CoreSearchHit, CoreSearchResults, CoreVectorDocument,
};
use crate::vector::index::CoreVectorChunk;
use crate::vector::stable_id::stable_uint64_ids_for_chunk_ids;
use crate::vector::turbovec::TurboVecNativeIndex;
use crate::version::{CORE_VERSION, STORAGE_SCHEMA_VERSION};

const LEXICAL_POOL_FACTOR: usize = 5;
const LEXICAL_POOL_FLOOR: usize = 50;

/// In-memory vector-only native core engine.
#[derive(Default)]
pub struct CoreEngine {
    indexes: BTreeMap<String, VectorOnlyIndex>,
    next_plan_id: u64,
    persistence: Option<PersistenceState>,
    replaying_wal: bool,
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
        let lock = if options.acquire_writer_lock {
            Some(PersistentLock::acquire(&path)?)
        } else {
            None
        };
        let mut engine = Self {
            indexes: BTreeMap::new(),
            next_plan_id: 0,
            persistence: Some(PersistenceState {
                path,
                read_only: false,
                fsync: options.durability == "fsync",
                commit_mode: options.commit_mode.clone(),
                store_text: options.store_text,
                index_text: options.index_text,
                chunk_character_limit: options.chunk_character_limit,
                _lock: lock,
            }),
            replaying_wal: false,
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
                commit_mode: options.commit_mode.clone(),
                store_text: options.store_text,
                index_text: options.index_text,
                chunk_character_limit: options.chunk_character_limit,
                _lock: None,
            }),
            replaying_wal: false,
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
        self.create_index_with_options(CoreIndexCreateOptions::native_default(
            index_id, vector_dim, bit_width,
        ))
    }

    /// Creates a vector-only index with explicit persisted metadata.
    pub fn create_index_with_options(
        &mut self,
        options: CoreIndexCreateOptions,
    ) -> Result<(), CoreError> {
        self.require_writable()?;
        validate_index_shape(options.vector_dim, options.bit_width)?;
        validate_index_options(&options)?;
        if self.indexes.contains_key(&options.index_id)
            || self
                .indexes
                .values()
                .any(|index| index.index_key == options.index_key)
        {
            return invalid("index already exists");
        }
        self.indexes
            .insert(options.index_id.clone(), VectorOnlyIndex::new(options));
        Ok(())
    }

    /// Upserts vector documents into an existing index.
    pub fn upsert_vectors(
        &mut self,
        index_id: &str,
        documents: &[CoreVectorDocument],
    ) -> Result<CoreMutationResult, CoreError> {
        self.require_writable()?;
        // Capture the raw-text/lexical retention policy before borrowing the
        // index (it lives on the persistence options).
        let (store_text, index_text) = self.text_capture_policy();
        let index = self.index_mut(index_id)?;
        index.require_vectors_mutable()?;
        // Validate the whole batch before mutating any state. A bad row found
        // mid-loop must not leave earlier rows inserted in `documents` while the
        // live TurboVec index and generation/WAL stay unwritten, which would make
        // a failed request partially visible and committable on the next persist.
        for document in documents {
            if document.document_id.trim().is_empty() {
                return invalid("document_id is required");
            }
            if document.vector.len() != index.vector_dim {
                return invalid("vector dimension does not match index");
            }
            // Reject NaN / Inf / out-of-range coordinates here, at the core, so
            // every binding (PyO3 array upsert, FFI, Swift) is covered. The JSON
            // path cannot carry NaN, but the raw-array paths can, and a poisoned
            // row makes later TurboVec searches fail. Same finiteness contract as
            // TurboVec's own input check.
            if turbovec::first_invalid_coord(&document.vector, index.vector_dim).is_some() {
                return invalid("vector contains a non-finite or out-of-range value");
            }
        }
        let mut changed = 0usize;
        let mut chunks_upserted = 0usize;
        let mut changed_filter_fields = BTreeSet::new();
        // Collect every upserted chunk and sync the live TurboVec index once after
        // the loop instead of one `upsert_with_ids_2d` re-encode per document. The
        // result is identical (still O(changed)); the single batched encode avoids
        // n separate calibration-bound encode calls on a large add.
        let mut upserted_chunks: Vec<CoreVectorChunk> = Vec::with_capacity(documents.len());
        for document in documents {
            let content_hash = crate::text::hash::sha256_f32_le(&document.vector);
            // Mirror Python's vector-in text policy: retain raw text only when
            // store_text is on, and tokenize the optional caption into the lexical
            // index only when index_text is on (a vector document is a single
            // chunk keyed by its document id, so its caption is one token list).
            let retained_text = if store_text {
                document.text.clone()
            } else {
                None
            };
            let caption_tokens: Vec<String> = if index_text {
                document.text.as_deref().map(tokenize).unwrap_or_default()
            } else {
                Vec::new()
            };
            let token_lists = if caption_tokens.is_empty() {
                Vec::new()
            } else {
                vec![caption_tokens.clone()]
            };
            // Unchanged-vector fast path (mirrors Python's `_ingest_vectors`): an
            // identical re-add is a full no-op (no generation bump, no delta), and
            // a same-vector/changed-metadata refresh updates document state without
            // re-encoding the vector or re-syncing TurboVec.
            let (vector_unchanged, fully_unchanged) =
                match index.documents.get(&document.document_id) {
                    Some(record) => {
                        let vector_unchanged = record.content_hash == content_hash;
                        let fully_unchanged = vector_unchanged
                            && record.metadata == document.metadata
                            && record.text == retained_text
                            && record.token_lists == token_lists;
                        (vector_unchanged, fully_unchanged)
                    }
                    None => (false, false),
                };
            if fully_unchanged {
                continue;
            }
            let chunks = vec![ChunkRecord {
                chunk_id: document.document_id.clone(),
                vector: document.vector.clone(),
            }];
            let old_record = index.documents.insert(
                document.document_id.clone(),
                DocumentRecord {
                    // A vector-in document hashes its float32 vector bytes, matching
                    // the Python writer's `_vector_content_hash`, so the persisted
                    // content hash is identical across writers: `list_documents`
                    // output agrees and re-adding the same vector is recognized as
                    // unchanged regardless of which engine authored the store.
                    content_hash,
                    metadata: document.metadata.clone(),
                    text: retained_text,
                    token_lists,
                    chunks: chunks.clone(),
                },
            );
            if let Some(old_record) = old_record {
                index.remove_document_indexes(
                    &document.document_id,
                    &old_record.metadata,
                    &old_record.chunks,
                    &mut changed_filter_fields,
                );
            }
            index.add_document_indexes(
                &document.document_id,
                &document.metadata,
                &chunks,
                &mut changed_filter_fields,
            );
            if caption_tokens.is_empty() {
                index.lexical_index.remove_group(&document.document_id);
            } else {
                index.lexical_index.replace_group(
                    &document.document_id,
                    &[(document.document_id.clone(), caption_tokens)],
                );
            }
            // Only re-sync the vector when it actually changed; a same-vector
            // metadata refresh keeps its existing live-index row.
            if !vector_unchanged {
                upserted_chunks.push(CoreVectorChunk::new(
                    document.document_id.clone(),
                    document.document_id.clone(),
                    document.vector.clone(),
                ));
                chunks_upserted += 1;
            }
            changed += 1;
        }
        // A document id repeated within one batch must collapse to its last
        // vector: the per-document state above already applied last-wins, and
        // TurboVec's batched upsert rejects duplicate ids in a single call.
        let mut last_pos_by_doc: BTreeMap<&str, usize> = BTreeMap::new();
        for (pos, chunk) in upserted_chunks.iter().enumerate() {
            last_pos_by_doc.insert(chunk.document_id.as_str(), pos);
        }
        if upserted_chunks.is_empty() {
            // No vector changed this batch (all reused / metadata-only); avoid even
            // building/touching the live index for a no-op sync.
        } else if last_pos_by_doc.len() == upserted_chunks.len() {
            index.sync_vector_index_upsert(&upserted_chunks)?;
        } else {
            let mut keep: Vec<usize> = last_pos_by_doc.into_values().collect();
            keep.sort_unstable();
            let deduped: Vec<CoreVectorChunk> = keep
                .into_iter()
                .map(|pos| upserted_chunks[pos].clone())
                .collect();
            index.sync_vector_index_upsert(&deduped)?;
        }
        if changed > 0 {
            index.finalize_filter_fields(&changed_filter_fields);
            index.generation += 1;
        }
        let generation = index.generation;
        let index_key = index.index_key.clone();
        if changed > 0 && self.should_append_wal() {
            self.append_wal_record(
                &index_key,
                "upsert_vectors",
                serde_json::json!({
                    "vectors": documents.iter().map(|document| {
                        // Never write raw text to the WAL when store_text is off
                        // (privacy). When index_text is on, write the derived
                        // caption tokens so replay can rebuild lexical postings
                        // without retaining the raw text on disk.
                        let text = if store_text {
                            serde_json::json!(document.text)
                        } else {
                            Value::Null
                        };
                        let tokens = match (index_text, document.text.as_deref()) {
                            (true, Some(text)) => serde_json::json!([tokenize(text)]),
                            _ => Value::Null,
                        };
                        serde_json::json!({
                            "document_id": document.document_id,
                            "vector": document.vector,
                            "metadata": document.metadata,
                            "text": text,
                            "tokens": tokens,
                        })
                    }).collect::<Vec<_>>()
                }),
            )?;
        }
        Ok(CoreMutationResult {
            documents_upserted: changed,
            documents_deleted: 0,
            chunks_upserted,
            chunks_deleted: 0,
            generation,
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
        index.require_vectors_mutable()?;
        let mut deleted = 0usize;
        let mut deleted_chunks = 0usize;
        let mut seen = BTreeSet::new();
        let mut changed_filter_fields = BTreeSet::new();
        for document_id in document_ids {
            if document_id.trim().is_empty() {
                return invalid("document_id is required");
            }
            if seen.insert(document_id.clone()) {
                let Some(record) = index.documents.remove(document_id) else {
                    continue;
                };
                index.remove_document_indexes(
                    document_id,
                    &record.metadata,
                    &record.chunks,
                    &mut changed_filter_fields,
                );
                index.lexical_index.remove_group(document_id);
                let chunk_ids = record
                    .chunks
                    .iter()
                    .map(|chunk| chunk.chunk_id.clone())
                    .collect::<Vec<_>>();
                index.sync_vector_index_remove(&chunk_ids);
                deleted += 1;
                deleted_chunks += record.chunks.len();
            }
        }
        if deleted > 0 {
            index.finalize_filter_fields(&changed_filter_fields);
            index.delete_count += deleted;
            index.deleted_chunk_count += deleted_chunks;
            index.generation += 1;
        }
        let generation = index.generation;
        let index_key = index.index_key.clone();
        if deleted > 0 {
            self.append_wal_record(
                &index_key,
                "delete_documents",
                serde_json::json!({
                    "document_ids": document_ids,
                }),
            )?;
        }
        Ok(CoreMutationResult {
            documents_upserted: 0,
            documents_deleted: deleted,
            chunks_upserted: 0,
            chunks_deleted: deleted_chunks,
            generation,
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
        // Capture the inputs for the WAL record before they are moved into the
        // document. `text` is a three-state Option<Option<String>> (unchanged /
        // clear / set), encoded below by key presence so replay is exact.
        let metadata_for_wal = metadata.clone();
        let text_for_wal = text.clone();
        // Same raw-text/lexical retention policy as upsert_vectors.
        let (store_text, index_text) = self.text_capture_policy();
        let index = self.index_mut(index_id)?;
        index.require_vectors_mutable()?;
        let Some(record) = index.documents.get(document_id) else {
            return invalid("document not found");
        };
        let old_metadata = record.metadata.clone();
        let chunks = record.chunks.clone();
        let metadata_changed = metadata.is_some();
        let mut changed_filter_fields = BTreeSet::new();
        if metadata_changed {
            index.remove_document_indexes(
                document_id,
                &old_metadata,
                &chunks,
                &mut changed_filter_fields,
            );
        }
        let record = index
            .documents
            .get_mut(document_id)
            .ok_or_else(|| invalid_err("document not found"))?;
        if let Some(metadata) = metadata {
            record.metadata = metadata;
        }
        if let Some(text) = text {
            record.content_hash = text
                .as_ref()
                .map(|text| crate::text::hash::sha256_text(text))
                .unwrap_or_else(|| crate::text::hash::sha256_text(document_id));
            // Retain raw text only when store_text is on (privacy); otherwise the
            // updated caption must not be kept in memory or written to disk.
            record.text = if store_text { text.clone() } else { None };
            // The caption changed, so refresh its lexical postings: tokenize the
            // new caption when index_text is on, otherwise clear stale postings.
            let caption_tokens: Vec<String> = if index_text {
                text.as_deref().map(tokenize).unwrap_or_default()
            } else {
                Vec::new()
            };
            record.token_lists = if caption_tokens.is_empty() {
                Vec::new()
            } else {
                vec![caption_tokens.clone()]
            };
            if caption_tokens.is_empty() {
                index.lexical_index.remove_group(document_id);
            } else {
                index
                    .lexical_index
                    .replace_group(document_id, &[(document_id.to_string(), caption_tokens)]);
            }
        }
        if metadata_changed {
            let metadata = record.metadata.clone();
            index.add_document_indexes(document_id, &metadata, &chunks, &mut changed_filter_fields);
            index.finalize_filter_fields(&changed_filter_fields);
        }
        index.generation += 1;
        let generation = index.generation;
        let index_key = index.index_key.clone();
        // Make payload-only updates crash-durable in WAL mode like the other
        // native mutations: encode metadata only when set, and the three text
        // states by key presence (absent = unchanged, null = clear, string = set).
        let mut payload = serde_json::Map::new();
        payload.insert(
            "document_id".to_string(),
            Value::String(document_id.to_string()),
        );
        if let Some(metadata) = &metadata_for_wal {
            payload.insert("metadata".to_string(), serde_json::json!(metadata));
        }
        if let Some(text) = &text_for_wal {
            // Privacy: write raw text to the WAL only when store_text is on. When
            // index_text is on, write the derived caption tokens so replay can
            // rebuild lexical postings without retaining the raw text.
            if store_text {
                payload.insert("text".to_string(), serde_json::json!(text));
            }
            if index_text {
                let tokens = text.as_deref().map(tokenize).unwrap_or_default();
                let token_lists: Vec<Vec<String>> = if tokens.is_empty() {
                    Vec::new()
                } else {
                    vec![tokens]
                };
                payload.insert("tokens".to_string(), serde_json::json!(token_lists));
            }
        }
        self.append_wal_record(
            &index_key,
            "update_document_payload",
            Value::Object(payload),
        )?;
        Ok(CoreMutationResult {
            documents_upserted: 1,
            documents_deleted: 0,
            chunks_upserted: 0,
            chunks_deleted: 0,
            generation,
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
        let index = self.index(index_id)?;
        index.require_vectors_mutable()?;
        let base_generation = index.generation;
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
                let tokens = if index_text || store_text {
                    tokenize(&piece)
                } else {
                    Vec::new()
                };
                // O(1) existence check via the maintained chunk->owner map rather
                // than cloning every chunk vector in the corpus just to test
                // membership (the latter made each incremental text add O(corpus)).
                let needs_embedding = !index.chunk_owner_by_id.contains_key(&chunk_id);
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
        // Decide once whether a WAL record is needed; when not (in-memory mirror,
        // generation write-through, read-only, or replay) we skip building the WAL
        // payload below, avoiding per-chunk embedding clones and JSON construction.
        let append_wal = self.should_append_wal();
        let index = self.index_mut(&plan.index_id)?;
        index.require_vectors_mutable()?;
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
            // Same core-level finiteness guard as upsert_vectors: a NaN/Inf
            // embedding from a binding's array path must not enter the index.
            if turbovec::first_invalid_coord(embedding, index.vector_dim).is_some() {
                return invalid("embedding contains a non-finite or out-of-range value");
            }
        }

        let mut new_embeddings = plan
            .chunks_to_embed
            .iter()
            .zip(embeddings)
            .map(|(chunk, embedding)| (chunk.chunk_id.clone(), embedding.clone()))
            .collect::<BTreeMap<_, _>>();
        // Reusable chunk vectors for chunks the plan keeps but does not re-embed.
        // Fetch only those (via the O(1) chunk->owner map) instead of cloning the
        // whole corpus's vectors, which made each incremental text add O(corpus).
        let existing_chunks = reusable_chunk_vectors(
            index,
            plan.documents
                .iter()
                .flat_map(|document| document.chunks.iter().map(|chunk| chunk.chunk_id.as_str())),
            &new_embeddings,
        );
        let mut chunks_upserted = 0usize;
        let mut reused_chunks = 0usize;
        let mut removed_chunk_ids = BTreeSet::new();
        let mut active_chunk_ids = BTreeSet::new();
        let mut added_chunks = Vec::new();
        let mut added_vector_chunks = Vec::new();
        let mut wal_documents = Vec::new();
        let mut changed_filter_fields = BTreeSet::new();
        for document in &plan.documents {
            let mut chunks = Vec::with_capacity(document.chunks.len());
            let token_lists = document
                .chunks
                .iter()
                .map(|chunk| chunk.tokens.clone())
                .collect::<Vec<_>>();
            let lexical_units = document
                .chunks
                .iter()
                .map(|chunk| (chunk.chunk_id.clone(), chunk.tokens.clone()))
                .collect::<Vec<_>>();
            for chunk in &document.chunks {
                let vector = if let Some(embedding) = new_embeddings.remove(&chunk.chunk_id) {
                    chunks_upserted += 1;
                    added_vector_chunks.push(CoreVectorChunk::new(
                        chunk.chunk_id.clone(),
                        document.document_id.clone(),
                        embedding.clone(),
                    ));
                    if append_wal {
                        added_chunks.push(serde_json::json!({
                            "chunk_id": chunk.chunk_id,
                            "document_id": document.document_id,
                            "content_hash": crate::text::hash::sha256_text(&chunk.text),
                            "embedding": embedding.clone(),
                        }));
                    }
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
                active_chunk_ids.insert(chunk.chunk_id.clone());
            }
            let content_hash = document
                .text
                .as_ref()
                .map(|text| crate::text::hash::sha256_text(text))
                .unwrap_or_else(|| {
                    crate::text::hash::sha256_text(
                        &document
                            .chunks
                            .iter()
                            .map(|chunk| chunk.text.as_str())
                            .collect::<Vec<_>>()
                            .join("\n"),
                    )
                });
            let old_record = index.documents.insert(
                document.document_id.clone(),
                DocumentRecord {
                    content_hash: content_hash.clone(),
                    metadata: document.metadata.clone(),
                    text: document.text.clone(),
                    token_lists: token_lists.clone(),
                    chunks: chunks.clone(),
                },
            );
            if let Some(old_record) = old_record {
                for chunk in &old_record.chunks {
                    removed_chunk_ids.insert(chunk.chunk_id.clone());
                }
                index.remove_document_indexes(
                    &document.document_id,
                    &old_record.metadata,
                    &old_record.chunks,
                    &mut changed_filter_fields,
                );
            }
            index.add_document_indexes(
                &document.document_id,
                &document.metadata,
                &chunks,
                &mut changed_filter_fields,
            );
            index
                .lexical_index
                .replace_group(&document.document_id, &lexical_units);
            if append_wal {
                wal_documents.push(serde_json::json!({
                    "document_id": document.document_id,
                    "content_hash": content_hash,
                    "metadata": document.metadata,
                    "text": document.text,
                    "chunk_ids": chunks.iter().map(|chunk| chunk.chunk_id.clone()).collect::<Vec<_>>(),
                    "tokens": token_lists,
                }));
            }
        }
        if !plan.documents.is_empty() {
            let removed_chunk_ids_vec = removed_chunk_ids
                .iter()
                .filter(|chunk_id| !active_chunk_ids.contains(*chunk_id))
                .cloned()
                .collect::<Vec<_>>();
            index.sync_vector_index_remove(&removed_chunk_ids_vec);
            index.sync_vector_index_upsert(&added_vector_chunks)?;
            index.finalize_filter_fields(&changed_filter_fields);
            index.generation += 1;
        }
        let generation = index.generation;
        let index_key = index.index_key.clone();
        if append_wal && !plan.documents.is_empty() {
            self.append_wal_record(
                &index_key,
                "apply_embedded_documents",
                serde_json::json!({
                    "documents": wal_documents,
                    "added_chunks": added_chunks,
                    "removed_chunk_ids": removed_chunk_ids.into_iter().collect::<Vec<_>>(),
                }),
            )?;
        }
        Ok(TextApplyResult {
            mutation: CoreMutationResult {
                documents_upserted: plan.documents.len(),
                documents_deleted: 0,
                chunks_upserted,
                chunks_deleted: 0,
                generation,
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
        match query_plan.mode.as_str() {
            "vector" => {
                let embedding = query_embedding
                    .ok_or_else(|| invalid_err("query embedding is required for this mode"))?;
                self.query_vector(index_id, embedding, top_k, filter)
            }
            "lexical" => self.query_lexical_text(index_id, query_plan, top_k, filter),
            "hybrid" => {
                let embedding = query_embedding
                    .ok_or_else(|| invalid_err("query embedding is required for this mode"))?;
                let pool = lexical_pool_width(top_k);
                let vector_results = self.query_vector(index_id, embedding, pool, filter)?;
                let lexical_results =
                    self.query_lexical_text(index_id, query_plan, pool, filter)?;
                let hits = fuse_hybrid_hits(vector_results.hits, lexical_results.hits, top_k);
                Ok(CoreSearchResults {
                    hits,
                    total_considered: vector_results.total_considered,
                })
            }
            _ => invalid("unsupported query mode"),
        }
    }

    fn query_lexical_text(
        &self,
        index_id: &str,
        query_plan: &QueryPlan,
        top_k: usize,
        filter: Option<&Value>,
    ) -> Result<CoreSearchResults, CoreError> {
        if top_k == 0 {
            return invalid("top_k must be positive");
        }
        let index = self.index(index_id)?;
        let candidates = filter
            .map(|filter| index.resolve_filter(Some(filter)))
            .transpose()?;
        let total_considered = candidates
            .as_ref()
            .map_or(index.all_docs.len(), BTreeSet::len);
        let allowed_positions = candidates.as_ref().map(|candidates| {
            let mut positions = BTreeSet::new();
            for document_id in candidates {
                let Some(record) = index.documents.get(document_id) else {
                    continue;
                };
                for chunk in &record.chunks {
                    if let Some(position) = index.lexical_index.position_of(&chunk.chunk_id) {
                        positions.insert(position);
                    }
                }
            }
            positions
        });
        let hits = index
            .lexical_index
            .rank(&query_plan.query, Some(top_k), allowed_positions.as_ref())
            .into_iter()
            .filter_map(|(chunk_id, score)| {
                index
                    .document_for_chunk(&chunk_id)
                    .map(|(document_id, record)| CoreSearchHit {
                        document_id,
                        chunk_id,
                        score: score as f32,
                        metadata: record.metadata.clone(),
                    })
            })
            .collect();
        Ok(CoreSearchResults {
            hits,
            total_considered,
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
        index.require_vectors_seeded()?;
        if index.chunk_count() == 0 {
            return Ok(CoreSearchResults {
                hits: Vec::new(),
                total_considered: 0,
            });
        }
        match index.query_vector_turbovec(query_vector, top_k, filter) {
            Ok(results) => Ok(results),
            Err(error) if error.code() == CoreErrorCode::Unsupported => {
                self.query_vector_scalar(index_id, query_vector, top_k, filter)
            }
            Err(error) => Err(error),
        }
    }

    fn query_vector_scalar(
        &self,
        index_id: &str,
        query_vector: &[f32],
        top_k: usize,
        filter: Option<&Value>,
    ) -> Result<CoreSearchResults, CoreError> {
        let index = self.index(index_id)?;
        let rotated_query;
        let query = if let Some(rotation) = &index.query_rotation {
            rotated_query = rotate_query(query_vector, rotation, index.vector_dim)?;
            rotated_query.as_slice()
        } else {
            query_vector
        };
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
                        score: dot(query, &chunk.vector),
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
        if top_k == 0 {
            return invalid("top_k must be positive");
        }
        let index = self.index(index_id)?;
        index.require_vectors_seeded()?;
        for query in query_vectors {
            if query.len() != index.vector_dim {
                return invalid("query dimension does not match index");
            }
        }
        if index.chunk_count() == 0 {
            return Ok(query_vectors
                .iter()
                .map(|_| CoreSearchResults {
                    hits: Vec::new(),
                    total_considered: 0,
                })
                .collect());
        }
        match index.query_vectors_batch_turbovec(query_vectors, top_k, filter) {
            Ok(results) => Ok(results),
            Err(error) if error.code() == CoreErrorCode::Unsupported => query_vectors
                .iter()
                .map(|query| self.query_vector_scalar(index_id, query, top_k, filter))
                .collect(),
            Err(error) => Err(error),
        }
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

    /// Returns one stored raw-text payload by document id.
    pub fn get_document_text(
        &self,
        index_id: &str,
        document_id: &str,
    ) -> Result<Option<String>, CoreError> {
        if document_id.trim().is_empty() {
            return invalid("document_id is required");
        }
        let index = self.index(index_id)?;
        Ok(index
            .documents
            .get(document_id)
            .and_then(|record| record.text.clone()))
    }

    /// Returns stored raw-text payloads for the requested document ids.
    pub fn get_document_texts(
        &self,
        index_id: &str,
        document_ids: &[String],
    ) -> Result<BTreeMap<String, String>, CoreError> {
        let index = self.index(index_id)?;
        let mut out = BTreeMap::new();
        for document_id in document_ids {
            if document_id.trim().is_empty() {
                return invalid("document_id is required");
            }
            if let Some(text) = index
                .documents
                .get(document_id)
                .and_then(|record| record.text.as_ref())
            {
                out.insert(document_id.clone(), text.clone());
            }
        }
        Ok(out)
    }

    /// Returns one payload-free document record.
    pub fn get_document(
        &self,
        index_id: &str,
        document_id: &str,
    ) -> Result<Option<Value>, CoreError> {
        if document_id.trim().is_empty() {
            return invalid("document_id is required");
        }
        let index = self.index(index_id)?;
        Ok(index
            .documents
            .get(document_id)
            .map(|record| document_resource_payload(document_id, record)))
    }

    /// Lists payload-free document records, optionally through the metadata/doc-id planner.
    pub fn list_documents(
        &self,
        index_id: &str,
        filter: Option<&Value>,
    ) -> Result<Vec<Value>, CoreError> {
        let index = self.index(index_id)?;
        let document_ids = index.resolve_filter(filter)?;
        Ok(document_ids
            .into_iter()
            .filter_map(|document_id| {
                index
                    .documents
                    .get(&document_id)
                    .map(|record| document_resource_payload(&document_id, record))
            })
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
        let dir = persistence.path.clone();
        let fsync = persistence.fsync;
        let store_text = persistence.store_text;
        let index_text = persistence.index_text;
        let commit_mode = persistence.commit_mode.clone();
        for index in self.indexes.values_mut() {
            persist_index_generation(index, &dir, fsync, store_text, index_text)?;
            if commit_mode == "wal" {
                crate::storage::wal::truncate(
                    &crate::storage::wal::wal_path(&dir, &index.index_key),
                    fsync,
                )?;
            }
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

    /// The raw-text (store_text) and lexical-token (index_text) retention policy.
    /// Persistent engines carry it on their open options; an in-memory engine has
    /// no on-disk privacy surface, so it captures both (the binding still gates
    /// what it mirrors).
    fn text_capture_policy(&self) -> (bool, bool) {
        match &self.persistence {
            Some(persistence) => (persistence.store_text, persistence.index_text),
            None => (true, true),
        }
    }

    /// Whether a mutation should stage a WAL record. False for in-memory engines,
    /// read-only handles, WAL replay, and non-WAL commit modes -- callers check
    /// this before building the (potentially large) WAL payload so they do not
    /// clone vectors/JSON only to discard them (e.g. the default in-memory native
    /// mirror or a generation write-through writer).
    fn should_append_wal(&self) -> bool {
        match &self.persistence {
            Some(persistence) => {
                !self.replaying_wal && !persistence.read_only && persistence.commit_mode == "wal"
            }
            None => false,
        }
    }

    fn append_wal_record(
        &self,
        index_key: &str,
        op: &str,
        payload: Value,
    ) -> Result<(), CoreError> {
        let Some(persistence) = &self.persistence else {
            return Ok(());
        };
        if self.replaying_wal || persistence.read_only || persistence.commit_mode != "wal" {
            return Ok(());
        }
        crate::storage::wal::append_record(
            &crate::storage::wal::wal_path(&persistence.path, index_key),
            op,
            &payload,
            persistence.fsync,
        )?;
        Ok(())
    }

    fn apply_native_wal_record(
        &mut self,
        index_id: &str,
        record: &crate::storage::wal::WalRecord,
    ) -> Result<(), CoreError> {
        match record.op.as_str() {
            "upsert_vectors" => {
                let documents = record
                    .payload
                    .get("vectors")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .map(vector_document_from_wal)
                    .collect::<Result<Vec<_>, _>>()?;
                self.upsert_vectors(index_id, &documents)?;
                // Restore lexical caption tokens captured in the WAL. Needed when
                // store_text was off (so raw text was not written and upsert_vectors
                // re-derived no tokens) but index_text retained the caption tokens.
                if let Some(vectors) = record.payload.get("vectors").and_then(Value::as_array) {
                    self.restore_wal_vector_tokens(index_id, vectors)?;
                }
            }
            "delete_documents" => {
                let document_ids = record
                    .payload
                    .get("document_ids")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                    .map(ToString::to_string)
                    .collect::<Vec<_>>();
                self.delete_documents(index_id, &document_ids)?;
            }
            "apply_embedded_documents" => {
                self.apply_embedded_documents_wal(index_id, &record.payload)?;
            }
            "update_document_payload" => {
                let document_id = record
                    .payload
                    .get("document_id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid_err("wal update_document_payload missing document_id"))?
                    .to_string();
                // Key presence distinguishes the three states: an absent
                // "metadata"/"text" key means "leave unchanged"; a present "text"
                // with null means "clear".
                let metadata = record.payload.get("metadata").map(metadata_from_value);
                let text = match record.payload.get("text") {
                    Some(Value::Null) => Some(None),
                    Some(Value::String(text)) => Some(Some(text.clone())),
                    Some(_) => {
                        return Err(invalid_err(
                            "wal update_document_payload text must be a string or null",
                        ))
                    }
                    None => None,
                };
                self.update_document_payload(index_id, &document_id, metadata, text)?;
                // Restore caption tokens captured in the WAL (needed when
                // store_text was off, so the raw text was not written and the
                // update above re-derived no tokens, but index_text retained them).
                if record.payload.get("tokens").is_some() {
                    let entry = serde_json::json!({
                        "document_id": document_id,
                        "tokens": record.payload.get("tokens").cloned().unwrap_or(Value::Null),
                    });
                    self.restore_wal_vector_tokens(index_id, std::slice::from_ref(&entry))?;
                }
            }
            other => {
                return Err(CoreError::new(
                    CoreErrorCode::Unsupported,
                    format!("native WAL replay does not support {other}"),
                ));
            }
        }
        Ok(())
    }

    /// Restores per-document lexical token lists for replayed vector upserts.
    /// Each vector document is a single chunk keyed by its document id.
    fn restore_wal_vector_tokens(
        &mut self,
        index_id: &str,
        vectors: &[Value],
    ) -> Result<(), CoreError> {
        let index = self.index_mut(index_id)?;
        for entry in vectors {
            let Some(document_id) = entry.get("document_id").and_then(Value::as_str) else {
                continue;
            };
            let token_lists: Vec<Vec<String>> = match entry.get("tokens").and_then(Value::as_array)
            {
                Some(arrays) => arrays
                    .iter()
                    .map(|inner| {
                        inner
                            .as_array()
                            .into_iter()
                            .flatten()
                            .filter_map(Value::as_str)
                            .map(ToString::to_string)
                            .collect()
                    })
                    .collect(),
                None => continue,
            };
            if let Some(record) = index.documents.get_mut(document_id) {
                record.token_lists = token_lists.clone();
            } else {
                continue;
            }
            let units = vec![(
                document_id.to_string(),
                token_lists.into_iter().next().unwrap_or_default(),
            )];
            index.lexical_index.replace_group(document_id, &units);
        }
        Ok(())
    }

    fn apply_embedded_documents_wal(
        &mut self,
        index_id: &str,
        payload: &Value,
    ) -> Result<(), CoreError> {
        self.require_writable()?;
        let index = self.index_mut(index_id)?;
        index.require_vectors_mutable()?;
        let removed_chunk_ids = payload
            .get("removed_chunk_ids")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(ToString::to_string)
            .collect::<BTreeSet<_>>();
        let added_vectors = payload
            .get("added_chunks")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|chunk| {
                let chunk_id = chunk
                    .get("chunk_id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid_err("WAL added chunk missing chunk_id"))?
                    .to_string();
                let vector = chunk
                    .get("embedding")
                    .and_then(Value::as_array)
                    .ok_or_else(|| invalid_err("WAL added chunk missing embedding"))?
                    .iter()
                    .map(|value| value.as_f64().unwrap_or(0.0) as f32)
                    .collect::<Vec<_>>();
                if vector.len() != index.vector_dim {
                    return invalid("WAL added chunk embedding dimension does not match index");
                }
                Ok((chunk_id, vector))
            })
            .collect::<Result<BTreeMap<_, _>, CoreError>>()?;
        let documents = payload
            .get("documents")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid_err("WAL embedded payload missing documents"))?;
        // Reuse vectors only for the chunks these documents reference that are not
        // in this record's added_chunks, via the O(1) owner map, rather than
        // cloning the whole corpus on every replayed record.
        let existing_chunks = reusable_chunk_vectors(
            index,
            documents.iter().flat_map(|document| {
                document
                    .get("chunk_ids")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
            }),
            &added_vectors,
        );
        let mut changed_filter_fields = BTreeSet::new();
        let mut active_chunk_ids = BTreeSet::new();
        for document in documents {
            let document_id = document
                .get("document_id")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid_err("WAL embedded document missing document_id"))?
                .to_string();
            let metadata = metadata_from_value(document.get("metadata").unwrap_or(&Value::Null));
            let text = document
                .get("text")
                .and_then(Value::as_str)
                .map(ToString::to_string);
            let token_lists = document
                .get("tokens")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .map(|tokens| {
                    tokens
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter_map(Value::as_str)
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let chunk_ids = document
                .get("chunk_ids")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(ToString::to_string)
                .collect::<Vec<_>>();
            let mut chunks = Vec::with_capacity(chunk_ids.len());
            for chunk_id in chunk_ids {
                let vector = added_vectors
                    .get(&chunk_id)
                    .or_else(|| existing_chunks.get(&chunk_id))
                    .ok_or_else(|| invalid_err("WAL embedded payload references missing chunk"))?
                    .clone();
                chunks.push(ChunkRecord { chunk_id, vector });
            }
            active_chunk_ids.extend(chunks.iter().map(|chunk| chunk.chunk_id.clone()));
            let old_record = index.documents.insert(
                document_id.clone(),
                DocumentRecord {
                    content_hash: document
                        .get("content_hash")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    metadata: metadata.clone(),
                    text,
                    token_lists: token_lists.clone(),
                    chunks: chunks.clone(),
                },
            );
            if let Some(old_record) = old_record {
                index.remove_document_indexes(
                    &document_id,
                    &old_record.metadata,
                    &old_record.chunks,
                    &mut changed_filter_fields,
                );
            }
            index.add_document_indexes(
                &document_id,
                &metadata,
                &chunks,
                &mut changed_filter_fields,
            );
            let lexical_units = chunks
                .iter()
                .enumerate()
                .map(|(offset, chunk)| {
                    (
                        chunk.chunk_id.clone(),
                        token_lists.get(offset).cloned().unwrap_or_default(),
                    )
                })
                .collect::<Vec<_>>();
            index
                .lexical_index
                .replace_group(&document_id, &lexical_units);
        }
        let removed_chunk_ids_vec = removed_chunk_ids
            .iter()
            .filter(|chunk_id| !active_chunk_ids.contains(*chunk_id))
            .cloned()
            .collect::<Vec<_>>();
        if !removed_chunk_ids_vec.is_empty() {
            for record in index.documents.values_mut() {
                record
                    .chunks
                    .retain(|chunk| !removed_chunk_ids_vec.contains(&chunk.chunk_id));
            }
        }
        if !documents.is_empty() || !removed_chunk_ids.is_empty() {
            let added_vector_chunks = added_vectors
                .iter()
                .filter_map(|(chunk_id, vector)| {
                    index.chunk_owner_by_id.get(chunk_id).map(|document_id| {
                        CoreVectorChunk::new(chunk_id, document_id, vector.clone())
                    })
                })
                .collect::<Vec<_>>();
            index.sync_vector_index_remove(&removed_chunk_ids_vec);
            index.sync_vector_index_upsert(&added_vector_chunks)?;
            index.finalize_filter_fields(&changed_filter_fields);
            index.generation += 1;
        }
        Ok(())
    }

    fn load_persisted_indexes(&mut self, replay_wal: bool) -> Result<(), CoreError> {
        let Some(persistence) = &self.persistence else {
            return Ok(());
        };
        let persistence_path = persistence.path.clone();
        let persistence_read_only = persistence.read_only;
        let persistence_chunk_character_limit = persistence.chunk_character_limit;
        if !persistence_path.is_dir() {
            return Ok(());
        }
        let mut index_keys = Vec::new();
        for entry in fs::read_dir(&persistence_path).map_err(core_io_error)? {
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
            let loaded = crate::storage::load_store(
                &persistence_path,
                &index_key,
                crate::storage::LoadOptions {
                    read_only: persistence_read_only,
                    read_wal: false,
                },
            )?;
            if replay_wal && !persistence_read_only {
                let records = crate::storage::wal::read_records(&crate::storage::wal::wal_path(
                    &persistence_path,
                    &index_key,
                ))?;
                if !records.is_empty() {
                    if records.iter().all(is_native_replayable_wal_record) {
                        let index =
                            index_from_loaded_store(&loaded, persistence_chunk_character_limit)?;
                        let index_id = index.index_id.clone();
                        self.indexes.insert(index_id.clone(), index);
                        self.replaying_wal = true;
                        let replay_result = records
                            .iter()
                            .try_for_each(|record| self.apply_native_wal_record(&index_id, record));
                        self.replaying_wal = false;
                        replay_result?;
                        self.persist()?;
                        continue;
                    } else {
                        // The WAL contains records native cannot faithfully replay
                        // or checkpoint (e.g. Python `upsert_documents`, whose chunk
                        // vectors live in the state journal with no committed
                        // TurboVec snapshot to seed from). The previous code
                        // checkpointed such a WAL with `tvim: None` and truncated
                        // it, leaving a generation Python could no longer open
                        // ("direct TurboVec snapshot is required but missing").
                        //
                        // Fail closed and leave the WAL untouched on disk so the
                        // Python writer — which checkpoints its own WAL on open and
                        // writes a complete snapshot — stays the owner of these
                        // stores. In the Python SDK, Python opens (and checkpoints)
                        // before the native engine, so the native engine never
                        // reaches this branch; a native init failure here also
                        // falls back to the Python oracle.
                        return Err(CoreError::new(
                            CoreErrorCode::Unsupported,
                            "native writable open cannot checkpoint a WAL containing \
                             non-native records; open this store with the Python \
                             engine first",
                        ));
                    }
                }
            }
            let index = index_from_loaded_store(&loaded, persistence_chunk_character_limit)?;
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
    commit_mode: String,
    store_text: bool,
    index_text: bool,
    chunk_character_limit: usize,
    _lock: Option<PersistentLock>,
}

#[derive(Debug)]
struct PersistentLock {
    // The BSD advisory lock is released when this descriptor is dropped (closed);
    // the sentinel file is intentionally left in place, matching the Python lock.
    _file: File,
}

impl PersistentLock {
    /// Takes the shared single-writer lock on ``<dir>/.lodedb.lock`` — the same
    /// sentinel and BSD advisory-lock mechanism the Python writer uses
    /// (``fcntl.flock(LOCK_EX|LOCK_NB)``), so a standalone native/FFI/Swift writer
    /// contends with a Python writer (and another native writer) across processes.
    /// The lock releases when the file descriptor is dropped; the sentinel file is
    /// left in place, matching Python. Honours ``LODEDB_PERSIST_LOCK_TIMEOUT``.
    fn acquire(path: &Path) -> Result<Self, CoreError> {
        let lock_path = path.join(".lodedb.lock");
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .map_err(|error| {
                CoreError::new(
                    CoreErrorCode::InvalidArgument,
                    format!(
                        "could not open writer lock {}: {error}",
                        lock_path.display()
                    ),
                )
            })?;
        Self::lock_blocking(&file)?;
        Ok(Self { _file: file })
    }

    #[cfg(unix)]
    fn lock_blocking(file: &File) -> Result<(), CoreError> {
        use rustix::fs::{flock, FlockOperation};
        let deadline = Instant::now() + Duration::from_secs_f64(lock_timeout_seconds());
        loop {
            match flock(file, FlockOperation::NonBlockingLockExclusive) {
                Ok(()) => return Ok(()),
                Err(err)
                    if err == rustix::io::Errno::WOULDBLOCK || err == rustix::io::Errno::AGAIN =>
                {
                    if Instant::now() >= deadline {
                        return Err(CoreError::new(
                            CoreErrorCode::InvalidArgument,
                            "another writer holds the lodedb lock".to_string(),
                        ));
                    }
                    std::thread::sleep(Duration::from_millis(25));
                }
                Err(err) => {
                    return Err(CoreError::new(
                        CoreErrorCode::InvalidArgument,
                        format!("could not acquire writer lock: {err}"),
                    ));
                }
            }
        }
    }

    // Non-unix platforms do not have BSD flock; the Python SDK opens its native
    // engine with acquire_writer_lock=false (Python's own lock guards the store),
    // so this path only affects standalone non-unix native writers, which are not
    // a current target. Opening the sentinel is enough to keep behaviour uniform.
    #[cfg(not(unix))]
    fn lock_blocking(_file: &File) -> Result<(), CoreError> {
        Ok(())
    }
}

fn lock_timeout_seconds() -> f64 {
    match std::env::var("LODEDB_PERSIST_LOCK_TIMEOUT") {
        Ok(raw) => raw
            .trim()
            .parse::<f64>()
            .ok()
            .filter(|v| *v >= 0.0)
            .unwrap_or(30.0),
        Err(_) => 30.0,
    }
}

fn index_from_loaded_store(
    loaded: &crate::storage::LoadedStore,
    chunk_character_limit: usize,
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
    let index_key = state
        .get("index_key")
        .and_then(Value::as_str)
        .unwrap_or(&loaded.index_key)
        .to_string();
    let client_id_hash = state
        .get("client_id_hash")
        .and_then(Value::as_str)
        .unwrap_or(&index_key)
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
    let chunk_vectors = reconstruct_tvim_vectors(loaded, vector_dim)?;
    let vectors_seeded = loaded.chunk_count() == 0 || chunk_vectors.is_some();
    let (chunk_vectors, query_rotation) = chunk_vectors
        .map(|reconstructed| (reconstructed.vectors, reconstructed.query_rotation))
        .unwrap_or_default();
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
        let chunks: Vec<ChunkRecord> = chunk_ids_by_doc
            .get(&document_id)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(|chunk_id| ChunkRecord {
                chunk_id: chunk_id.to_string(),
                vector: chunk_vectors
                    .get(chunk_id)
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; vector_dim]),
            })
            .collect();
        let token_lists = loaded
            .lexical_tokens
            .get(&document_id)
            .cloned()
            .unwrap_or_else(|| {
                loaded
                    .raw_text
                    .get(&document_id)
                    .map(|text| token_lists_from_raw_text(text, chunk_character_limit))
                    .unwrap_or_default()
            });
        documents.insert(
            document_id.clone(),
            DocumentRecord {
                content_hash: state
                    .get("document_hashes")
                    .and_then(Value::as_object)
                    .and_then(|object| object.get(&document_id))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                metadata,
                text: loaded.raw_text.get(&document_id).cloned(),
                token_lists,
                chunks,
            },
        );
    }
    let lexical_index = lexical_index_for_documents(&documents);
    let (field_indexes, all_docs) = filter_indexes_for_documents(&documents);
    let chunk_owner_by_id = chunk_owner_by_id_for_documents(&documents);
    let vector_chunks = vector_chunks_for_documents(&documents);
    let vector_index = match loaded.tvim_path.as_ref() {
        Some(tvim_path) if vectors_seeded => Some(TurboVecNativeIndex::load_with_manifest(
            tvim_path,
            loaded.tvim_manifest.as_ref(),
            &vector_chunks,
            loaded.generation,
        )?),
        _ => None,
    };
    // Native owns the generation store as the write-through writer, so it keeps the
    // loaded base epoch and calibration and appends further deltas onto that base
    // across reopens (no co-writer to invalidate it). It opens with no pending
    // changes since the loaded base+deltas are already durable.
    let base_calibration_fingerprint = vector_index
        .as_ref()
        .map_or(0, TurboVecNativeIndex::calibration_fingerprint);
    let base_epoch = loaded.base_epoch;
    Ok(VectorOnlyIndex {
        index_id,
        index_key,
        client_id_hash,
        name: state
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("lodedb-local")
            .to_string(),
        model: state
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or("native-core")
            .to_string(),
        provider: state
            .get("provider")
            .and_then(Value::as_str)
            .unwrap_or("native")
            .to_string(),
        task: state
            .get("task")
            .and_then(Value::as_str)
            .unwrap_or("native-core")
            .to_string(),
        route_profile: state
            .get("route_profile")
            .and_then(Value::as_str)
            .unwrap_or("native-core")
            .to_string(),
        storage_profile: state
            .get("storage_profile")
            .and_then(Value::as_str)
            .unwrap_or("native-core")
            .to_string(),
        vector_dim,
        bit_width,
        documents,
        generation: loaded.generation,
        vectors_seeded,
        query_rotation,
        delete_count: state
            .get("delete_count")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize,
        deleted_chunk_count: state
            .get("deleted_chunk_count")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize,
        lexical_index,
        vector_index: RefCell::new(vector_index),
        field_indexes,
        all_docs,
        chunk_owner_by_id,
        base_epoch,
        base_calibration_fingerprint,
        pending_upserts: BTreeSet::new(),
        pending_deletes: BTreeSet::new(),
        pending_removed_stable_ids: BTreeSet::new(),
    })
}

fn token_lists_from_raw_text(text: &str, chunk_character_limit: usize) -> Vec<Vec<String>> {
    chunk_text(text, chunk_character_limit)
        .unwrap_or_default()
        .iter()
        .map(|chunk| tokenize(chunk))
        .collect()
}

fn lexical_index_for_documents(documents: &BTreeMap<String, DocumentRecord>) -> Bm25Index {
    let mut lexical_index = Bm25Index::empty();
    for (document_id, record) in documents {
        let units = record
            .chunks
            .iter()
            .enumerate()
            .map(|(offset, chunk)| {
                (
                    chunk.chunk_id.clone(),
                    record.token_lists.get(offset).cloned().unwrap_or_default(),
                )
            })
            .collect::<Vec<_>>();
        lexical_index.replace_group(document_id, &units);
    }
    lexical_index
}

fn vector_chunks_for_documents(
    documents: &BTreeMap<String, DocumentRecord>,
) -> Vec<CoreVectorChunk> {
    documents
        .iter()
        .flat_map(|(document_id, record)| {
            record.chunks.iter().map(|chunk| {
                CoreVectorChunk::new(
                    chunk.chunk_id.clone(),
                    document_id.clone(),
                    chunk.vector.clone(),
                )
            })
        })
        .collect()
}

fn filter_indexes_for_documents(
    documents: &BTreeMap<String, DocumentRecord>,
) -> (BTreeMap<String, FieldIndex>, DocSet) {
    let metadata_by_id = documents
        .iter()
        .map(|(document_id, record)| (document_id.clone(), record.metadata.clone()))
        .collect::<BTreeMap<_, _>>();
    build_field_indexes(&metadata_by_id)
}

fn chunk_owner_by_id_for_documents(
    documents: &BTreeMap<String, DocumentRecord>,
) -> HashMap<String, String> {
    documents
        .iter()
        .flat_map(|(document_id, record)| {
            record
                .chunks
                .iter()
                .map(|chunk| (chunk.chunk_id.clone(), document_id.clone()))
        })
        .collect()
}

#[derive(Debug, Clone, Default)]
struct ReconstructedTvimVectors {
    vectors: BTreeMap<String, Vec<f32>>,
    query_rotation: Option<Vec<f32>>,
}

struct TvimBaseBytes {
    bytes: Vec<u8>,
    rows: usize,
    calibration_fingerprint: u64,
}

fn tvim_base_for_index(index: &VectorOnlyIndex) -> Result<Option<TvimBaseBytes>, CoreError> {
    if !index.vectors_seeded {
        return Ok(None);
    }
    if let Some(tvim) = tvim_base_from_cached_index(index)? {
        return Ok(Some(tvim));
    }
    let chunks = index.persistable_chunks();
    if chunks.is_empty() {
        return Ok(None);
    }
    let mut embeddings = Vec::with_capacity(chunks.len() * index.vector_dim);
    let chunk_ids = chunks
        .iter()
        .map(|(chunk_id, _)| chunk_id.clone())
        .collect::<Vec<_>>();
    for (_, vector) in &chunks {
        embeddings.extend_from_slice(vector);
    }
    let stable_ids = stable_uint64_ids_for_chunk_ids(&chunk_ids);
    let mut tvim = IdMapIndex::new(index.vector_dim, index.bit_width).map_err(turbovec_error)?;
    tvim.add_with_ids(&embeddings, &stable_ids)
        .map_err(turbovec_error)?;
    let scratch_path = scratch_tvim_path(&index.index_id, index.generation);
    tvim.write(&scratch_path).map_err(core_io_error)?;
    let bytes = match fs::read(&scratch_path).map_err(core_io_error) {
        Ok(bytes) => bytes,
        Err(error) => {
            let _ = fs::remove_file(&scratch_path);
            return Err(error);
        }
    };
    let _ = fs::remove_file(&scratch_path);
    Ok(Some(TvimBaseBytes {
        bytes,
        rows: stable_ids.len(),
        calibration_fingerprint: tvim.calibration_fingerprint(),
    }))
}

fn tvim_base_from_cached_index(
    index: &VectorOnlyIndex,
) -> Result<Option<TvimBaseBytes>, CoreError> {
    let cached = index.vector_index.borrow();
    let Some(tvim) = cached.as_ref() else {
        return Ok(None);
    };
    if tvim.len() != index.chunk_count() {
        return Err(CoreError::new(
            CoreErrorCode::CorruptStore,
            "live TurboVec index row count does not match JSON state",
        ));
    }
    let scratch_path = scratch_tvim_path(&index.index_id, index.generation);
    tvim.write(&scratch_path)?;
    let bytes = match fs::read(&scratch_path).map_err(core_io_error) {
        Ok(bytes) => bytes,
        Err(error) => {
            let _ = fs::remove_file(&scratch_path);
            return Err(error);
        }
    };
    let _ = fs::remove_file(&scratch_path);
    Ok(Some(TvimBaseBytes {
        bytes,
        rows: tvim.len(),
        calibration_fingerprint: tvim.calibration_fingerprint(),
    }))
}

fn scratch_tvim_path(index_id: &str, generation: u64) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "lodedb-core-tvim-{}-{index_id}-{generation}-{nanos}.tvim",
        std::process::id()
    ))
}

fn reconstruct_tvim_vectors(
    loaded: &crate::storage::LoadedStore,
    vector_dim: usize,
) -> Result<Option<ReconstructedTvimVectors>, CoreError> {
    let Some(tvim_path) = &loaded.tvim_path else {
        return Ok(None);
    };
    if loaded.chunk_count() == 0 {
        return Ok(Some(ReconstructedTvimVectors::default()));
    }

    let chunk_ids = loaded
        .state
        .get("chunks")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|row| row.get("chunk_id").and_then(Value::as_str))
        .map(str::to_string)
        .collect::<Vec<_>>();
    if chunk_ids.is_empty() {
        return Ok(Some(ReconstructedTvimVectors::default()));
    }
    let stable_ids = stable_uint64_ids_for_chunk_ids(&chunk_ids);
    let index = crate::vector::turbovec::load_id_map_with_manifest(
        tvim_path,
        loaded.tvim_manifest.as_ref(),
    )?;
    if index.dim() != vector_dim {
        return invalid("persisted TurboVec dimension does not match JSON state");
    }
    let rows = index
        .reconstruct_rows(&stable_ids)
        .map_err(|error| CoreError::new(CoreErrorCode::CorruptStore, error.to_string()))?;
    if rows.len() != chunk_ids.len() * vector_dim {
        return Err(CoreError::new(
            CoreErrorCode::CorruptStore,
            "persisted TurboVec reconstruction returned malformed rows",
        ));
    }
    let vectors = chunk_ids
        .into_iter()
        .zip(rows.chunks_exact(vector_dim))
        .map(|(chunk_id, row)| (chunk_id, row.to_vec()))
        .collect::<BTreeMap<_, _>>();
    Ok(Some(ReconstructedTvimVectors {
        vectors,
        query_rotation: index.rotation_matrix(),
    }))
}

/// Scalar state snapshot shared by full rewrites and journal deltas. The document
/// collections are deliberately excluded: a delta replays documents individually,
/// so embedding the full collections in every delta segment would make each
/// commit O(corpus) again. Mirrors the host writer's `_state_header_payload`.
fn state_header_for_index(index: &VectorOnlyIndex) -> serde_json::Map<String, Value> {
    let serde_json::Value::Object(header) = serde_json::json!({
        "cache_reuse_count": 0,
        "client_id_hash": index.client_id_hash,
        "columnar_generation": index.generation.max(1),
        "created_at": "1970-01-01T00:00:00+00:00",
        "delete_count": index.delete_count,
        "deleted_chunk_count": index.deleted_chunk_count,
        "embedded_chunk_count": index.chunk_count(),
        "fallback_count": 0,
        "fallback_reasons": {},
        "index_id": index.index_id,
        "index_key": index.index_key,
        "metadata": {},
        "model": index.model,
        "name": index.name,
        "native_dim": index.vector_dim,
        "provider": index.provider,
        "query_count": 0,
        "route_profile": index.route_profile,
        "schema_version": 1,
        "status": "ready",
        "storage_profile": index.storage_profile,
        "task": index.task,
        "turbovec_bit_width": index.bit_width,
        "updated_at": "1970-01-01T00:00:00+00:00"
    }) else {
        unreachable!("state header literal is an object")
    };
    header
}

/// Serializes one document's redacted state-journal row (chunk ids, metadata, and
/// per-chunk content hashes), matching the rows a full base writes for the same
/// document so a delta replay reconstructs identical state.
fn state_journal_document_entry(document_id: &str, record: &DocumentRecord) -> Value {
    serde_json::json!({
        "document_id": document_id,
        "document_hash": record.content_hash,
        "chunk_ids": record
            .chunks
            .iter()
            .map(|chunk| chunk.chunk_id.clone())
            .collect::<Vec<_>>(),
        "metadata": record
            .metadata
            .iter()
            .map(|(key, value)| (key.clone(), Value::String(value.clone())))
            .collect::<serde_json::Map<_, _>>(),
        "chunks": record
            .chunks
            .iter()
            .map(|chunk| serde_json::json!({
                "chunk_id": chunk.chunk_id,
                "content_hash": crate::text::hash::sha256_text(&chunk.chunk_id),
                "document_id": document_id,
            }))
            .collect::<Vec<_>>(),
    })
}

fn state_payload_for_index(index: &VectorOnlyIndex) -> Value {
    let mut chunks = Vec::new();
    let mut document_hashes = serde_json::Map::new();
    let mut document_chunk_ids = serde_json::Map::new();
    let mut document_metadata = serde_json::Map::new();
    for (document_id, record) in &index.documents {
        document_hashes.insert(
            document_id.clone(),
            Value::String(record.content_hash.clone()),
        );
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
    let mut payload = state_header_for_index(index);
    payload.insert("chunks".to_string(), Value::Array(chunks));
    payload.insert(
        "document_hashes".to_string(),
        Value::Object(document_hashes),
    );
    payload.insert(
        "document_chunk_ids".to_string(),
        Value::Object(document_chunk_ids),
    );
    payload.insert(
        "document_metadata".to_string(),
        Value::Object(document_metadata),
    );
    Value::Object(payload)
}

/// Rewrite a fresh base after this many delta segments, matching the host writer.
const MAX_GENERATION_DELTA_SEGMENTS: usize = 64;

/// Persists one index as a generation commit, appending an O(changed) delta onto
/// the live base when possible and only rewriting a full base on a cold build, a
/// calibration change, a compaction threshold, or a missing store base.
fn persist_index_generation(
    index: &mut VectorOnlyIndex,
    dir: &Path,
    fsync: bool,
    store_text: bool,
    index_text: bool,
) -> Result<(), CoreError> {
    let nothing_pending = index.pending_upserts.is_empty()
        && index.pending_deletes.is_empty()
        && index.pending_removed_stable_ids.is_empty();
    // A committed base with nothing pending is already durable on disk.
    if index.base_epoch != 0 && nothing_pending {
        return Ok(());
    }
    let generation = index.generation.max(1);
    let live_fingerprint = index
        .vector_index
        .borrow()
        .as_ref()
        .map_or(0, TurboVecNativeIndex::calibration_fingerprint);
    let needs_base = index.base_epoch == 0
        || !index.vectors_seeded
        || live_fingerprint != index.base_calibration_fingerprint
        || !native_base_appendable(
            dir,
            &index.index_key,
            index.base_epoch,
            store_text,
            index_text,
        )
        || generation_should_compact(
            dir,
            &index.index_key,
            index.base_epoch,
            index.documents.len(),
        )?;

    if needs_base {
        write_index_base(index, dir, fsync, store_text, index_text, generation)?;
        index.base_epoch = generation;
        index.base_calibration_fingerprint = live_fingerprint;
    } else {
        write_index_delta(index, dir, fsync, store_text, index_text, generation)?;
    }
    index.pending_upserts.clear();
    index.pending_deletes.clear();
    index.pending_removed_stable_ids.clear();
    Ok(())
}

/// Returns whether native's own base at `base_epoch` still has the base files a
/// delta needs to append onto. Defensive against a base file disappearing; the
/// caller rewrites a fresh authored base when it returns false.
fn native_base_appendable(
    dir: &Path,
    index_key: &str,
    base_epoch: u64,
    store_text: bool,
    index_text: bool,
) -> bool {
    use crate::storage::commit_manifest::{base_json_path, base_tvlex_path, base_tvtext_path};
    if base_epoch == 0 {
        return false;
    }
    if !base_json_path(dir, index_key, base_epoch).is_file() {
        return false;
    }
    if store_text && !base_tvtext_path(dir, index_key, base_epoch).is_file() {
        return false;
    }
    if index_text && !base_tvlex_path(dir, index_key, base_epoch).is_file() {
        return false;
    }
    true
}

/// Returns whether the delta backlog on native's own base warrants a fresh base.
fn generation_should_compact(
    dir: &Path,
    index_key: &str,
    base_epoch: u64,
    document_count: usize,
) -> Result<bool, CoreError> {
    let (segments, delta_documents) =
        crate::storage::generation_delta_backlog(dir, index_key, base_epoch)?;
    if segments == 0 {
        return Ok(false);
    }
    if segments >= MAX_GENERATION_DELTA_SEGMENTS {
        return Ok(true);
    }
    // Fold when journaled documents reach 25% of the live document set.
    Ok(delta_documents * 4 >= document_count.max(1))
}

/// Writes a full base for one index (the cold-build / compaction path).
fn write_index_base(
    index: &VectorOnlyIndex,
    dir: &Path,
    fsync: bool,
    store_text: bool,
    index_text: bool,
    generation: u64,
) -> Result<(), CoreError> {
    let tvim_base = tvim_base_for_index(index)?;
    let state = state_payload_for_index(index);
    let raw_text = if store_text {
        raw_text_for_documents(index, index.documents.keys().map(String::as_str))
    } else {
        BTreeMap::new()
    };
    let lexical_tokens = if index_text {
        lexical_tokens_for_documents(index, index.documents.keys().map(String::as_str))
    } else {
        BTreeMap::new()
    };
    crate::storage::write_generation_commit(
        dir,
        crate::storage::GenerationCommitInput {
            index_key: &index.index_key,
            generation,
            base_epoch: generation,
            state: &state,
            tvim: tvim_base
                .as_ref()
                .map(|tvim| crate::storage::TvimBaseWrite {
                    bytes: &tvim.bytes,
                    rows: tvim.rows,
                    calibration_fingerprint: tvim.calibration_fingerprint,
                }),
            raw_text: Some(&raw_text),
            lexical_tokens: Some(&lexical_tokens),
        },
        crate::storage::GenerationWriteOptions {
            fsync,
            retained_epochs: 4,
        },
    )?;
    Ok(())
}

/// Appends an O(changed) generation delta for one index onto the live base.
fn write_index_delta(
    index: &VectorOnlyIndex,
    dir: &Path,
    fsync: bool,
    store_text: bool,
    index_text: bool,
    generation: u64,
) -> Result<(), CoreError> {
    let upserted_documents = index
        .pending_upserts
        .iter()
        .filter_map(|document_id| {
            index
                .documents
                .get(document_id)
                .map(|record| state_journal_document_entry(document_id, record))
        })
        .collect::<Vec<_>>();
    let deleted_document_ids = index.pending_deletes.iter().cloned().collect::<Vec<_>>();
    let raw_text_upserts = store_text
        .then(|| raw_text_for_documents(index, index.pending_upserts.iter().map(String::as_str)));
    let lexical_upserts = index_text.then(|| {
        lexical_tokens_for_documents(index, index.pending_upserts.iter().map(String::as_str))
    });
    let state_header = Value::Object(state_header_for_index(index));
    crate::storage::write_generation_delta(
        dir,
        crate::storage::GenerationDeltaInput {
            index_key: &index.index_key,
            generation,
            base_epoch: index.base_epoch,
            state_header: &state_header,
            upserted_documents,
            deleted_document_ids: deleted_document_ids.clone(),
            document_count_after: index.documents.len(),
            chunk_count_after: index.chunk_count(),
            tvim: build_tvim_delta(index)?,
            raw_text_upserts,
            lexical_upserts,
            document_deletes: deleted_document_ids,
        },
        fsync,
    )?;
    Ok(())
}

/// Builds the encoded tvim delta from the live index: current rows for the
/// pending-upsert documents (overwriting in place) plus the stable ids dropped
/// since the base. Returns `None` when the index holds no vector rows.
fn build_tvim_delta(
    index: &VectorOnlyIndex,
) -> Result<Option<crate::storage::TvimDeltaWrite>, CoreError> {
    let guard = index.vector_index.borrow();
    let Some(live) = guard.as_ref() else {
        return Ok(None);
    };
    let mut upsert_chunk_ids = Vec::new();
    for document_id in &index.pending_upserts {
        if let Some(record) = index.documents.get(document_id) {
            for chunk in &record.chunks {
                upsert_chunk_ids.push(chunk.chunk_id.clone());
            }
        }
    }
    let upsert_stable_ids = live.stable_ids_for_chunks(&upsert_chunk_ids);
    let (upsert_codes, upsert_scales) = if upsert_stable_ids.is_empty() {
        (Vec::new(), Vec::new())
    } else {
        live.export_encoded(&upsert_stable_ids)?
    };
    Ok(Some(crate::storage::TvimDeltaWrite {
        upsert_stable_ids,
        upsert_codes,
        upsert_scales,
        removed_stable_ids: index.pending_removed_stable_ids.iter().copied().collect(),
        rows_after: live.len(),
        calibration_fingerprint: live.calibration_fingerprint(),
    }))
}

fn raw_text_for_documents<'a>(
    index: &VectorOnlyIndex,
    document_ids: impl Iterator<Item = &'a str>,
) -> BTreeMap<String, String> {
    document_ids
        .filter_map(|document_id| {
            index
                .documents
                .get(document_id)
                .and_then(|record| record.text.as_ref())
                .map(|text| (document_id.to_string(), text.clone()))
        })
        .collect()
}

fn lexical_tokens_for_documents<'a>(
    index: &VectorOnlyIndex,
    document_ids: impl Iterator<Item = &'a str>,
) -> BTreeMap<String, Vec<Vec<String>>> {
    document_ids
        .filter_map(|document_id| {
            index
                .documents
                .get(document_id)
                .filter(|record| !record.token_lists.is_empty())
                .map(|record| (document_id.to_string(), record.token_lists.clone()))
        })
        .collect()
}

fn core_io_error(error: std::io::Error) -> CoreError {
    CoreError::new(
        CoreErrorCode::Internal,
        format!("persistent engine I/O failed: {error}"),
    )
}

fn is_native_replayable_wal_record(record: &crate::storage::wal::WalRecord) -> bool {
    matches!(
        record.op.as_str(),
        "upsert_vectors"
            | "delete_documents"
            | "apply_embedded_documents"
            | "update_document_payload"
    )
}

fn vector_document_from_wal(value: &Value) -> Result<CoreVectorDocument, CoreError> {
    let document_id = value
        .get("document_id")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_err("WAL vector payload missing document_id"))?
        .to_string();
    let vector = value
        .get("vector")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid_err("WAL vector payload missing vector"))?
        .iter()
        .map(|value| value.as_f64().unwrap_or(0.0) as f32)
        .collect::<Vec<_>>();
    let metadata = metadata_from_value(value.get("metadata").unwrap_or(&Value::Null));
    let text = value
        .get("text")
        .and_then(Value::as_str)
        .map(ToString::to_string);
    Ok(CoreVectorDocument {
        document_id,
        vector,
        metadata,
        text,
    })
}

fn metadata_from_value(value: &Value) -> CoreMetadata {
    value
        .as_object()
        .map(|object| {
            object
                .iter()
                .map(|(key, value)| (key.clone(), value.as_str().unwrap_or("").to_string()))
                .collect()
        })
        .unwrap_or_default()
}

fn document_resource_payload(document_id: &str, record: &DocumentRecord) -> Value {
    serde_json::json!({
        "document_id": document_id,
        "metadata": record.metadata.clone(),
        "chunk_count": record.chunks.len(),
        "content_hash": record.content_hash.clone(),
    })
}

struct VectorOnlyIndex {
    index_id: String,
    index_key: String,
    client_id_hash: String,
    name: String,
    model: String,
    provider: String,
    task: String,
    route_profile: String,
    storage_profile: String,
    vector_dim: usize,
    bit_width: usize,
    documents: BTreeMap<String, DocumentRecord>,
    generation: u64,
    vectors_seeded: bool,
    query_rotation: Option<Vec<f32>>,
    delete_count: usize,
    deleted_chunk_count: usize,
    lexical_index: Bm25Index,
    vector_index: RefCell<Option<TurboVecNativeIndex>>,
    field_indexes: BTreeMap<String, FieldIndex>,
    all_docs: DocSet,
    chunk_owner_by_id: HashMap<String, String>,
    /// Epoch of the current on-disk base; 0 until the first base is persisted.
    /// Generation commits append deltas onto this base and only rewrite a fresh
    /// base (bumping this) on cold build or compaction, keeping commits O(changed).
    base_epoch: u64,
    /// TurboVec calibration fingerprint of the current tvim base. A tvim delta is
    /// only valid against a base with the same fingerprint (the read path rejects
    /// a mismatch), so a fingerprint change forces a base rewrite.
    base_calibration_fingerprint: u64,
    /// Document ids upserted since the current base (drives the state/text/lexical
    /// delta upsert sets). Disjoint from `pending_deletes`.
    pending_upserts: BTreeSet<String>,
    /// Document ids deleted since the current base.
    pending_deletes: BTreeSet<String>,
    /// TurboVec stable ids dropped from the live index since the current base and
    /// not re-added (drives the tvim delta removed set). Tracked at the vector-sync
    /// choke points because a removed chunk's stable id leaves the live id map.
    pending_removed_stable_ids: BTreeSet<u64>,
}

impl VectorOnlyIndex {
    fn new(options: CoreIndexCreateOptions) -> Self {
        Self {
            index_id: options.index_id,
            index_key: options.index_key,
            client_id_hash: options.client_id_hash,
            name: options.name,
            model: options.model,
            provider: options.provider,
            task: options.task,
            route_profile: options.route_profile,
            storage_profile: options.storage_profile,
            vector_dim: options.vector_dim,
            bit_width: options.bit_width,
            documents: BTreeMap::new(),
            generation: 0,
            vectors_seeded: true,
            query_rotation: None,
            delete_count: 0,
            deleted_chunk_count: 0,
            lexical_index: Bm25Index::empty(),
            vector_index: RefCell::new(None),
            field_indexes: BTreeMap::new(),
            all_docs: DocSet::new(),
            chunk_owner_by_id: HashMap::new(),
            base_epoch: 0,
            base_calibration_fingerprint: 0,
            pending_upserts: BTreeSet::new(),
            pending_deletes: BTreeSet::new(),
            pending_removed_stable_ids: BTreeSet::new(),
        }
    }

    fn require_vectors_seeded(&self) -> Result<(), CoreError> {
        if !self.vectors_seeded {
            return Err(CoreError::new(
                CoreErrorCode::Unsupported,
                "persisted vector sidecars are not yet loaded by native core",
            ));
        }
        Ok(())
    }

    fn require_vectors_mutable(&self) -> Result<(), CoreError> {
        self.require_vectors_seeded()
    }

    fn sync_vector_index_upsert(&mut self, chunks: &[CoreVectorChunk]) -> Result<(), CoreError> {
        // Build the live index on first use from all current documents (which
        // already include these chunks), then maintain it incrementally. It must
        // exist before any generation commit so it is the single calibrated source
        // of truth for both the base write and the tvim deltas; otherwise a delta
        // built against a separately-calibrated index would be rejected on replay.
        let built_now = if self.vector_index.borrow().is_none() {
            let built = TurboVecNativeIndex::build(
                &self.vector_chunks(),
                self.vector_dim,
                self.bit_width,
                self.generation,
            )?;
            *self.vector_index.borrow_mut() = Some(built);
            true
        } else {
            false
        };
        let readded = {
            let mut guard = self.vector_index.borrow_mut();
            let Some(index) = guard.as_mut() else {
                return Ok(());
            };
            // A just-built index already contains every current chunk; only an
            // existing index needs the incremental upsert.
            if !built_now {
                index.upsert_chunks(chunks)?;
            }
            let chunk_ids: Vec<String> =
                chunks.iter().map(|chunk| chunk.chunk_id.clone()).collect();
            index.stable_ids_for_chunks(&chunk_ids)
        };
        // Rows written this cycle are no longer "removed" relative to the base.
        for stable_id in readded {
            self.pending_removed_stable_ids.remove(&stable_id);
        }
        Ok(())
    }

    fn sync_vector_index_remove(&mut self, chunk_ids: &[String]) {
        let removed = {
            let mut guard = self.vector_index.borrow_mut();
            let Some(index) = guard.as_mut() else {
                return;
            };
            // Read the stable ids before removal: a removed id leaves the live id
            // map, so the tvim delta could not recover it at persist time.
            let removed = index.stable_ids_for_chunks(chunk_ids);
            index.remove_chunks(chunk_ids);
            removed
        };
        self.pending_removed_stable_ids.extend(removed);
    }

    fn add_document_indexes(
        &mut self,
        document_id: &str,
        metadata: &CoreMetadata,
        chunks: &[ChunkRecord],
        changed_filter_fields: &mut BTreeSet<String>,
    ) {
        self.all_docs.insert(document_id.to_string());
        for (key, value) in metadata {
            let field = self.field_indexes.entry(key.clone()).or_default();
            field.insert(document_id.to_string(), value.clone());
            changed_filter_fields.insert(key.clone());
        }
        for chunk in chunks {
            self.chunk_owner_by_id
                .insert(chunk.chunk_id.clone(), document_id.to_string());
        }
        // Track this document as pending for the next generation delta; the tvim
        // row set is tracked separately at the vector-sync choke points.
        self.pending_deletes.remove(document_id);
        self.pending_upserts.insert(document_id.to_string());
    }

    fn remove_document_indexes(
        &mut self,
        document_id: &str,
        metadata: &CoreMetadata,
        chunks: &[ChunkRecord],
        changed_filter_fields: &mut BTreeSet<String>,
    ) {
        self.all_docs.remove(document_id);
        for (key, value) in metadata {
            if let Some(field) = self.field_indexes.get_mut(key) {
                field.remove(document_id, value);
                changed_filter_fields.insert(key.clone());
            }
        }
        // Track the document-level removal for the next generation delta; the tvim
        // row removals are tracked at the vector-sync choke points.
        self.pending_upserts.remove(document_id);
        self.pending_deletes.insert(document_id.to_string());
        for chunk in chunks {
            self.chunk_owner_by_id.remove(&chunk.chunk_id);
        }
    }

    fn finalize_filter_fields(&mut self, changed_filter_fields: &BTreeSet<String>) {
        // Ordered-filter partitions are rebuilt lazily at query time (and invalidated
        // by `FieldIndex::insert`/`remove`), so writes only need to drop fields that
        // became empty. This keeps single-row mutations O(changed), not O(cardinality).
        for key in changed_filter_fields {
            if self
                .field_indexes
                .get(key)
                .is_some_and(FieldIndex::is_empty)
            {
                self.field_indexes.remove(key);
            }
        }
    }

    fn resolve_filter(&self, filter: Option<&Value>) -> Result<BTreeSet<String>, CoreError> {
        let Some(filter) = filter else {
            return Ok(self.all_docs.clone());
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
        let (document_ids, metadata_filter) = if structured {
            (object.get("document_ids"), object.get("metadata"))
        } else {
            (None, Some(filter))
        };

        // Build candidates straight from the constraints rather than cloning the
        // whole corpus first: a `document_ids` allowlist is bounded by the
        // requested ids, and a metadata filter already resolves to a subset of
        // `all_docs` (it uses `all_docs` internally for $ne/$nin/$exists:false).
        // Cloning `all_docs` up front made every filtered query O(corpus).
        let id_candidates = match document_ids {
            Some(value) => {
                let array = value
                    .as_array()
                    .ok_or_else(|| invalid_err("document_ids filter must be a list"))?;
                // Match the SDK's filter normalizer so every binding fails closed
                // alike: a document_ids filter must be a non-empty list of non-blank
                // strings rather than silently dropping malformed entries.
                if array.is_empty() {
                    return Err(invalid_err("document_ids filter must not be empty"));
                }
                let mut ids = BTreeSet::new();
                for entry in array {
                    let id = entry.as_str().ok_or_else(|| {
                        invalid_err("document_ids filter entries must be strings")
                    })?;
                    if id.trim().is_empty() {
                        return Err(invalid_err("document_ids filter entries must be non-blank"));
                    }
                    if self.all_docs.contains(id) {
                        ids.insert(id.to_string());
                    }
                }
                Some(ids)
            }
            None => None,
        };
        let metadata_docs = match metadata_filter {
            Some(metadata_filter) => {
                let coerced = coerce_sdk_filter(metadata_filter)?;
                Some(resolve_filter(
                    &coerced,
                    &self.field_indexes,
                    &self.all_docs,
                )?)
            }
            None => None,
        };

        Ok(match (id_candidates, metadata_docs) {
            (Some(ids), Some(metadata)) => ids.intersection(&metadata).cloned().collect(),
            (Some(ids), None) => ids,
            (None, Some(metadata)) => metadata,
            // Unreachable: a structured filter always carries at least one of the
            // two keys, and a non-structured filter sets `metadata_filter`.
            (None, None) => self.all_docs.clone(),
        })
    }

    /// Resolves the chunk allowlist and considered-document count for a query.
    ///
    /// The unfiltered path must NOT clone `all_docs` (a `BTreeSet` of every
    /// document id): for a 20k-corpus single-query loop that clone runs once per
    /// query and dominates the per-call cost once JSON marshalling is removed.
    /// Unfiltered queries scan the whole index, so the allowlist is empty and the
    /// considered count is simply `all_docs.len()`.
    fn query_allowlist(
        &self,
        filter: Option<&Value>,
    ) -> Result<(usize, Vec<String>, bool), CoreError> {
        match filter {
            None => Ok((self.all_docs.len(), Vec::new(), false)),
            Some(_) => {
                let candidates = self.resolve_filter(filter)?;
                let total_considered = candidates.len();
                if candidates.is_empty() {
                    return Ok((total_considered, Vec::new(), true));
                }
                Ok((
                    total_considered,
                    self.chunk_ids_for_documents(&candidates),
                    false,
                ))
            }
        }
    }

    fn query_vector_turbovec(
        &self,
        query_vector: &[f32],
        top_k: usize,
        filter: Option<&Value>,
    ) -> Result<CoreSearchResults, CoreError> {
        let (total_considered, allowlist, empty_filter) = self.query_allowlist(filter)?;
        if empty_filter {
            return Ok(CoreSearchResults {
                hits: Vec::new(),
                total_considered,
            });
        }
        let index = self.turbovec_index()?;
        let hits = index
            .search(query_vector, top_k, &allowlist)?
            .into_iter()
            .filter_map(|hit| {
                self.documents
                    .get(&hit.document_id)
                    .map(|record| CoreSearchHit {
                        document_id: hit.document_id,
                        chunk_id: hit.chunk_id,
                        score: hit.score,
                        metadata: record.metadata.clone(),
                    })
            })
            .collect();
        Ok(CoreSearchResults {
            hits,
            total_considered,
        })
    }

    fn query_vectors_batch_turbovec(
        &self,
        query_vectors: &[Vec<f32>],
        top_k: usize,
        filter: Option<&Value>,
    ) -> Result<Vec<CoreSearchResults>, CoreError> {
        let (total_considered, allowlist, empty_filter) = self.query_allowlist(filter)?;
        if empty_filter {
            return Ok(query_vectors
                .iter()
                .map(|_| CoreSearchResults {
                    hits: Vec::new(),
                    total_considered,
                })
                .collect());
        }
        let index = self.turbovec_index()?;
        index
            .search_batch(query_vectors, top_k, &allowlist)?
            .into_iter()
            .map(|row| {
                let hits = row
                    .into_iter()
                    .filter_map(|hit| {
                        self.documents
                            .get(&hit.document_id)
                            .map(|record| CoreSearchHit {
                                document_id: hit.document_id,
                                chunk_id: hit.chunk_id,
                                score: hit.score,
                                metadata: record.metadata.clone(),
                            })
                    })
                    .collect();
                Ok(CoreSearchResults {
                    hits,
                    total_considered,
                })
            })
            .collect()
    }

    fn turbovec_index(&self) -> Result<Ref<'_, TurboVecNativeIndex>, CoreError> {
        {
            let mut cached = self.vector_index.borrow_mut();
            if cached.is_none() {
                *cached = Some(TurboVecNativeIndex::build(
                    &self.vector_chunks(),
                    self.vector_dim,
                    self.bit_width,
                    self.generation,
                )?);
            }
        }
        Ref::filter_map(self.vector_index.borrow(), Option::as_ref).map_err(|_| {
            CoreError::new(
                CoreErrorCode::Unsupported,
                "native TurboVec index is unavailable",
            )
        })
    }

    fn chunk_count(&self) -> usize {
        self.documents
            .values()
            .map(|record| record.chunks.len())
            .sum()
    }

    fn persistable_chunks(&self) -> Vec<(String, Vec<f32>)> {
        self.documents
            .values()
            .flat_map(|record| {
                record
                    .chunks
                    .iter()
                    .map(|chunk| (chunk.chunk_id.clone(), chunk.vector.clone()))
            })
            .collect()
    }

    fn vector_chunks(&self) -> Vec<CoreVectorChunk> {
        self.documents
            .iter()
            .flat_map(|(document_id, record)| {
                record.chunks.iter().map(|chunk| {
                    CoreVectorChunk::new(
                        chunk.chunk_id.clone(),
                        document_id.clone(),
                        chunk.vector.clone(),
                    )
                })
            })
            .collect()
    }

    fn chunk_ids_for_documents(&self, document_ids: &BTreeSet<String>) -> Vec<String> {
        document_ids
            .iter()
            .filter_map(|document_id| self.documents.get(document_id))
            .flat_map(|record| record.chunks.iter().map(|chunk| chunk.chunk_id.clone()))
            .collect()
    }

    fn document_for_chunk(&self, chunk_id: &str) -> Option<(String, &DocumentRecord)> {
        let document_id = self.chunk_owner_by_id.get(chunk_id)?;
        let record = self.documents.get(document_id)?;
        Some((document_id.clone(), record))
    }
}

/// Collects the current vectors for the given chunk ids that are NOT in `skip`
/// (the chunks being freshly (re)embedded). Looks each one up via the O(1)
/// chunk->owner map instead of cloning every chunk vector in the corpus, so an
/// incremental text upsert/replay stays O(changed) rather than O(corpus).
fn reusable_chunk_vectors<'a>(
    index: &VectorOnlyIndex,
    chunk_ids: impl Iterator<Item = &'a str>,
    skip: &BTreeMap<String, Vec<f32>>,
) -> BTreeMap<String, Vec<f32>> {
    let mut reused: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    for chunk_id in chunk_ids {
        if skip.contains_key(chunk_id) || reused.contains_key(chunk_id) {
            continue;
        }
        if let Some((_, record)) = index.document_for_chunk(chunk_id) {
            if let Some(existing) = record
                .chunks
                .iter()
                .find(|chunk| chunk.chunk_id == chunk_id)
            {
                reused.insert(chunk_id.to_string(), existing.vector.clone());
            }
        }
    }
    reused
}

#[derive(Debug, Clone)]
struct DocumentRecord {
    content_hash: String,
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
    // The TurboVec serving index requires a positive multiple of 8 (its bit-plane
    // layout), and the index is built lazily on the first query. Reject an
    // unsupported dimension at create time rather than accepting an index that
    // upserts successfully but becomes unsearchable when the serving index builds.
    if vector_dim % 8 != 0 {
        return invalid("vector_dim must be a positive multiple of 8");
    }
    if !matches!(bit_width, 2 | 4) {
        return invalid("bit_width must be 2 or 4");
    }
    Ok(())
}

fn validate_index_options(options: &CoreIndexCreateOptions) -> Result<(), CoreError> {
    for (field, value) in [
        ("index_id", options.index_id.as_str()),
        ("index_key", options.index_key.as_str()),
        ("client_id_hash", options.client_id_hash.as_str()),
        ("name", options.name.as_str()),
        ("model", options.model.as_str()),
        ("provider", options.provider.as_str()),
        ("task", options.task.as_str()),
        ("route_profile", options.route_profile.as_str()),
        ("storage_profile", options.storage_profile.as_str()),
    ] {
        if value.trim().is_empty() {
            return invalid(format!("{field} is required"));
        }
    }
    Ok(())
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn lexical_pool_width(top_k: usize) -> usize {
    (top_k * LEXICAL_POOL_FACTOR).max(LEXICAL_POOL_FLOOR)
}

fn fuse_hybrid_hits(
    vector_hits: Vec<CoreSearchHit>,
    lexical_hits: Vec<CoreSearchHit>,
    top_k: usize,
) -> Vec<CoreSearchHit> {
    let mut fused = HashMap::<String, (f64, CoreSearchHit)>::with_capacity(
        vector_hits.len() + lexical_hits.len(),
    );
    add_rrf_hits(&mut fused, vector_hits);
    add_rrf_hits(&mut fused, lexical_hits);
    let mut rows = fused.into_values().collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .0
            .partial_cmp(&left.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.1.chunk_id.cmp(&right.1.chunk_id))
    });
    rows.truncate(top_k);
    rows.into_iter()
        .map(|(score, mut hit)| {
            hit.score = score as f32;
            hit
        })
        .collect()
}

fn add_rrf_hits(fused: &mut HashMap<String, (f64, CoreSearchHit)>, hits: Vec<CoreSearchHit>) {
    let mut seen = HashSet::new();
    for (position, hit) in hits.into_iter().enumerate() {
        if !seen.insert(hit.chunk_id.clone()) {
            continue;
        }
        let contribution = 1.0 / (RRF_C + position as f64 + 1.0);
        fused
            .entry(hit.chunk_id.clone())
            .and_modify(|(score, _)| *score += contribution)
            .or_insert((contribution, hit));
    }
}

fn rotate_query(query: &[f32], rotation: &[f32], dim: usize) -> Result<Vec<f32>, CoreError> {
    if rotation.len() != dim * dim {
        return Err(CoreError::new(
            CoreErrorCode::CorruptStore,
            "persisted TurboVec rotation matrix has invalid dimensions",
        ));
    }
    let mut rotated = vec![0.0; dim];
    for out_d in 0..dim {
        let mut acc = 0.0;
        for in_d in 0..dim {
            acc += query[in_d] * rotation[out_d * dim + in_d];
        }
        rotated[out_d] = acc;
    }
    Ok(rotated)
}

fn invalid<T>(message: impl Into<String>) -> Result<T, CoreError> {
    Err(invalid_err(message))
}

fn invalid_err(message: impl Into<String>) -> CoreError {
    CoreError::new(CoreErrorCode::InvalidArgument, message)
}

fn turbovec_error(error: impl std::fmt::Display) -> CoreError {
    CoreError::new(CoreErrorCode::Internal, error.to_string())
}
