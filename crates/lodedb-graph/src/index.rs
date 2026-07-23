//! The rebuildable semantic index over the topology: a private `lodedb-core`
//! `CoreEngine` that indexes entity labels and fact text for hybrid (vector + BM25)
//! entry-point search, time-scoped by the sortable-string timestamp mirror.
//!
//! Port target: LodeDB's own Python graph index driver
//! (`src/lodedb/graph/knowledge_graph.py` — `_index_node` / `_index_edge` /
//! `_search_index` / `reindex`), retargeted from the Python `LodeDB` SDK onto the
//! Rust `lodedb_core::engine::CoreEngine`.
//!
//! The engine does not embed. For text-in, call
//! `prepare_text_upsert` → embed the returned chunks via the caller's
//! [`Embedder`] → `apply_text_upsert`. For vector-in, `upsert_vectors`. Search:
//! `prepare_query_text` + embed query + `search_embedded_text`, or `query_vector`.
//! Metadata mirrors `kind` (`entity`/`fact`), `type`/`relation`, and the encoded
//! `valid_at`/`invalid_at`/`expired_at` strings (see `crate::temporal`) so as-of
//! filters run engine-side. Doc-id prefixes `n:` (entity) / `e:` (fact), like the
//! Python layer.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::{json, Value};

use lodedb_core::engine::CoreEngine;
use lodedb_core::types::{CoreDocument, CoreOpenOptions, CoreVectorDocument};

use crate::error::{GraphError, Result};
use crate::model::{AsOf, EmbedRole, Embedder, Entity, Fact, GraphConfig};
use crate::temporal::{encode_ts, encode_ts_open, encode_ts_start};

/// Doc-id prefix for an entity's index document (Graphiti node).
const ENTITY_PREFIX: &str = "n:";
/// Doc-id prefix for a fact's index document (Graphiti edge).
const FACT_PREFIX: &str = "e:";
/// The single index id this driver uses within its private engine.
const INDEX_ID: &str = "default";
/// Character budget per text chunk handed to `prepare_text_upsert`.
const CHUNK_CHARACTER_LIMIT: usize = 8192;

/// A scored hit from the semantic index: the entity/fact id and its score.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexHit {
    pub id: String,
    pub score: f32,
}

/// The `lodedb-core`-backed semantic index. Derived from the topology store and
/// rebuildable via [`SemanticIndex::reindex_from`].
pub struct SemanticIndex {
    /// The private vector/lexical engine this index drives.
    engine: CoreEngine,
    /// The index id used for every call into `engine` (always [`INDEX_ID`]).
    index_id: String,
    /// The embedding dimension, mirrored from `GraphConfig::vector_dim`.
    dim: usize,
    /// Whether fact (edge) text is indexed for `semantic_facts`.
    index_facts: bool,
    /// Whether label/fact text is retained + tokenized for lexical (BM25) hybrid
    /// search. Seeds `store_text`/`index_text` on the text-ingest path and selects
    /// hybrid vs. vector query mode.
    index_text: bool,
}

impl SemanticIndex {
    /// Open (creating if needed) the index at `path` with `config`.
    pub fn open(path: &Path, config: &GraphConfig) -> Result<Self> {
        let options = CoreOpenOptions {
            path: path.to_string_lossy().to_string(),
            read_only: false,
            durability: "relaxed".to_string(),
            commit_mode: "wal".to_string(),
            store_text: config.index_text,
            index_text: config.index_text,
            compress_text: true,
            chunk_character_limit: CHUNK_CHARACTER_LIMIT,
            acquire_writer_lock: true,
        };
        let mut engine = CoreEngine::open(options)?;
        if engine.index_ids().iter().any(|id| id == INDEX_ID) {
            // Reopening: the on-disk dimension is authoritative. A config that
            // disagrees means the caller swapped embedders; failing here beats the
            // confusing per-call dimension errors (or silently mixed vector
            // spaces) it would otherwise cause downstream.
            let existing_dim = engine.stats(INDEX_ID)?.vector_dim;
            if existing_dim != config.vector_dim {
                return Err(GraphError::InvalidArgument(format!(
                    "the graph's semantic index was created with dimension \
                     {existing_dim} but the embedder/config supplies \
                     {}; reopen with the original embedding dimension",
                    config.vector_dim
                )));
            }
        } else {
            engine.create_index(INDEX_ID, config.vector_dim, 4)?;
        }
        Ok(Self::from_engine(engine, config))
    }

    /// Open an in-memory index (tests).
    pub fn open_in_memory(config: &GraphConfig) -> Result<Self> {
        let mut engine = CoreEngine::new_in_memory();
        engine.create_index(INDEX_ID, config.vector_dim, 4)?;
        Ok(Self::from_engine(engine, config))
    }

    fn from_engine(engine: CoreEngine, config: &GraphConfig) -> Self {
        SemanticIndex {
            engine,
            index_id: INDEX_ID.to_string(),
            dim: config.vector_dim,
            index_facts: config.index_facts,
            index_text: config.index_text,
        }
    }

    // -- indexing (mirror one entity/fact into the index) -------------------

    /// Index (or, if it has no embeddable text/embedding, clear) an entity.
    /// `embedder` is `None` on the vector-in path, where `vector` must be `Some`.
    pub fn index_entity(
        &mut self,
        entity: &Entity,
        embedder: Option<&dyn Embedder>,
        vector: Option<&[f32]>,
    ) -> Result<()> {
        let doc_id = format!("{ENTITY_PREFIX}{}", entity.id);
        let mut metadata = BTreeMap::new();
        metadata.insert("kind".to_string(), "entity".to_string());
        metadata.insert("type".to_string(), entity.entity_type.clone());
        metadata.insert("entity_id".to_string(), entity.id.clone());
        self.mirror_temporal(&mut metadata, entity.valid_at, entity.invalid_at, entity.expired_at);
        self.write_document(doc_id, metadata, &entity.label, embedder, vector)
    }

    /// Index (or clear) a fact's text/embedding for `semantic_facts`.
    pub fn index_fact(
        &mut self,
        fact: &Fact,
        embedder: Option<&dyn Embedder>,
        vector: Option<&[f32]>,
    ) -> Result<()> {
        let doc_id = format!("{FACT_PREFIX}{}", fact.id);
        // Facts are topology-only unless edge indexing is enabled; clear any stale doc.
        if !self.index_facts {
            self.engine.delete_documents(&self.index_id, &[doc_id])?;
            return Ok(());
        }
        let mut metadata = BTreeMap::new();
        metadata.insert("kind".to_string(), "fact".to_string());
        metadata.insert("relation".to_string(), fact.relation.clone());
        metadata.insert("fact_id".to_string(), fact.id.clone());
        metadata.insert("src".to_string(), fact.src.clone());
        metadata.insert("dst".to_string(), fact.dst.clone());
        self.mirror_temporal(&mut metadata, fact.valid_at, fact.invalid_at, fact.expired_at);
        self.write_document(doc_id, metadata, &fact.fact, embedder, vector)
    }

    /// Remove an entity's index document by id.
    pub fn remove_entity(&mut self, id: &str) -> Result<()> {
        self.engine
            .delete_documents(&self.index_id, &[format!("{ENTITY_PREFIX}{id}")])?;
        Ok(())
    }

    /// Remove a fact's index document by id.
    pub fn remove_fact(&mut self, id: &str) -> Result<()> {
        self.engine
            .delete_documents(&self.index_id, &[format!("{FACT_PREFIX}{id}")])?;
        Ok(())
    }

    // -- retrieval ----------------------------------------------------------

    /// Top-`k` entities for a text query (hybrid) or a precomputed embedding
    /// (vector), optionally narrowed to `entity_type` and time-scoped by `as_of`.
    pub fn semantic_entities(
        &self,
        query: Option<&str>,
        embedding: Option<&[f32]>,
        embedder: Option<&dyn Embedder>,
        k: usize,
        entity_type: Option<&str>,
        as_of: AsOf,
    ) -> Result<Vec<IndexHit>> {
        let filter = asof_filter("entity", entity_type.map(|t| ("type", t)), as_of);
        self.search(query, embedding, embedder, k, filter, ENTITY_PREFIX)
    }

    /// Top-`k` facts for a query/embedding, optionally restricted to `relation`,
    /// time-scoped by `as_of` — Graphiti's default (edge/fact) search shape.
    pub fn semantic_facts(
        &self,
        query: Option<&str>,
        embedding: Option<&[f32]>,
        embedder: Option<&dyn Embedder>,
        k: usize,
        relation: Option<&str>,
        as_of: AsOf,
    ) -> Result<Vec<IndexHit>> {
        let filter = asof_filter("fact", relation.map(|r| ("relation", r)), as_of);
        self.search(query, embedding, embedder, k, filter, FACT_PREFIX)
    }

    /// Candidate entities whose label matches `name` (embedding + lexical), for the
    /// caller's entity-resolution step. The engine surfaces candidates; the caller
    /// (an LLM, or a threshold) decides the merge. This is the helper Graphiti's
    /// resolution leans on, exposed without the LLM.
    pub fn resolve_entity(
        &self,
        name: &str,
        embedder: Option<&dyn Embedder>,
        k: usize,
    ) -> Result<Vec<IndexHit>> {
        // Candidates regardless of temporal validity (AsOf::All): resolution merges
        // against every version, not just the currently-live ones.
        self.semantic_entities(Some(name), None, embedder, k, None, AsOf::All)
    }

    // -- maintenance --------------------------------------------------------

    /// Drop index documents whose ids are not in `live_entity_ids` / `live_fact_ids`
    /// (orphans), returning the count removed. Used by `reindex`.
    pub fn drop_orphans(
        &mut self,
        live_entity_ids: &[String],
        live_fact_ids: &[String],
    ) -> Result<usize> {
        let entity_orphans =
            self.orphan_doc_ids("entity", ENTITY_PREFIX, live_entity_ids)?;
        let fact_orphans = self.orphan_doc_ids("fact", FACT_PREFIX, live_fact_ids)?;
        let mut removed = 0usize;
        for batch in [entity_orphans, fact_orphans] {
            if !batch.is_empty() {
                let deleted = self.engine.delete_documents(&self.index_id, &batch)?;
                removed += deleted.documents_deleted;
            }
        }
        Ok(removed)
    }

    /// Checkpoint the underlying engine to disk.
    pub fn persist(&mut self) -> Result<()> {
        self.engine.persist()?;
        Ok(())
    }

    /// Number of indexed documents.
    pub fn count(&self) -> Result<usize> {
        Ok(self.engine.stats(&self.index_id)?.document_count)
    }

    // -- private helpers ----------------------------------------------------

    /// Mirror the three bi-temporal endpoints into sortable-string metadata, so the
    /// as-of filter runs engine-side. Open endpoints map to the [`crate::temporal`]
    /// sentinel via [`encode_ts_open`].
    fn mirror_temporal(
        &self,
        metadata: &mut BTreeMap<String, String>,
        valid_at: Option<i64>,
        invalid_at: Option<i64>,
        expired_at: Option<i64>,
    ) {
        // Open START (valid_at) → epoch floor (started long ago); open END
        // (invalid_at / expired_at) → far future (still valid). Keeps the index's
        // as-of filter consistent with the SQL topology's `IS NULL` semantics.
        metadata.insert("valid_at".to_string(), encode_ts_start(valid_at));
        metadata.insert("invalid_at".to_string(), encode_ts_open(invalid_at));
        metadata.insert("expired_at".to_string(), encode_ts_open(expired_at));
    }

    /// The write half shared by entities and facts: vector-in upserts the given
    /// vector, an embeddable text goes through the prepare/embed/apply text path,
    /// and anything else clears the (possibly stale) index document.
    fn write_document(
        &mut self,
        doc_id: String,
        metadata: BTreeMap<String, String>,
        text: &str,
        embedder: Option<&dyn Embedder>,
        vector: Option<&[f32]>,
    ) -> Result<()> {
        if let Some(vector) = vector {
            if vector.len() != self.dim {
                return Err(GraphError::InvalidArgument(format!(
                    "vector has dimension {} but the index is {}",
                    vector.len(),
                    self.dim
                )));
            }
            let document = CoreVectorDocument {
                document_id: doc_id,
                vector: vector.to_vec(),
                metadata,
                text: None,
                patch_matrix: None,
            };
            self.engine.upsert_vectors(&self.index_id, &[document])?;
        } else if embedder.is_some() && !text.trim().is_empty() {
            let embedder = embedder.expect("checked is_some above");
            let document = CoreDocument {
                document_id: doc_id,
                text: text.to_string(),
                metadata,
            };
            let plan = self.engine.prepare_text_upsert(
                &self.index_id,
                &[document],
                self.index_text,
                self.index_text,
                CHUNK_CHARACTER_LIMIT,
            )?;
            let texts: Vec<String> = plan
                .chunks_to_embed
                .iter()
                .map(|chunk| chunk.text.clone())
                .collect();
            let embeddings = if texts.is_empty() {
                Vec::new()
            } else {
                let embeddings = embedder.embed(&texts, EmbedRole::Document)?;
                self.check_embedding_shape(&embeddings)?;
                embeddings
            };
            self.engine.apply_text_upsert(&plan, &embeddings, 0.0)?;
        } else {
            // No embeddable content: make sure any stale index doc is cleared.
            self.engine.delete_documents(&self.index_id, &[doc_id])?;
        }
        Ok(())
    }

    /// The read half shared by entity/fact search: precomputed embedding → vector
    /// query; text query → embed + hybrid (or vector when there is no lexical side)
    /// `search_embedded_text`. Hits are de-prefixed back to entity/fact ids.
    fn search(
        &self,
        query: Option<&str>,
        embedding: Option<&[f32]>,
        embedder: Option<&dyn Embedder>,
        k: usize,
        filter: Option<Value>,
        prefix: &str,
    ) -> Result<Vec<IndexHit>> {
        let results = if let Some(embedding) = embedding {
            if embedding.len() != self.dim {
                return Err(GraphError::InvalidArgument(format!(
                    "query embedding has dimension {} but the index is {}",
                    embedding.len(),
                    self.dim
                )));
            }
            self.engine
                .query_vector(&self.index_id, embedding, k, filter.as_ref())?
        } else if let Some(query) = query {
            let embedder = embedder.ok_or_else(|| {
                GraphError::InvalidArgument(
                    "a text query needs an embedder (or pass a precomputed embedding)".to_string(),
                )
            })?;
            // Hybrid fuses BM25 with the vector seeds when a lexical side exists, so
            // exact tokens (codes, serials) the embedding misses still match; with no
            // lexical side fall back to a pure vector query.
            let mode = if self.index_text { "hybrid" } else { "vector" };
            let plan = self.engine.prepare_query_text(query, mode)?;
            let query_embedding = if plan.requires_embedding {
                let embeddings = embedder.embed(&[query.to_string()], EmbedRole::Query)?;
                self.check_embedding_shape(&embeddings)?;
                embeddings.into_iter().next()
            } else {
                None
            };
            self.engine.search_embedded_text(
                &self.index_id,
                &plan,
                query_embedding.as_deref(),
                k,
                filter.as_ref(),
            )?
        } else {
            return Err(GraphError::InvalidArgument(
                "provide a query string or a precomputed embedding".to_string(),
            ));
        };
        Ok(results
            .hits
            .into_iter()
            .map(|hit| IndexHit {
                id: strip_prefix(&hit.document_id, prefix).to_string(),
                score: hit.score,
            })
            .collect())
    }

    /// Validate the caller-supplied embeddings' width against the index dimension so
    /// a wrong-shape embedder fails as an [`GraphError::Embedding`] rather than a
    /// deeper engine error.
    fn check_embedding_shape(&self, embeddings: &[Vec<f32>]) -> Result<()> {
        if let Some(bad) = embeddings.iter().find(|embedding| embedding.len() != self.dim) {
            return Err(GraphError::Embedding(format!(
                "embedder returned dimension {} but the index is {}",
                bad.len(),
                self.dim
            )));
        }
        Ok(())
    }

    /// The prefixed doc-ids of `kind` documents whose stripped id is not in `live`.
    fn orphan_doc_ids(
        &self,
        kind: &str,
        prefix: &str,
        live: &[String],
    ) -> Result<Vec<String>> {
        let live: std::collections::HashSet<&str> = live.iter().map(String::as_str).collect();
        let filter = json!({ "kind": kind });
        let mut orphans = Vec::new();
        for record in self
            .engine
            .list_documents(&self.index_id, Some(&filter), None, None)?
        {
            let Some(doc_id) = record.get("document_id").and_then(Value::as_str) else {
                continue;
            };
            let id = strip_prefix(doc_id, prefix);
            if !live.contains(id) {
                orphans.push(doc_id.to_string());
            }
        }
        Ok(orphans)
    }
}

/// Build the engine-side as-of metadata filter. Always constrains `kind`, adds a
/// `type`/`relation` equality when given, and folds the temporal frame in:
///
/// - [`AsOf::Now`]: both open sentinels (`expired_at`/`invalid_at` still open).
/// - [`AsOf::At`]: `valid_at <= t` and `invalid_at > t` over the sortable strings.
/// - [`AsOf::All`]: no temporal clause (every version).
fn asof_filter(
    kind: &str,
    type_or_relation: Option<(&str, &str)>,
    as_of: AsOf,
) -> Option<Value> {
    let mut clauses: Vec<Value> = vec![json!({ "kind": kind })];
    if let Some((key, value)) = type_or_relation {
        clauses.push(json!({ key: value }));
    }
    match as_of {
        AsOf::Now => {
            clauses.push(json!({ "expired_at": encode_ts_open(None) }));
            clauses.push(json!({ "invalid_at": encode_ts_open(None) }));
        }
        AsOf::At(t) => {
            clauses.push(json!({ "valid_at": { "$lte": encode_ts(t) } }));
            clauses.push(json!({ "invalid_at": { "$gt": encode_ts(t) } }));
        }
        AsOf::All => {}
    }
    Some(json!({ "$and": clauses }))
}

/// Strip a known doc-id prefix to recover the entity/fact id.
fn strip_prefix<'a>(value: &'a str, prefix: &str) -> &'a str {
    value.strip_prefix(prefix).unwrap_or(value)
}

#[cfg(test)]
mod tests {
    use super::{SemanticIndex, INDEX_ID};
    use crate::error::Result;
    use crate::model::{AsOf, EmbedRole, Embedder, Entity, Fact, GraphConfig};
    use serde_json::Value;

    /// A deterministic bag-of-chars embedder: hash each char into one of `dim`
    /// buckets, then L2-normalize. Distinct texts land on distinct unit vectors, so
    /// membership (not exact order) is stable across the vector/hybrid paths.
    struct HashEmbedder {
        dim: usize,
    }

    impl HashEmbedder {
        fn vector(&self, text: &str) -> Vec<f32> {
            let mut v = vec![0.0f32; self.dim];
            for ch in text.chars() {
                v[(ch as usize) % self.dim] += 1.0;
            }
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in v.iter_mut() {
                    *x /= norm;
                }
            } else {
                v[0] = 1.0;
            }
            v
        }
    }

    impl Embedder for HashEmbedder {
        fn dimension(&self) -> usize {
            self.dim
        }
        fn embed(&self, texts: &[String], _role: EmbedRole) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|t| self.vector(t)).collect())
        }
    }

    fn config() -> GraphConfig {
        GraphConfig {
            vector_dim: 8,
            index_text: true,
            index_facts: true,
        }
    }

    fn entity(id: &str, entity_type: &str, label: &str) -> Entity {
        Entity {
            id: id.to_string(),
            entity_type: entity_type.to_string(),
            label: label.to_string(),
            properties: Value::Null,
            valid_at: None,
            invalid_at: None,
            created_at: 1_000,
            expired_at: None,
        }
    }

    fn fact(id: &str, src: &str, relation: &str, dst: &str, text: &str) -> Fact {
        Fact {
            id: id.to_string(),
            src: src.to_string(),
            relation: relation.to_string(),
            dst: dst.to_string(),
            fact: text.to_string(),
            properties: Value::Null,
            episodes: Vec::new(),
            valid_at: None,
            invalid_at: None,
            created_at: 1_000,
            expired_at: None,
            reference_time: None,
        }
    }

    fn ids(hits: &[super::IndexHit]) -> Vec<&str> {
        hits.iter().map(|h| h.id.as_str()).collect()
    }

    #[test]
    fn indexes_searches_and_scopes_by_as_of() {
        let embedder = HashEmbedder { dim: 8 };
        let mut index = SemanticIndex::open_in_memory(&config()).unwrap();

        let alice = entity("alice", "Person", "Alice software engineer");
        let acme = entity("acme", "Org", "Acme robotics corporation");
        index.index_entity(&alice, Some(&embedder), None).unwrap();
        index.index_entity(&acme, Some(&embedder), None).unwrap();
        assert_eq!(index.count().unwrap(), 2);

        // Both entities are returned (k >= corpus); assert membership, not order.
        let hits = index
            .semantic_entities(Some("Alice engineer"), None, Some(&embedder), 5, None, AsOf::Now)
            .unwrap();
        assert!(ids(&hits).contains(&"alice"), "alice must be found: {:?}", ids(&hits));
        assert!(ids(&hits).contains(&"acme"), "acme must be found: {:?}", ids(&hits));

        // The entity_type filter narrows to Person only.
        let persons = index
            .semantic_entities(Some("Alice"), None, Some(&embedder), 5, Some("Person"), AsOf::Now)
            .unwrap();
        assert!(ids(&persons).contains(&"alice"));
        assert!(!ids(&persons).contains(&"acme"), "Org must be excluded: {:?}", ids(&persons));

        // resolve_entity surfaces candidates (AsOf::All) for the caller's merge step.
        let candidates = index.resolve_entity("robotics", Some(&embedder), 5).unwrap();
        assert!(ids(&candidates).contains(&"acme"));

        // A live fact is found by semantic_facts under AsOf::Now. Give it a concrete
        // event-time start so the AsOf::At assertions below exercise a real
        // boundary (an open `valid_at` encodes to the epoch floor and would satisfy
        // `valid_at <= t` for every t; see encode_ts_start and the
        // open_start_as_of_consistency test).
        let mut f = fact("f1", "alice", "works_at", "acme", "Alice works at Acme robotics");
        f.valid_at = Some(1_000);
        index.index_fact(&f, Some(&embedder), None).unwrap();
        let live = index
            .semantic_facts(Some("works at Acme"), None, Some(&embedder), 5, None, AsOf::Now)
            .unwrap();
        assert!(live.iter().any(|h| h.id == "f1"), "live fact must be found: {:?}", ids(&live));

        // Close (invalidate) the fact and re-index the same doc-id: AsOf::Now must
        // now exclude it while AsOf::All still returns it.
        let closed = Fact {
            invalid_at: Some(2_000),
            expired_at: Some(2_000),
            ..f.clone()
        };
        index.index_fact(&closed, Some(&embedder), None).unwrap();

        let now = index
            .semantic_facts(Some("works at Acme"), None, Some(&embedder), 5, None, AsOf::Now)
            .unwrap();
        assert!(
            !now.iter().any(|h| h.id == "f1"),
            "invalidated fact must be excluded from AsOf::Now: {:?}",
            ids(&now)
        );
        let all = index
            .semantic_facts(Some("works at Acme"), None, Some(&embedder), 5, None, AsOf::All)
            .unwrap();
        assert!(
            all.iter().any(|h| h.id == "f1"),
            "invalidated fact must appear under AsOf::All: {:?}",
            ids(&all)
        );

        // AsOf::At sees the fact before it was invalidated, but not after.
        let before = index
            .semantic_facts(Some("works at Acme"), None, Some(&embedder), 5, None, AsOf::At(1_500))
            .unwrap();
        assert!(before.iter().any(|h| h.id == "f1"), "At(1500) is inside validity: {:?}", ids(&before));
        let after = index
            .semantic_facts(Some("works at Acme"), None, Some(&embedder), 5, None, AsOf::At(2_500))
            .unwrap();
        assert!(!after.iter().any(|h| h.id == "f1"), "At(2500) is past invalidation: {:?}", ids(&after));
    }

    #[test]
    fn vector_in_path_and_orphan_drop() {
        let mut index = SemanticIndex::open_in_memory(&config()).unwrap();
        let embedder = HashEmbedder { dim: 8 };

        // Vector-in path: no embedder, a precomputed embedding per entity.
        let a = entity("a", "Thing", "");
        let b = entity("b", "Thing", "");
        let va = embedder.vector("alpha thing");
        let vb = embedder.vector("bravo widget");
        index.index_entity(&a, None, Some(&va)).unwrap();
        index.index_entity(&b, None, Some(&vb)).unwrap();
        assert_eq!(index.count().unwrap(), 2);

        // Query by a precomputed embedding (no embedder needed).
        let hits = index
            .semantic_entities(None, Some(&va), None, 5, None, AsOf::All)
            .unwrap();
        assert!(ids(&hits).contains(&"a"), "vector query must surface a: {:?}", ids(&hits));

        // A wrong-dimension precomputed embedding is rejected up front.
        assert!(index
            .semantic_entities(None, Some(&[1.0, 0.0, 0.0]), None, 5, None, AsOf::All)
            .is_err());

        // drop_orphans removes docs whose id is not in the live set.
        let removed = index.drop_orphans(&["a".to_string()], &[]).unwrap();
        assert_eq!(removed, 1, "only b is an orphan");
        assert_eq!(index.count().unwrap(), 1);
        let remaining = index
            .semantic_entities(None, Some(&va), None, 5, None, AsOf::All)
            .unwrap();
        assert!(!ids(&remaining).contains(&"b"));

        // remove_entity clears the last document.
        index.remove_entity("a").unwrap();
        assert_eq!(index.count().unwrap(), 0);
    }

    #[test]
    fn facts_disabled_clears_and_skips_edge_indexing() {
        let mut index = SemanticIndex::open_in_memory(&GraphConfig {
            vector_dim: 8,
            index_text: true,
            index_facts: false,
        })
        .unwrap();
        let embedder = HashEmbedder { dim: 8 };

        let f = fact("f1", "a", "rel", "b", "some fact text");
        index.index_fact(&f, Some(&embedder), None).unwrap();
        // No fact document was written (edge indexing is off).
        assert_eq!(index.count().unwrap(), 0);
        let hits = index
            .semantic_facts(Some("some fact"), None, Some(&embedder), 5, None, AsOf::All)
            .unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn uses_the_default_index_id() {
        let index = SemanticIndex::open_in_memory(&config()).unwrap();
        assert_eq!(index.index_id, INDEX_ID);
    }
}
