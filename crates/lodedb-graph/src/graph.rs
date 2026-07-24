//! `TemporalGraph` — the public facade wiring the topology truth store, the
//! `lodedb-core` semantic index, and the bi-temporal helpers into one handle.
//! This is the surface the Python/Swift bindings marshal.
//!
//! Verb parity with Graphiti's `Graphiti` class, minus the LLM pipeline:
//! `add_episode` stores (no extraction); `add_fact` is the LLM-free `add_triplet`
//! analogue and performs invalidation; `search_subgraph` is semantic seeds + k-hop.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::error::{GraphError, Result};
use crate::index::SemanticIndex;
use crate::model::{
    AsOf, Direction, Embedder, Entity, EntityPropertyVersion, Episode, Fact, GraphConfig, Subgraph,
    TimeMs,
};
use crate::temporal;
use crate::topology::TopologyStore;

static ID_COUNTER: AtomicU64 = AtomicU64::new(0);
const INDEX_BATCH_SIZE: usize = 256;

/// Wall-clock epoch milliseconds (transaction-time stamp).
fn now_ms() -> TimeMs {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as TimeMs)
        .unwrap_or(0)
}

/// Capture a strict current frame once so SQL, index, and hydration checks share
/// exactly the same wall-clock instant.
pub fn now_valid_frame() -> AsOf {
    AsOf::NowValid(now_ms())
}

/// A unique id with a kind prefix (`ep`/`ent`/`f`). Includes the process id so two
/// writers on the same DB — or a restart within the same millisecond, which resets the
/// counter to 0 — cannot mint the same id and collapse bi-temporal fact history via id
/// reuse (`upsert_fact`'s `ON CONFLICT DO UPDATE` would otherwise overwrite a prior).
fn gen_id(prefix: &str) -> String {
    let n = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}-{:x}-{:x}-{:x}", now_ms(), std::process::id(), n)
}

/// A bi-temporal knowledge graph: authoritative topology + rebuildable semantic
/// index, over one directory.
pub struct TemporalGraph {
    topology: TopologyStore,
    index: SemanticIndex,
    embedder: Option<Box<dyn Embedder>>,
    #[allow(dead_code)]
    config: GraphConfig,
}

impl TemporalGraph {
    fn validate_open_config(config: &GraphConfig, embedder: Option<&dyn Embedder>) -> Result<()> {
        if config.vector_dim == 0 {
            return Err(GraphError::InvalidArgument(
                "vector_dim must be greater than zero".to_string(),
            ));
        }
        if let Some(embedder) = embedder {
            let actual = embedder.dimension();
            if actual != config.vector_dim {
                return Err(GraphError::InvalidArgument(format!(
                    "embedder dimension {actual} does not match vector_dim {}",
                    config.vector_dim
                )));
            }
        }
        Ok(())
    }

    /// Open (creating if needed) a graph rooted at `path`: `path/topology.sqlite3`
    /// (truth) + `path/index` (semantic). `embedder` drives the text-in path; pass
    /// `None` for a vector-in graph and use the `*_vec` verbs.
    pub fn open(
        path: &Path,
        config: GraphConfig,
        embedder: Option<Box<dyn Embedder>>,
    ) -> Result<Self> {
        Self::validate_open_config(&config, embedder.as_deref())?;
        std::fs::create_dir_all(path)
            .map_err(|e| GraphError::Topology(format!("create dir {}: {e}", path.display())))?;
        let topology = TopologyStore::open(&path.join("topology.sqlite3"))?;
        // Validate known topology metadata before opening/creating the derivative
        // index. Then let an existing legacy index validate its dimension before
        // claiming metadata that a failed open would make impossible to correct.
        topology.validate_configuration(&config)?;
        let index = SemanticIndex::open(&path.join("index"), &config)?;
        topology.configure(&config)?;
        let mut graph = TemporalGraph {
            topology,
            index,
            embedder,
            config,
        };
        if graph.index.was_created() {
            // A missing derived index is recoverable even when the last explicit
            // checkpoint left the topology's marker clean.
            graph.topology.mark_index_dirty()?;
        }
        graph.repair_index_if_dirty()?;
        Ok(graph)
    }

    /// Open a fully in-memory graph (tests).
    pub fn open_in_memory(
        config: GraphConfig,
        embedder: Option<Box<dyn Embedder>>,
    ) -> Result<Self> {
        Self::validate_open_config(&config, embedder.as_deref())?;
        let topology = TopologyStore::open_in_memory()?;
        topology.configure(&config)?;
        let index = SemanticIndex::open_in_memory(&config)?;
        Ok(TemporalGraph {
            topology,
            index,
            embedder,
            config,
        })
    }

    // -- episodes -----------------------------------------------------------

    /// Store a raw observation (no extraction). Returns its id. `mentions` records
    /// the episode → entity provenance links (Graphiti `MENTIONS`).
    pub fn add_episode(
        &mut self,
        source: &str,
        body: &str,
        occurred_at: TimeMs,
        properties: Value,
        mentions: &[String],
    ) -> Result<String> {
        self.add_episode_with_id(None, source, body, occurred_at, properties, mentions)
    }

    /// [`add_episode`] with an optional caller-stable id. Repeating the same id
    /// and payload is a no-op; reusing it for a different payload fails closed.
    pub fn add_episode_with_id(
        &mut self,
        caller_id: Option<&str>,
        source: &str,
        body: &str,
        occurred_at: TimeMs,
        properties: Value,
        mentions: &[String],
    ) -> Result<String> {
        if caller_id.is_some_and(|id| id.trim().is_empty()) {
            return Err(GraphError::InvalidArgument(
                "episode id must not be blank".to_string(),
            ));
        }
        let id = caller_id
            .map(str::to_string)
            .unwrap_or_else(|| gen_id("ep"));
        if let Some(existing) = self.topology.get_episode(&id)? {
            let mut expected_mentions: Vec<String> = mentions.to_vec();
            expected_mentions.sort();
            expected_mentions.dedup();
            let same = existing.source == source
                && existing.body == body
                && existing.occurred_at == occurred_at
                && canonical_properties(&existing.properties) == canonical_properties(&properties)
                && self.topology.episode_mentions(&id)? == expected_mentions;
            if same {
                return Ok(id);
            }
            return Err(GraphError::InvalidArgument(format!(
                "episode id {id:?} already exists with a different payload"
            )));
        }
        let episode = Episode {
            id: id.clone(),
            source: source.to_string(),
            body: body.to_string(),
            occurred_at,
            created_at: now_ms(),
            properties,
        };
        // One transaction: a bad mention id must not leave a half-written episode.
        self.topology
            .upsert_episode_with_mentions(&episode, mentions)?;
        Ok(id)
    }

    pub fn get_episode(&self, id: &str) -> Result<Option<Episode>> {
        self.topology.get_episode(id)
    }

    pub fn episodes(&self) -> Result<Vec<Episode>> {
        self.topology.list_episodes()
    }

    pub fn facts_by_episode(&self, id: &str) -> Result<Vec<Fact>> {
        self.topology.facts_by_episode(id)
    }

    // -- entities -----------------------------------------------------------

    /// Create or replace an entity (upsert by stable id) and (re)index it.
    #[allow(clippy::too_many_arguments)]
    pub fn upsert_entity(
        &mut self,
        id: &str,
        entity_type: &str,
        label: &str,
        properties: Value,
        valid_at: Option<TimeMs>,
        invalid_at: Option<TimeMs>,
    ) -> Result<String> {
        self.upsert_entity_with_sources(
            id,
            entity_type,
            label,
            properties,
            valid_at,
            invalid_at,
            &BTreeMap::new(),
        )
    }

    /// [`upsert_entity`] with per-property episode provenance.
    #[allow(clippy::too_many_arguments)]
    pub fn upsert_entity_with_sources(
        &mut self,
        id: &str,
        entity_type: &str,
        label: &str,
        properties: Value,
        valid_at: Option<TimeMs>,
        invalid_at: Option<TimeMs>,
        property_sources: &BTreeMap<String, String>,
    ) -> Result<String> {
        if id.trim().is_empty() {
            return Err(GraphError::InvalidArgument("entity id is required".into()));
        }
        let recorded_at = now_ms();
        let existing = self.topology.get_entity(id)?;
        let created_at = existing.map(|e| e.created_at).unwrap_or(recorded_at);
        let entity = Entity {
            id: id.to_string(),
            entity_type: entity_type.to_string(),
            label: label.to_string(),
            properties,
            valid_at,
            invalid_at,
            created_at,
            expired_at: None,
        };
        if self.embedder.is_none() && !entity.label.trim().is_empty() {
            return Err(GraphError::InvalidArgument(
                "upsert_entity needs the graph's embedder; use upsert_entity_vec on a vector-in graph"
                    .into(),
            ));
        }
        self.topology.mark_index_dirty()?;
        let result = {
            let topology = &self.topology;
            let index = &mut self.index;
            let embedder = self.embedder.as_deref();
            topology.upsert_entity_with_lineage_before_commit(
                &entity,
                None,
                property_sources,
                recorded_at,
                || index.index_entity(&entity, embedder, None),
            )
        };
        self.finish_indexed_mutation(result)?;
        Ok(entity.id)
    }

    /// Create or replace an entity indexed by a caller-supplied vector (vector-in).
    pub fn upsert_entity_vec(
        &mut self,
        id: &str,
        entity_type: &str,
        label: &str,
        properties: Value,
        embedding: &[f32],
        valid_at: Option<TimeMs>,
        invalid_at: Option<TimeMs>,
    ) -> Result<String> {
        self.upsert_entity_vec_with_sources(
            id,
            entity_type,
            label,
            properties,
            embedding,
            valid_at,
            invalid_at,
            &BTreeMap::new(),
        )
    }

    /// Vector-in [`upsert_entity_with_sources`].
    #[allow(clippy::too_many_arguments)]
    pub fn upsert_entity_vec_with_sources(
        &mut self,
        id: &str,
        entity_type: &str,
        label: &str,
        properties: Value,
        embedding: &[f32],
        valid_at: Option<TimeMs>,
        invalid_at: Option<TimeMs>,
        property_sources: &BTreeMap<String, String>,
    ) -> Result<String> {
        if id.trim().is_empty() {
            return Err(GraphError::InvalidArgument("entity id is required".into()));
        }
        self.index.validate_vector(embedding)?;
        let recorded_at = now_ms();
        let created_at = self
            .topology
            .get_entity(id)?
            .map(|e| e.created_at)
            .unwrap_or(recorded_at);
        let entity = Entity {
            id: id.to_string(),
            entity_type: entity_type.to_string(),
            label: label.to_string(),
            properties,
            valid_at,
            invalid_at,
            created_at,
            expired_at: None,
        };
        self.topology.mark_index_dirty()?;
        let result = {
            let topology = &self.topology;
            let index = &mut self.index;
            topology.upsert_entity_with_lineage_before_commit(
                &entity,
                Some(embedding),
                property_sources,
                recorded_at,
                || index.index_entity(&entity, None, Some(embedding)),
            )
        };
        self.finish_indexed_mutation(result)?;
        Ok(entity.id)
    }

    pub fn get_entity(&self, id: &str) -> Result<Option<Entity>> {
        self.topology.get_entity(id)
    }

    pub fn entity_property_history(
        &self,
        id: &str,
        key: Option<&str>,
    ) -> Result<Vec<EntityPropertyVersion>> {
        self.topology.entity_property_history(id, key)
    }

    /// Complete-set enumeration by type (nil = all) in a temporal frame.
    pub fn entities(&self, entity_type: Option<&str>, as_of: AsOf) -> Result<Vec<Entity>> {
        self.topology.list_entities(entity_type, as_of)
    }

    // -- facts --------------------------------------------------------------

    /// Assert a fact. Each call is a distinct, uniquely-identified assertion (so
    /// history is preserved). When `invalidates` is given, those prior facts are
    /// closed in the same logical step using Graphiti's rule
    /// (`invalid_at = new valid_at`, `expired_at = now`). Returns the new fact id.
    #[allow(clippy::too_many_arguments)]
    pub fn add_fact(
        &mut self,
        src: &str,
        relation: &str,
        dst: &str,
        fact_text: &str,
        properties: Value,
        episodes: Vec<String>,
        valid_at: Option<TimeMs>,
        invalidates: &[String],
    ) -> Result<String> {
        self.add_fact_inner(
            None,
            src,
            relation,
            dst,
            fact_text,
            properties,
            episodes,
            valid_at,
            invalidates,
            None,
        )
    }

    /// Vector-in [`add_fact`]: index the fact by a caller-supplied `fact_embedding`
    /// rather than the engine embedder. This is the on-device path — Swift embeds the
    /// fact text and passes the vector, so the graph needs no embedder over the FFI.
    #[allow(clippy::too_many_arguments)]
    pub fn add_fact_vec(
        &mut self,
        src: &str,
        relation: &str,
        dst: &str,
        fact_text: &str,
        properties: Value,
        episodes: Vec<String>,
        valid_at: Option<TimeMs>,
        invalidates: &[String],
        fact_embedding: &[f32],
    ) -> Result<String> {
        self.add_fact_inner(
            None,
            src,
            relation,
            dst,
            fact_text,
            properties,
            episodes,
            valid_at,
            invalidates,
            Some(fact_embedding),
        )
    }

    /// [`add_fact`] with an optional caller-stable id.
    #[allow(clippy::too_many_arguments)]
    pub fn add_fact_with_id(
        &mut self,
        caller_id: Option<&str>,
        src: &str,
        relation: &str,
        dst: &str,
        fact_text: &str,
        properties: Value,
        episodes: Vec<String>,
        valid_at: Option<TimeMs>,
        invalidates: &[String],
    ) -> Result<String> {
        self.add_fact_inner(
            caller_id,
            src,
            relation,
            dst,
            fact_text,
            properties,
            episodes,
            valid_at,
            invalidates,
            None,
        )
    }

    /// Vector-in [`add_fact_with_id`].
    #[allow(clippy::too_many_arguments)]
    pub fn add_fact_vec_with_id(
        &mut self,
        caller_id: Option<&str>,
        src: &str,
        relation: &str,
        dst: &str,
        fact_text: &str,
        properties: Value,
        episodes: Vec<String>,
        valid_at: Option<TimeMs>,
        invalidates: &[String],
        fact_embedding: &[f32],
    ) -> Result<String> {
        self.add_fact_inner(
            caller_id,
            src,
            relation,
            dst,
            fact_text,
            properties,
            episodes,
            valid_at,
            invalidates,
            Some(fact_embedding),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn add_fact_inner(
        &mut self,
        caller_id: Option<&str>,
        src: &str,
        relation: &str,
        dst: &str,
        fact_text: &str,
        properties: Value,
        episodes: Vec<String>,
        valid_at: Option<TimeMs>,
        invalidates: &[String],
        fact_embedding: Option<&[f32]>,
    ) -> Result<String> {
        if caller_id.is_some_and(|id| id.trim().is_empty()) {
            return Err(GraphError::InvalidArgument(
                "fact id must not be blank".to_string(),
            ));
        }
        for (name, value) in [("src", src), ("relation", relation), ("dst", dst)] {
            if value.trim().is_empty() {
                return Err(GraphError::InvalidArgument(format!(
                    "fact {name} is required"
                )));
            }
        }
        if let Some(embedding) = fact_embedding {
            self.index.validate_vector(embedding)?;
        } else if self.embedder.is_none() && self.config.index_facts && !fact_text.trim().is_empty()
        {
            return Err(GraphError::InvalidArgument(
                "add_fact needs the graph's embedder; use add_fact_vec on a vector-in graph".into(),
            ));
        }
        let now = now_ms();
        let reference_time = self.reference_time_for(&episodes)?;
        let effective = reference_time.unwrap_or(now);
        // A superseding fact with no explicit start begins when the supersession is
        // observed (Graphiti defaults `valid_at` to the episode's reference_time), so it
        // does not overlap the prior it replaces: the prior's `invalid_at` and the new
        // fact's `valid_at` then meet exactly at `effective`. A standalone undated fact
        // (no invalidates) keeps `valid_at = None` — genuinely-unknown, unbounded start.
        let valid_at = if valid_at.is_none() && !invalidates.is_empty() {
            Some(effective)
        } else {
            valid_at
        };
        let id = caller_id.map(str::to_string).unwrap_or_else(|| gen_id("f"));
        if let Some(existing) = self.topology.get_fact(&id)? {
            let same = existing.src == src
                && existing.relation == relation
                && existing.dst == dst
                && existing.fact == fact_text
                && canonical_properties(&existing.properties) == canonical_properties(&properties)
                && existing.episodes == episodes
                && existing.valid_at == valid_at
                && existing.reference_time == reference_time;
            if same {
                return Ok(id);
            }
            return Err(GraphError::InvalidArgument(format!(
                "fact id {id:?} already exists with a different payload"
            )));
        }
        let fact = Fact {
            id,
            src: src.to_string(),
            relation: relation.to_string(),
            dst: dst.to_string(),
            fact: fact_text.to_string(),
            properties,
            episodes,
            valid_at,
            invalid_at: None,
            created_at: now,
            expired_at: None,
            reference_time,
        };
        // Close the superseded priors AND insert the replacement in ONE topology
        // transaction, so a crash can never leave priors closed with no replacement (an
        // event-time validity gap). A prior's event-time end is the new fact's valid_at
        // (now backfilled above when it was open) — never left open. Duplicate ids
        // collapse here; a prior that does not exist or is already expired fails the
        // whole call (see `supersede_and_insert`), so a typo'd id cannot silently
        // leave its target live.
        let mut seen = std::collections::BTreeSet::new();
        let priors: Vec<(String, Option<TimeMs>, TimeMs)> = invalidates
            .iter()
            .filter(|id| seen.insert(id.as_str()))
            .map(|id| {
                let (inv, exp) = temporal::supersede_timestamps(valid_at, effective, now);
                (id.clone(), inv, exp)
            })
            .collect();
        self.topology.mark_index_dirty()?;
        let result = {
            let topology = &self.topology;
            let index = &mut self.index;
            let graph_embedder = self.embedder.as_deref();
            topology.supersede_and_insert_before_commit(&priors, &fact, fact_embedding, |closed| {
                let mut records: Vec<(&Fact, Option<&[f32]>)> = closed
                    .iter()
                    .map(|record| (&record.fact, record.vector.as_deref()))
                    .collect();
                records.push((&fact, fact_embedding));
                for batch in records.chunks(INDEX_BATCH_SIZE) {
                    index.index_facts(batch, graph_embedder)?;
                }
                Ok(())
            })
        };
        self.finish_indexed_mutation(result)?;
        Ok(fact.id)
    }

    /// Close a fact's validity without a replacement (an explicit end / retraction).
    pub fn invalidate_fact(&mut self, id: &str, invalid_at: Option<TimeMs>) -> Result<bool> {
        self.topology.mark_index_dirty()?;
        let result = {
            let topology = &self.topology;
            let index = &mut self.index;
            let graph_embedder = self.embedder.as_deref();
            topology.close_fact_before_commit(id, invalid_at, now_ms(), |closed| {
                if let Some(record) = closed {
                    let embedder = if record.vector.is_some() {
                        None
                    } else {
                        graph_embedder
                    };
                    index.index_fact(&record.fact, embedder, record.vector.as_deref())?;
                }
                Ok(())
            })
        };
        self.finish_indexed_mutation(result)
    }

    pub fn get_fact(&self, id: &str) -> Result<Option<Fact>> {
        self.topology.get_fact(id)
    }

    /// The producing episode's event time becomes the fact's `reference_time`
    /// (Graphiti parity): use the earliest `occurred_at` among the referenced
    /// episodes, if any.
    fn reference_time_for(&self, episodes: &[String]) -> Result<Option<TimeMs>> {
        let mut earliest: Option<TimeMs> = None;
        for ep_id in episodes {
            if let Some(ep) = self.topology.get_episode(ep_id)? {
                earliest = Some(earliest.map_or(ep.occurred_at, |e| e.min(ep.occurred_at)));
            }
        }
        Ok(earliest)
    }

    // -- traversal ----------------------------------------------------------

    /// Facts incident to `id` in `direction`, optionally by `relation`, as-of `as_of`.
    pub fn neighbors(
        &self,
        id: &str,
        direction: Direction,
        relation: Option<&str>,
        as_of: AsOf,
    ) -> Result<Vec<Fact>> {
        self.topology
            .facts_for(&[id.to_string()], direction, relation, as_of)
    }

    /// Deterministic k-hop neighbourhood (BFS) around `seeds`, in a temporal frame.
    pub fn k_hop(
        &self,
        seeds: &[String],
        k: usize,
        direction: Direction,
        as_of: AsOf,
    ) -> Result<Subgraph> {
        self.k_hop_filtered(seeds, k, direction, as_of, None)
    }

    /// [`k_hop`] with an authorization/support predicate applied to both facts
    /// and entities before either can influence the next BFS frontier.
    pub fn k_hop_filtered(
        &self,
        seeds: &[String],
        k: usize,
        direction: Direction,
        as_of: AsOf,
        predicate: Option<&Value>,
    ) -> Result<Subgraph> {
        let predicate = predicate.map(validate_property_predicate).transpose()?;
        let seed_entities = self.topology.get_entities(seeds)?;
        let mut visited: BTreeSet<String> = seed_entities
            .into_iter()
            .filter(|entity| {
                entity.matches(as_of)
                    && property_matches(&entity.properties, predicate.as_ref())
            })
            .map(|entity| entity.id)
            .collect();
        let mut frontier: Vec<String> = visited.iter().cloned().collect();
        let mut facts_by_id: BTreeMap<String, Fact> = BTreeMap::new();
        for _hop in 0..k {
            if frontier.is_empty() {
                break;
            }
            let candidates: Vec<Fact> = self
                .topology
                .facts_for(&frontier, direction, None, as_of)?
                .into_iter()
                .filter(|fact| property_matches(&fact.properties, predicate.as_ref()))
                .collect();
            let endpoint_ids: Vec<String> = candidates
                .iter()
                .flat_map(|fact| [fact.src.clone(), fact.dst.clone()])
                .collect();
            let allowed: BTreeSet<String> = self
                .topology
                .get_entities(&endpoint_ids)?
                .into_iter()
                .filter(|entity| {
                    entity.matches(as_of)
                        && property_matches(&entity.properties, predicate.as_ref())
                })
                .map(|entity| entity.id)
                .collect();
            let mut next: Vec<String> = Vec::new();
            for f in candidates {
                if !allowed.contains(&f.src) || !allowed.contains(&f.dst) {
                    continue;
                }
                for endpoint in [&f.src, &f.dst] {
                    if visited.insert(endpoint.clone()) {
                        next.push(endpoint.clone());
                    }
                }
                facts_by_id.insert(f.id.clone(), f);
            }
            frontier = next;
        }
        let entities = self
            .topology
            .get_entities(&visited.iter().cloned().collect::<Vec<_>>())?
            .into_iter()
            .map(|e| (e.id.clone(), e))
            .collect();
        Ok(Subgraph {
            entities,
            facts: facts_by_id.into_values().collect(),
            seeds: Vec::new(),
        })
    }

    // -- semantic retrieval -------------------------------------------------

    /// Top-`k` entities for `query` (hybrid) or `embedding` (vector), optionally by
    /// `entity_type`, time-scoped by `as_of`.
    pub fn semantic_entities(
        &self,
        query: Option<&str>,
        embedding: Option<&[f32]>,
        k: usize,
        entity_type: Option<&str>,
        as_of: AsOf,
    ) -> Result<Vec<(f32, Entity)>> {
        self.semantic_entities_filtered(query, embedding, k, entity_type, as_of, None)
    }

    /// [`semantic_entities`] with an authorization/support predicate pushed into
    /// the semantic index before candidate generation.
    #[allow(clippy::too_many_arguments)]
    pub fn semantic_entities_filtered(
        &self,
        query: Option<&str>,
        embedding: Option<&[f32]>,
        k: usize,
        entity_type: Option<&str>,
        as_of: AsOf,
        predicate: Option<&Value>,
    ) -> Result<Vec<(f32, Entity)>> {
        let validated = predicate.map(validate_property_predicate).transpose()?;
        let hits = self.index.semantic_entities_filtered(
            query,
            embedding,
            self.embedder.as_deref(),
            k,
            entity_type,
            as_of,
            predicate,
        )?;
        let ids: Vec<String> = hits.iter().map(|h| h.id.clone()).collect();
        let by_id: std::collections::HashMap<String, Entity> = self
            .topology
            .get_entities(&ids)?
            .into_iter()
            .map(|e| (e.id.clone(), e))
            .collect();
        Ok(hits
            .into_iter()
            .filter_map(|h| by_id.get(&h.id).cloned().map(|e| (h.score, e)))
            // Re-check the frame against the authoritative row; a stale index hit
            // (crash between topology commit and index refresh) must not leak an
            // expired entity into a scoped read.
            .filter(|(_score, e)| e.matches(as_of))
            .filter(|(_score, e)| property_matches(&e.properties, validated.as_ref()))
            .collect())
    }

    /// Top-`k` facts for `query`/`embedding` (Graphiti's default search shape),
    /// optionally by `relation`, time-scoped by `as_of`.
    pub fn semantic_facts(
        &self,
        query: Option<&str>,
        embedding: Option<&[f32]>,
        k: usize,
        relation: Option<&str>,
        as_of: AsOf,
    ) -> Result<Vec<(f32, Fact)>> {
        self.semantic_facts_filtered(query, embedding, k, relation, as_of, None)
    }

    /// [`semantic_facts`] with an authorization/support predicate pushed into
    /// the semantic index before candidate generation.
    #[allow(clippy::too_many_arguments)]
    pub fn semantic_facts_filtered(
        &self,
        query: Option<&str>,
        embedding: Option<&[f32]>,
        k: usize,
        relation: Option<&str>,
        as_of: AsOf,
        predicate: Option<&Value>,
    ) -> Result<Vec<(f32, Fact)>> {
        let validated = predicate.map(validate_property_predicate).transpose()?;
        let hits = self.index.semantic_facts_filtered(
            query,
            embedding,
            self.embedder.as_deref(),
            k,
            relation,
            as_of,
            predicate,
        )?;
        let ids: Vec<String> = hits.iter().map(|hit| hit.id.clone()).collect();
        let by_id: std::collections::HashMap<String, Fact> = self
            .topology
            .get_facts(&ids)?
            .into_iter()
            .map(|fact| (fact.id.clone(), fact))
            .collect();
        Ok(hits
            .into_iter()
            .filter_map(|hit| by_id.get(&hit.id).cloned().map(|fact| (hit.score, fact)))
            // Re-check against authoritative rows so a stale derivative cannot
            // leak an expired fact into a scoped read.
            .filter(|(_score, fact)| fact.matches(as_of))
            .filter(|(_score, fact)| property_matches(&fact.properties, validated.as_ref()))
            .collect())
    }

    /// Semantic seed entities + k-hop expansion (the headline query), time-scoped.
    #[allow(clippy::too_many_arguments)]
    pub fn search_subgraph(
        &self,
        query: Option<&str>,
        embedding: Option<&[f32]>,
        k: usize,
        hops: usize,
        direction: Direction,
        entity_type: Option<&str>,
        as_of: AsOf,
    ) -> Result<Subgraph> {
        self.search_subgraph_filtered(
            query,
            embedding,
            k,
            hops,
            direction,
            entity_type,
            None,
            as_of,
            None,
            "entity",
        )
    }

    /// Search from semantic entity seeds, fact seeds, or both. A property
    /// predicate is applied before seed ranking and before every traversal hop.
    #[allow(clippy::too_many_arguments)]
    pub fn search_subgraph_filtered(
        &self,
        query: Option<&str>,
        embedding: Option<&[f32]>,
        k: usize,
        hops: usize,
        direction: Direction,
        entity_type: Option<&str>,
        relation: Option<&str>,
        as_of: AsOf,
        predicate: Option<&Value>,
        seed_kind: &str,
    ) -> Result<Subgraph> {
        if !matches!(seed_kind, "entity" | "fact" | "both") {
            return Err(GraphError::InvalidArgument(format!(
                "seed_kind must be entity|fact|both, got {seed_kind:?}"
            )));
        }
        let validated_predicate = predicate.map(validate_property_predicate).transpose()?;
        let entity_seeds = if matches!(seed_kind, "entity" | "both") {
            self.semantic_entities_filtered(query, embedding, k, entity_type, as_of, predicate)?
        } else {
            Vec::new()
        };
        let mut fact_seeds = if matches!(seed_kind, "fact" | "both") {
            self.semantic_facts_filtered(query, embedding, k, relation, as_of, predicate)?
        } else {
            Vec::new()
        };
        if !fact_seeds.is_empty() {
            let endpoint_ids: Vec<String> = fact_seeds
                .iter()
                .flat_map(|(_score, fact)| [fact.src.clone(), fact.dst.clone()])
                .collect();
            let allowed_endpoints: BTreeSet<String> = self
                .topology
                .get_entities(&endpoint_ids)?
                .into_iter()
                .filter(|entity| {
                    entity.matches(as_of)
                        && property_matches(&entity.properties, validated_predicate.as_ref())
                })
                .map(|entity| entity.id)
                .collect();
            fact_seeds.retain(|(_score, fact)| {
                allowed_endpoints.contains(&fact.src) && allowed_endpoints.contains(&fact.dst)
            });
        }
        let mut seed_ids: Vec<String> = entity_seeds
            .iter()
            .map(|(_score, entity)| entity.id.clone())
            .collect();
        for (_score, fact) in &fact_seeds {
            seed_ids.push(fact.src.clone());
            seed_ids.push(fact.dst.clone());
        }
        seed_ids.sort();
        seed_ids.dedup();

        let mut subgraph = self.k_hop_filtered(&seed_ids, hops, direction, as_of, predicate)?;
        for (score, entity) in &entity_seeds {
            subgraph.entities.insert(entity.id.clone(), entity.clone());
            subgraph.seeds.push((entity.id.clone(), *score));
        }
        let mut existing_fact_ids: BTreeSet<String> =
            subgraph.facts.iter().map(|fact| fact.id.clone()).collect();
        for (score, fact) in fact_seeds {
            if existing_fact_ids.insert(fact.id.clone()) {
                subgraph.facts.push(fact.clone());
            }
            for entity in self
                .topology
                .get_entities(&[fact.src.clone(), fact.dst.clone()])?
                .into_iter()
                .filter(|entity| {
                    property_matches(&entity.properties, validated_predicate.as_ref())
                })
            {
                subgraph.entities.insert(entity.id.clone(), entity);
            }
            subgraph.seeds.push((format!("fact:{}", fact.id), score));
        }
        Ok(subgraph)
    }

    /// Candidate entities matching `name` for the caller's resolution step
    /// (embedding + lexical); the caller decides the merge.
    pub fn resolve_entity(&self, name: &str, k: usize) -> Result<Vec<(f32, Entity)>> {
        self.resolve_entity_filtered(name, k, None)
    }

    /// [`resolve_entity`] with a predicate applied before candidate ranking.
    pub fn resolve_entity_filtered(
        &self,
        name: &str,
        k: usize,
        predicate: Option<&Value>,
    ) -> Result<Vec<(f32, Entity)>> {
        let validated = predicate.map(validate_property_predicate).transpose()?;
        let hits =
            self.index
                .resolve_entity_filtered(name, self.embedder.as_deref(), k, predicate)?;
        let ids: Vec<String> = hits.iter().map(|hit| hit.id.clone()).collect();
        let by_id: std::collections::HashMap<String, Entity> = self
            .topology
            .get_entities(&ids)?
            .into_iter()
            .map(|entity| (entity.id.clone(), entity))
            .collect();
        Ok(hits
            .into_iter()
            .filter_map(|hit| {
                by_id
                    .get(&hit.id)
                    .cloned()
                    .map(|entity| (hit.score, entity))
            })
            .filter(|(_score, entity)| property_matches(&entity.properties, validated.as_ref()))
            .collect())
    }

    /// Every fact ever touching an entity, all frames (history).
    pub fn history(&self, entity_id: &str) -> Result<Vec<Fact>> {
        self.topology.history(entity_id)
    }

    // -- maintenance --------------------------------------------------------

    /// Remove an entity and its incident facts from both stores.
    pub fn remove_entity(&mut self, id: &str) -> Result<bool> {
        self.topology.mark_index_dirty()?;
        let result = {
            let topology = &self.topology;
            let index = &mut self.index;
            topology.remove_entity_before_commit(id, |removed_fact_ids| {
                index.remove_entity(id)?;
                index.remove_facts(removed_fact_ids)
            })
        };
        self.finish_indexed_mutation(result)
            .map(|(existed, _)| existed)
    }

    /// Hard-remove a single fact from both stores (Graphiti edge deletion). Prefer
    /// [`TemporalGraph::invalidate_fact`], which preserves history; this deletes the
    /// row outright. Returns whether the fact existed.
    pub fn remove_fact(&mut self, id: &str) -> Result<bool> {
        self.topology.mark_index_dirty()?;
        let result = {
            let topology = &self.topology;
            let index = &mut self.index;
            topology.remove_fact_before_commit(id, || index.remove_fact(id))
        };
        self.finish_indexed_mutation(result)
    }

    /// Remove an episode and roll back facts for which it is the originating
    /// provenance entry. Facts with other primary support remain and only lose
    /// this episode link.
    pub fn remove_episode(&mut self, id: &str) -> Result<bool> {
        self.topology.mark_index_dirty()?;
        let result = {
            let topology = &self.topology;
            let index = &mut self.index;
            topology.remove_episode_before_commit(id, |removed_fact_ids| {
                index.remove_facts(removed_fact_ids)
            })
        };
        self.finish_indexed_mutation(result)
    }

    /// Rebuild the semantic index from the topology truth store: drop orphans, then
    /// re-index every entity and fact. Makes the index a throwaway artifact.
    pub fn reindex(&mut self) -> Result<ReindexStats> {
        self.topology.mark_index_dirty()?;
        let result = self.reindex_internal();
        if result.is_ok() {
            self.index.persist()?;
            self.topology.mark_index_clean()?;
        }
        result
    }

    fn reindex_internal(&mut self) -> Result<ReindexStats> {
        let entities = self.topology.iter_entity_index_records()?;
        let facts = self.topology.iter_fact_index_records()?;
        let live_entity_ids: Vec<String> = entities
            .iter()
            .map(|record| record.entity.id.clone())
            .collect();
        let live_fact_ids: Vec<String> =
            facts.iter().map(|record| record.fact.id.clone()).collect();
        for record in &entities {
            if record.vector.is_none()
                && self.embedder.is_none()
                && !record.entity.label.trim().is_empty()
            {
                return Err(GraphError::InvalidArgument(format!(
                    "entity {:?} has no retained vector and the graph has no embedder",
                    record.entity.id
                )));
            }
        }
        for record in &facts {
            if self.config.index_facts
                && record.vector.is_none()
                && self.embedder.is_none()
                && !record.fact.fact.trim().is_empty()
            {
                return Err(GraphError::InvalidArgument(format!(
                    "fact {:?} has no retained vector and the graph has no embedder",
                    record.fact.id
                )));
            }
        }
        // Do not mutate the derivative until every topology record is known to be
        // rebuildable. In particular, a legacy vector-in graph with no retained
        // vectors must fail without first deleting orphan/index documents.
        let removed_orphans = self.index.drop_orphans(&live_entity_ids, &live_fact_ids)?;
        for batch in entities.chunks(INDEX_BATCH_SIZE) {
            let documents: Vec<(&Entity, Option<&[f32]>)> = batch
                .iter()
                .map(|record| (&record.entity, record.vector.as_deref()))
                .collect();
            self.index
                .index_entities(&documents, self.embedder.as_deref())?;
        }
        for batch in facts.chunks(INDEX_BATCH_SIZE) {
            let documents: Vec<(&Fact, Option<&[f32]>)> = batch
                .iter()
                .map(|record| (&record.fact, record.vector.as_deref()))
                .collect();
            self.index
                .index_facts(&documents, self.embedder.as_deref())?;
        }
        Ok(ReindexStats {
            reindexed_entities: entities.len(),
            reindexed_facts: facts.len(),
            removed_orphans,
        })
    }

    fn repair_index_if_dirty(&mut self) -> Result<()> {
        if self.topology.index_dirty()? {
            self.reindex_internal()?;
            // A clean marker means the derivative is durable, not merely correct
            // in this process. Persist before clearing so a crash between these
            // steps can only cause a harmless extra rebuild.
            self.index.persist()?;
            self.topology.mark_index_clean()?;
        }
        Ok(())
    }

    /// Complete an index-coordinated topology transaction. The dirty marker is
    /// durable before either store changes. A callback/index/commit failure rolls
    /// the topology transaction back and rebuilds any partially changed derivative
    /// before returning the original error.
    fn finish_indexed_mutation<T>(&mut self, result: Result<T>) -> Result<T> {
        match result {
            // The core's WAL makes a successful mutation reopenable even before
            // an explicit generation checkpoint. Clear only after the topology
            // transaction also committed; every earlier crash window remains
            // covered by the durable dirty marker.
            Ok(value) => {
                self.topology.mark_index_clean()?;
                Ok(value)
            }
            Err(error) => match self.reindex_internal() {
                // Failure paths are rare, so make the repair durable before
                // returning. This also avoids turning a rejected validation-only
                // mutation into needless recovery work on the next open.
                Ok(_) => {
                    if let Err(repair_error) = self.index.persist() {
                        return Err(GraphError::Internal(format!(
                            "{error}; repaired semantic index could not be persisted: {repair_error}"
                        )));
                    }
                    if let Err(marker_error) = self.topology.mark_index_clean() {
                        return Err(GraphError::Internal(format!(
                            "{error}; repaired semantic index was persisted but its clean marker failed: {marker_error}"
                        )));
                    }
                    Err(error)
                }
                Err(repair_error) => Err(GraphError::Internal(format!(
                    "{error}; semantic-index repair also failed: {repair_error}"
                ))),
            },
        }
    }

    /// Node/fact counts and the index document count.
    pub fn stats(&self) -> Result<GraphStats> {
        Ok(GraphStats {
            entities: self.topology.entity_count()?,
            facts: self.topology.fact_count()?,
            indexed_documents: self.index.count()?,
        })
    }

    /// Checkpoint the semantic index (the topology store autocommits per write).
    pub fn persist(&mut self) -> Result<()> {
        self.index.persist()?;
        self.topology.mark_index_clean()
    }
}

fn validate_property_predicate(predicate: &Value) -> Result<Value> {
    lodedb_core::filter::coerce_sdk_filter(predicate)
        .map_err(|error| GraphError::InvalidArgument(error.to_string()))
}

fn canonical_properties(properties: &Value) -> Value {
    if properties.is_null() {
        serde_json::json!({})
    } else {
        properties.clone()
    }
}

fn property_matches(properties: &Value, predicate: Option<&Value>) -> bool {
    let Some(predicate) = predicate else {
        return true;
    };
    let mut metadata = BTreeMap::new();
    if let Some(object) = properties.as_object() {
        for (key, value) in object {
            let encoded = match value {
                Value::Null => Some(String::new()),
                Value::String(text) => Some(text.clone()),
                Value::Bool(flag) => Some(if *flag { "true" } else { "false" }.to_string()),
                Value::Number(number) => Some(number.to_string()),
                Value::Array(_) | Value::Object(_) => None,
            };
            if let Some(encoded) = encoded {
                metadata.insert(key.clone(), encoded);
            }
        }
    }
    lodedb_core::filter::matches_metadata_filter(&metadata, predicate).unwrap_or(false)
}

/// Counts returned by [`TemporalGraph::reindex`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReindexStats {
    pub reindexed_entities: usize,
    pub reindexed_facts: usize,
    pub removed_orphans: usize,
}

/// Counts returned by [`TemporalGraph::stats`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphStats {
    pub entities: usize,
    pub facts: usize,
    pub indexed_documents: usize,
}
