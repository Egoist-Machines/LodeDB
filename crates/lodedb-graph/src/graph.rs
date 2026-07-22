//! `TemporalGraph` — the public facade wiring the topology truth store, the
//! `lodedb-core` semantic index, and the bi-temporal helpers into one handle.
//!
//! This is the integration layer (owned by Wave 2): it composes the modules and is
//! the surface the Python/Swift bindings marshal. The module methods it calls are
//! implemented by Wave 1 (topology / index / temporal); until then the facade
//! compiles and each verb panics with `unimplemented!` from the module it drives.
//!
//! Verb parity with Graphiti's `Graphiti` class, minus the LLM pipeline:
//! `add_episode` stores (no extraction); `add_fact` is the LLM-free `add_triplet`
//! analogue and performs invalidation; `search_subgraph` is semantic seeds + k-hop.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::error::{GraphError, Result};
use crate::index::SemanticIndex;
use crate::model::{
    AsOf, Direction, Embedder, Entity, Episode, Fact, GraphConfig, Subgraph, TimeMs,
};
use crate::temporal;
use crate::topology::TopologyStore;

static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Wall-clock epoch milliseconds (transaction-time stamp).
fn now_ms() -> TimeMs {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as TimeMs)
        .unwrap_or(0)
}

/// A process-unique id with a kind prefix (`ep`/`ent`/`f`). Unique per assertion so
/// bi-temporal fact history is never collapsed by id reuse.
fn gen_id(prefix: &str) -> String {
    let n = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}-{:x}-{:x}", now_ms(), n)
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
    /// Open (creating if needed) a graph rooted at `path`: `path/topology.sqlite3`
    /// (truth) + `path/index` (semantic). `embedder` drives the text-in path; pass
    /// `None` for a vector-in graph and use the `*_vec` verbs.
    pub fn open(path: &Path, config: GraphConfig, embedder: Option<Box<dyn Embedder>>) -> Result<Self> {
        std::fs::create_dir_all(path)
            .map_err(|e| GraphError::Topology(format!("create dir {}: {e}", path.display())))?;
        let topology = TopologyStore::open(&path.join("topology.sqlite3"))?;
        let index = SemanticIndex::open(&path.join("index"), &config)?;
        Ok(TemporalGraph { topology, index, embedder, config })
    }

    /// Open a fully in-memory graph (tests).
    pub fn open_in_memory(config: GraphConfig, embedder: Option<Box<dyn Embedder>>) -> Result<Self> {
        let topology = TopologyStore::open_in_memory()?;
        let index = SemanticIndex::open_in_memory(&config)?;
        Ok(TemporalGraph { topology, index, embedder, config })
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
        let id = gen_id("ep");
        let episode = Episode {
            id: id.clone(),
            source: source.to_string(),
            body: body.to_string(),
            occurred_at,
            created_at: now_ms(),
            properties,
        };
        self.topology.upsert_episode(&episode)?;
        if !mentions.is_empty() {
            self.topology.link_mentions(&id, mentions)?;
        }
        Ok(id)
    }

    pub fn get_episode(&self, id: &str) -> Result<Option<Episode>> {
        self.topology.get_episode(id)
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
        if id.trim().is_empty() {
            return Err(GraphError::InvalidArgument("entity id is required".into()));
        }
        let existing = self.topology.get_entity(id)?;
        let created_at = existing.map(|e| e.created_at).unwrap_or_else(now_ms);
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
        self.topology.upsert_entity(&entity)?;
        self.index.index_entity(&entity, self.embedder.as_deref(), None)?;
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
        if id.trim().is_empty() {
            return Err(GraphError::InvalidArgument("entity id is required".into()));
        }
        let created_at = self.topology.get_entity(id)?.map(|e| e.created_at).unwrap_or_else(now_ms);
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
        self.topology.upsert_entity(&entity)?;
        self.index.index_entity(&entity, None, Some(embedding))?;
        Ok(entity.id)
    }

    pub fn get_entity(&self, id: &str) -> Result<Option<Entity>> {
        self.topology.get_entity(id)
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
        self.add_fact_inner(src, relation, dst, fact_text, properties, episodes, valid_at, invalidates, None)
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
            src, relation, dst, fact_text, properties, episodes, valid_at, invalidates,
            Some(fact_embedding),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn add_fact_inner(
        &mut self,
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
        let now = now_ms();
        // Close superseded facts first, then re-index them so as-of/current search
        // reflects their new closed interval.
        for prior_id in invalidates {
            let (inv, exp) = temporal::supersede_timestamps(valid_at, now);
            let existed = self.topology.close_fact(prior_id, inv, exp)?;
            if existed {
                if let Some(closed) = self.topology.get_fact(prior_id)? {
                    self.index.index_fact(&closed, self.embedder.as_deref(), None)?;
                }
            }
        }
        let reference_time = self.reference_time_for(&episodes)?;
        let fact = Fact {
            id: gen_id("f"),
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
        self.topology.upsert_fact(&fact)?;
        // Vector-in indexes by the supplied embedding; text-in uses the engine embedder.
        let embedder = if fact_embedding.is_some() { None } else { self.embedder.as_deref() };
        self.index.index_fact(&fact, embedder, fact_embedding)?;
        Ok(fact.id)
    }

    /// Close a fact's validity without a replacement (an explicit end / retraction).
    pub fn invalidate_fact(&mut self, id: &str, invalid_at: Option<TimeMs>) -> Result<bool> {
        let existed = self.topology.close_fact(id, invalid_at, now_ms())?;
        if existed {
            if let Some(closed) = self.topology.get_fact(id)? {
                self.index.index_fact(&closed, self.embedder.as_deref(), None)?;
            }
        }
        Ok(existed)
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
        self.topology.facts_for(&[id.to_string()], direction, relation, as_of)
    }

    /// Deterministic k-hop neighbourhood (BFS) around `seeds`, in a temporal frame.
    pub fn k_hop(
        &self,
        seeds: &[String],
        k: usize,
        direction: Direction,
        as_of: AsOf,
    ) -> Result<Subgraph> {
        use std::collections::{BTreeMap, BTreeSet};
        let mut visited: BTreeSet<String> = seeds.iter().cloned().collect();
        let mut frontier: Vec<String> = seeds.to_vec();
        let mut facts_by_id: BTreeMap<String, Fact> = BTreeMap::new();
        for _hop in 0..k {
            if frontier.is_empty() {
                break;
            }
            let facts = self.topology.facts_for(&frontier, direction, None, as_of)?;
            let mut next: Vec<String> = Vec::new();
            for f in facts {
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
        let hits = self.index.semantic_entities(
            query,
            embedding,
            self.embedder.as_deref(),
            k,
            entity_type,
            as_of,
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
        let hits = self.index.semantic_facts(
            query,
            embedding,
            self.embedder.as_deref(),
            k,
            relation,
            as_of,
        )?;
        let mut out = Vec::new();
        for h in hits {
            if let Some(f) = self.topology.get_fact(&h.id)? {
                out.push((h.score, f));
            }
        }
        Ok(out)
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
        let seeds = self.semantic_entities(query, embedding, k, entity_type, as_of)?;
        let seed_ids: Vec<String> = seeds.iter().map(|(_s, e)| e.id.clone()).collect();
        let mut subgraph = self.k_hop(&seed_ids, hops, direction, as_of)?;
        for (score, entity) in &seeds {
            subgraph.entities.entry(entity.id.clone()).or_insert_with(|| entity.clone());
            subgraph.seeds.push((entity.id.clone(), *score));
        }
        Ok(subgraph)
    }

    /// Candidate entities matching `name` for the caller's resolution step
    /// (embedding + lexical); the caller decides the merge.
    pub fn resolve_entity(&self, name: &str, k: usize) -> Result<Vec<(f32, Entity)>> {
        let hits = self.index.resolve_entity(name, self.embedder.as_deref(), k)?;
        let mut out = Vec::new();
        for h in hits {
            if let Some(e) = self.topology.get_entity(&h.id)? {
                out.push((h.score, e));
            }
        }
        Ok(out)
    }

    /// Every fact ever touching an entity, all frames (history).
    pub fn history(&self, entity_id: &str) -> Result<Vec<Fact>> {
        self.topology.history(entity_id)
    }

    // -- maintenance --------------------------------------------------------

    /// Remove an entity and its incident facts from both stores.
    pub fn remove_entity(&mut self, id: &str) -> Result<bool> {
        let (existed, removed_fact_ids) = self.topology.remove_entity(id)?;
        self.index.remove_entity(id)?;
        for fid in removed_fact_ids {
            self.index.remove_fact(&fid)?;
        }
        Ok(existed)
    }

    /// Hard-remove a single fact from both stores (Graphiti edge deletion). Prefer
    /// [`TemporalGraph::invalidate_fact`], which preserves history; this deletes the
    /// row outright. Returns whether the fact existed.
    pub fn remove_fact(&mut self, id: &str) -> Result<bool> {
        let existed = self.topology.remove_fact(id)?;
        self.index.remove_fact(id)?;
        Ok(existed)
    }

    /// Rebuild the semantic index from the topology truth store: drop orphans, then
    /// re-index every entity and fact. Makes the index a throwaway artifact.
    pub fn reindex(&mut self) -> Result<ReindexStats> {
        let entities = self.topology.iter_entities()?;
        let facts = self.topology.iter_facts()?;
        let live_entity_ids: Vec<String> = entities.iter().map(|e| e.id.clone()).collect();
        let live_fact_ids: Vec<String> = facts.iter().map(|f| f.id.clone()).collect();
        let removed_orphans = self.index.drop_orphans(&live_entity_ids, &live_fact_ids)?;
        let embedder = self.embedder.as_deref();
        for e in &entities {
            self.index.index_entity(e, embedder, None)?;
        }
        for f in &facts {
            self.index.index_fact(f, embedder, None)?;
        }
        Ok(ReindexStats {
            reindexed_entities: entities.len(),
            reindexed_facts: facts.len(),
            removed_orphans,
        })
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
        self.index.persist()
    }
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
