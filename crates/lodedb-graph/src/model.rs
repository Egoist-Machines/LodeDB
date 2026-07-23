//! The bi-temporal graph data model.
//!
//! A faithful, storage-oriented port of Graphiti's `graphiti_core` model
//! (`nodes.py`, `edges.py`):
//!
//! - [`Episode`]  ← `EpisodicNode`   (raw observation + provenance)
//! - [`Entity`]   ← `EntityNode`      (a resolved thing in the world)
//! - [`Fact`]     ← `EntityEdge`      (a typed, labeled, bi-temporal relationship)
//!
//! The one deliberate divergence from Graphiti: this crate is **not** an LLM
//! pipeline. It stores what it is given and answers temporal queries; entity
//! extraction, entity resolution, temporal extraction, and contradiction
//! detection stay with the caller (an LLM layer, exactly as Graphiti's own
//! `add_triplet` is the LLM-free insertion path). See `docs/temporal-graph.md`.
//!
//! Timestamps are epoch **milliseconds** (`i64`); `None` encodes an open interval,
//! matching Graphiti's nullable `datetime` fields.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{GraphError, Result};

/// Epoch-milliseconds timestamp. Event time and transaction time share the type.
pub type TimeMs = i64;

/// A raw observation the graph was built from — Graphiti `EpisodicNode`.
///
/// `source` mirrors Graphiti's `EpisodeType` (`message` / `text` / `json`) but is a
/// free string so callers may add their own kinds. `occurred_at` is Graphiti's
/// `reference_time`, the event-time anchor a fact extracted from this episode
/// inherits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Episode {
    pub id: String,
    pub source: String,
    pub body: String,
    /// Event time of the observation (Graphiti `reference_time`).
    pub occurred_at: TimeMs,
    /// Transaction time the episode was recorded.
    pub created_at: TimeMs,
    #[serde(default)]
    pub properties: Value,
}

/// A resolved thing in the world — Graphiti `EntityNode`.
///
/// `entity_type` is a caller-defined vocabulary string (the engine ships none).
/// `label` is the text embedded for semantic entry-point search (Graphiti's
/// `name`/`summary`). Entities that begin and end (a project, an event, a job)
/// may carry an event-time validity interval; all carry transaction time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    pub id: String,
    pub entity_type: String,
    pub label: String,
    #[serde(default)]
    pub properties: Value,
    #[serde(default)]
    pub valid_at: Option<TimeMs>,
    #[serde(default)]
    pub invalid_at: Option<TimeMs>,
    pub created_at: TimeMs,
    #[serde(default)]
    pub expired_at: Option<TimeMs>,
}

/// A typed, directed, labeled, bi-temporal relationship — Graphiti `EntityEdge`.
///
/// Field-for-field with `EntityEdge`: `relation` is Graphiti's edge `name`, `fact`
/// is the natural-language fact text (indexed for semantic fact search), `episodes`
/// is the provenance list, `properties` is Graphiti's `attributes`, and the four
/// timestamps plus `reference_time` carry the bi-temporal state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fact {
    pub id: String,
    pub src: String,
    /// The relation name — Graphiti `EntityEdge.name`.
    pub relation: String,
    pub dst: String,
    /// Natural-language fact text — embedded for semantic fact search.
    pub fact: String,
    #[serde(default)]
    pub properties: Value,
    /// Provenance: ids of the episodes that asserted this fact.
    #[serde(default)]
    pub episodes: Vec<String>,
    /// Event time — when the fact became true in the world.
    #[serde(default)]
    pub valid_at: Option<TimeMs>,
    /// Event time — when the fact stopped being true.
    #[serde(default)]
    pub invalid_at: Option<TimeMs>,
    /// Transaction time — when the system recorded the fact.
    pub created_at: TimeMs,
    /// Transaction time — when the system superseded / retracted the fact.
    #[serde(default)]
    pub expired_at: Option<TimeMs>,
    /// Event-time anchor of the producing episode — Graphiti `reference_time`.
    #[serde(default)]
    pub reference_time: Option<TimeMs>,
}

impl Fact {
    /// Whether the fact is currently live: not superseded on the transaction axis
    /// and not ended on the event axis. This is Graphiti's "current" view
    /// (`expired_at IS NULL AND invalid_at IS NULL`).
    pub fn is_live(&self) -> bool {
        self.expired_at.is_none() && self.invalid_at.is_none()
    }

    /// Whether the fact satisfies `as_of` per the authoritative row — the
    /// in-memory twin of the SQL frame (see `temporal::frame_matches`).
    pub fn matches(&self, as_of: AsOf) -> bool {
        crate::temporal::frame_matches(as_of, self.valid_at, self.invalid_at, self.expired_at)
    }
}

impl Entity {
    /// Whether the entity satisfies `as_of` per the authoritative row — the
    /// in-memory twin of the SQL frame (see `temporal::frame_matches`).
    pub fn matches(&self, as_of: AsOf) -> bool {
        crate::temporal::frame_matches(as_of, self.valid_at, self.invalid_at, self.expired_at)
    }
}

/// Direction of traversal over facts, relative to a set of seed nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Facts whose `src` is in the seed set.
    Out,
    /// Facts whose `dst` is in the seed set.
    In,
    /// Either endpoint in the seed set.
    Both,
}

impl Direction {
    /// Parse the wire form used by the bindings.
    pub fn parse(s: &str) -> Result<Direction> {
        match s {
            "out" => Ok(Direction::Out),
            "in" => Ok(Direction::In),
            "both" => Ok(Direction::Both),
            other => Err(GraphError::InvalidArgument(format!(
                "direction must be out|in|both, got {other:?}"
            ))),
        }
    }
}

/// The temporal frame a read resolves under.
///
/// - [`AsOf::Now`] — the current view (live facts only): Graphiti's default.
/// - [`AsOf::At`] — as-of an event-time instant `T`: facts valid at `T`.
/// - [`AsOf::All`] — every version, no temporal filter (history).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsOf {
    Now,
    At(TimeMs),
    All,
}

impl Default for AsOf {
    fn default() -> Self {
        AsOf::Now
    }
}

/// Which side of a prefix-asymmetric embedder to request (mirrors LodeDB's
/// document/query roles; BGE-style models prefix queries).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedRole {
    Document,
    Query,
}

/// A caller-supplied embedding model.
///
/// `lodedb-core` does not embed — embedding lives in the binding layer — so, like
/// Swift's `LodeEmbedder`, the graph is driven with a caller-provided embedder.
/// For vector-in callers who bring their own vectors this may be a no-op and the
/// `_vec` verbs used instead.
pub trait Embedder: Send + Sync {
    /// The embedding dimension; must match the index dimension.
    fn dimension(&self) -> usize;
    /// Embed a batch of texts for the given role.
    fn embed(&self, texts: &[String], role: EmbedRole) -> Result<Vec<Vec<f32>>>;
}

/// Open-time configuration for the semantic index half.
#[derive(Debug, Clone)]
pub struct GraphConfig {
    /// Embedding dimension of the semantic index.
    pub vector_dim: usize,
    /// Retain fact/label text for lexical (BM25) hybrid search. On by default.
    pub index_text: bool,
    /// Index edge (fact) text for `semantic_facts`, not just entities. On by default.
    pub index_facts: bool,
}

impl Default for GraphConfig {
    fn default() -> Self {
        GraphConfig {
            vector_dim: 384,
            index_text: true,
            index_facts: true,
        }
    }
}

/// A retrieved neighbourhood: entities by id, the facts among them, and the
/// semantic seed entities (with scores) the expansion started from. Port of
/// Graphiti's search result shape, minus communities.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Subgraph {
    pub entities: std::collections::BTreeMap<String, Entity>,
    pub facts: Vec<Fact>,
    pub seeds: Vec<(String, f32)>,
}

impl Subgraph {
    /// Number of entities in the subgraph.
    pub fn len(&self) -> usize {
        self.entities.len()
    }
    /// Whether the subgraph has no entities.
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }
}
