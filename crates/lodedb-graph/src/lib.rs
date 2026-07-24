//! # lodedb-graph
//!
//! A bi-temporal knowledge graph over [`lodedb_core`] — a storage-and-query port of
//! the temporal core of [Graphiti](https://github.com/getzep/graphiti), for local,
//! embedded, on-device agent memory.
//!
//! Two stores behind one [`TemporalGraph`] handle:
//!
//! - a **topology truth store** (embedded SQLite): episodes, entities, typed facts,
//!   and their bi-temporal validity — the authoritative adjacency;
//! - a **semantic index** (`lodedb-core`): a rebuildable vector + BM25 index over
//!   entity labels and fact text, for hybrid entry-point search.
//!
//! Every fact carries four timestamps, exactly as Graphiti's `EntityEdge`:
//! `valid_at`/`invalid_at` (event time) and `created_at`/`expired_at` (transaction
//! time). Contradictions **invalidate** rather than delete, so history is preserved
//! and "as-of T" queries fall out of the model.
//!
//! ## Scope — not an LLM pipeline
//!
//! Unlike Graphiti's `add_episode`, this crate performs **no** LLM extraction,
//! entity resolution, temporal extraction, or contradiction detection. It stores
//! what it is given ([`TemporalGraph::add_fact`] is the analogue of Graphiti's
//! LLM-free `add_triplet`) and answers temporal queries; the LLM layer is the
//! caller's. It needs only a caller-supplied [`Embedder`] (or precomputed vectors),
//! mirroring how `lodedb-core` keeps embedding in the binding layer.
//!
//! See `docs/temporal-graph.md` for the full design.

mod error;
mod graph;
mod index;
mod model;
mod search;
mod temporal;
mod topology;

pub use error::{GraphError, Result};
pub use graph::{now_valid_frame, GraphStats, ReindexStats, TemporalGraph};
pub use index::IndexHit;
pub use model::{
    AsOf, Direction, EmbedRole, Embedder, Entity, EntityPropertyVersion, Episode, Fact,
    GraphConfig, Subgraph, TimeMs,
};

// The reranker primitives (ported from Graphiti's search_utils) are public so the
// bindings and advanced callers can compose custom search pipelines.
pub mod rerank {
    pub use crate::search::{
        episode_mentions_reranker, maximal_marginal_relevance, node_distance_reranker, rrf,
    };
}
