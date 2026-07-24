//! C ABI for the bi-temporal knowledge graph (`lodedb-graph`), so the Swift
//! `LodeGraph` runs it on device.
//!
//! Uniform JSON convention: every verb takes one JSON request `LodeStringView` and
//! writes one JSON response `LodeOwnedString`, reusing the crate's shared helpers
//! (`ffi_result` / `read_json_view` / `write_owned_json`) and status/error contract.
//! The graph is opened **without** an embedder — embedding lives in the Swift layer,
//! which embeds label/fact/query text on device and passes vectors to the vector-in
//! verbs, so the core never calls back across the ABI.

use lodedb_core::{CoreError, CoreErrorCode};
use lodedb_graph::{now_valid_frame, AsOf, Direction, GraphConfig, GraphError, TemporalGraph};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;

use crate::{
    ffi_result, read_json_view, require_out, write_owned_json, LodeError, LodeOwnedString,
    LodeStringView,
};

/// Opaque handle wrapping a native `TemporalGraph`.
///
/// Not thread-safe: the verbs synthesize a `&mut TemporalGraph` from `*mut LodeGraph`,
/// so a handle must be used by one thread at a time. Callers serialize access (the Swift
/// `LodeGraph` guards every call with an `NSLock`).
#[repr(C)]
pub struct LodeGraph {
    graph: TemporalGraph,
}

fn graph_err(error: GraphError) -> CoreError {
    let code = match error {
        GraphError::InvalidArgument(_) | GraphError::Embedding(_) => CoreErrorCode::InvalidArgument,
        GraphError::NotFound(_) => CoreErrorCode::NotFound,
        _ => CoreErrorCode::Internal,
    };
    CoreError::new(code, error.to_string())
}

fn graph_mut<'a>(graph: *mut LodeGraph) -> Result<&'a mut TemporalGraph, CoreError> {
    if graph.is_null() {
        return Err(CoreError::new(
            CoreErrorCode::InvalidArgument,
            "graph pointer is null",
        ));
    }
    Ok(unsafe { &mut (*graph).graph })
}

fn graph_ref<'a>(graph: *const LodeGraph) -> Result<&'a TemporalGraph, CoreError> {
    if graph.is_null() {
        return Err(CoreError::new(
            CoreErrorCode::InvalidArgument,
            "graph pointer is null",
        ));
    }
    Ok(unsafe { &(*graph).graph })
}

fn resolve_as_of(
    as_of_ms: Option<i64>,
    all_time: Option<bool>,
    strict_now: Option<bool>,
    known_at_ms: Option<i64>,
) -> Result<AsOf, CoreError> {
    if all_time.unwrap_or(false) {
        Ok(AsOf::All)
    } else if strict_now.unwrap_or(false) {
        if as_of_ms.is_some() || known_at_ms.is_some() {
            return Err(CoreError::new(
                CoreErrorCode::InvalidArgument,
                "strict_now cannot be combined with as_of_ms or known_at_ms",
            ));
        }
        Ok(now_valid_frame())
    } else if let Some(known_at) = known_at_ms {
        let valid_at = as_of_ms.ok_or_else(|| {
            CoreError::new(
                CoreErrorCode::InvalidArgument,
                "known_at_ms requires an event-time as_of_ms",
            )
        })?;
        Ok(AsOf::AtKnown { valid_at, known_at })
    } else if let Some(t) = as_of_ms {
        Ok(AsOf::At(t))
    } else {
        Ok(AsOf::Now)
    }
}

// -- request payloads ------------------------------------------------------

#[derive(Deserialize)]
struct OpenReq {
    path: Option<String>,
    vector_dim: usize,
    index_facts: Option<bool>,
    /// Retain label/fact text in the semantic index for lexical (BM25) hybrid
    /// search. `false` keeps the index vector-only (no text on the index side).
    index_text: Option<bool>,
}

#[derive(Deserialize)]
struct AddEpisodeReq {
    id: Option<String>,
    source: String,
    body: String,
    occurred_at: i64,
    properties: Option<Value>,
    mentions: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct UpsertEntityVecReq {
    id: String,
    #[serde(rename = "type")]
    entity_type: String,
    label: String,
    embedding: Vec<f32>,
    properties: Option<Value>,
    valid_at: Option<i64>,
    invalid_at: Option<i64>,
    property_sources: Option<BTreeMap<String, String>>,
}

#[derive(Deserialize)]
struct AddFactVecReq {
    id: Option<String>,
    src: String,
    relation: String,
    dst: String,
    fact: String,
    embedding: Vec<f32>,
    properties: Option<Value>,
    episodes: Option<Vec<String>>,
    valid_at: Option<i64>,
    invalidates: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct InvalidateFactReq {
    id: String,
    invalid_at: Option<i64>,
}

#[derive(Deserialize)]
struct IdReq {
    id: String,
}

#[derive(Deserialize)]
struct EntitiesReq {
    #[serde(rename = "type")]
    entity_type: Option<String>,
    as_of_ms: Option<i64>,
    all_time: Option<bool>,
    strict_now: Option<bool>,
    known_at_ms: Option<i64>,
}

#[derive(Deserialize)]
struct NeighborsReq {
    id: String,
    direction: String,
    relation: Option<String>,
    as_of_ms: Option<i64>,
    all_time: Option<bool>,
    strict_now: Option<bool>,
    known_at_ms: Option<i64>,
}

#[derive(Deserialize)]
struct KHopReq {
    seeds: Vec<String>,
    k: usize,
    direction: String,
    as_of_ms: Option<i64>,
    all_time: Option<bool>,
    strict_now: Option<bool>,
    known_at_ms: Option<i64>,
    predicate: Option<Value>,
}

#[derive(Deserialize)]
struct SemanticEntitiesReq {
    embedding: Vec<f32>,
    k: usize,
    #[serde(rename = "type")]
    entity_type: Option<String>,
    as_of_ms: Option<i64>,
    all_time: Option<bool>,
    strict_now: Option<bool>,
    known_at_ms: Option<i64>,
    predicate: Option<Value>,
}

#[derive(Deserialize)]
struct SemanticFactsReq {
    embedding: Vec<f32>,
    k: usize,
    relation: Option<String>,
    as_of_ms: Option<i64>,
    all_time: Option<bool>,
    strict_now: Option<bool>,
    known_at_ms: Option<i64>,
    predicate: Option<Value>,
}

#[derive(Deserialize)]
struct SearchSubgraphReq {
    embedding: Vec<f32>,
    k: usize,
    hops: usize,
    direction: String,
    #[serde(rename = "type")]
    entity_type: Option<String>,
    relation: Option<String>,
    as_of_ms: Option<i64>,
    all_time: Option<bool>,
    strict_now: Option<bool>,
    known_at_ms: Option<i64>,
    predicate: Option<Value>,
    seed_kind: Option<String>,
}

#[derive(Deserialize)]
struct PropertyHistoryReq {
    id: String,
    key: Option<String>,
}

// -- lifecycle -------------------------------------------------------------

/// Open (or create) a graph from a JSON `{path?, vector_dim, index_facts?}` request.
/// `path` absent opens an in-memory graph. No embedder — use the vector-in verbs.
///
/// # Safety
/// `request` must hold valid UTF-8 JSON; `out` and `error` must be valid pointers.
#[no_mangle]
pub unsafe extern "C" fn lodedb_graph_open_json(
    request: LodeStringView,
    out: *mut *mut LodeGraph,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        require_out(out)?;
        let req = read_json_view::<OpenReq>(request)?;
        let config = GraphConfig {
            vector_dim: req.vector_dim,
            index_text: req.index_text.unwrap_or(true),
            index_facts: req.index_facts.unwrap_or(true),
        };
        let graph = match req.path {
            Some(ref p) => TemporalGraph::open(std::path::Path::new(p), config, None),
            None => TemporalGraph::open_in_memory(config, None),
        }
        .map_err(graph_err)?;
        unsafe {
            *out = Box::into_raw(Box::new(LodeGraph { graph }));
        }
        Ok(())
    })
}

/// Frees a graph handle.
///
/// # Safety
/// `graph` must be null or a pointer returned by `lodedb_graph_open_json`, freed once.
#[no_mangle]
pub unsafe extern "C" fn lodedb_graph_free(graph: *mut LodeGraph) {
    if !graph.is_null() {
        let _ = Box::from_raw(graph);
    }
}

// -- macro for the uniform (graph, request) -> json verbs ------------------

macro_rules! graph_verb {
    ($name:ident, $graph:ident, $req:ident : $req_ty:ty, $body:expr) => {
        /// # Safety
        /// `request`, `out`, and `error` must be valid; `request` is UTF-8 JSON. The
        /// `graph` handle must be **exclusively owned** for the duration of the call: no
        /// other thread may invoke any `lodedb_graph_*` verb on the same handle
        /// concurrently. Each verb synthesizes a `&mut TemporalGraph` from the handle, so
        /// concurrent calls would alias `&mut` (a data race / UB). Serialize access
        /// externally (the Swift `LodeGraph` does this with an `NSLock`).
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            graph: *mut LodeGraph,
            request: LodeStringView,
            out: *mut *mut LodeOwnedString,
            error: *mut *mut LodeError,
        ) -> u32 {
            ffi_result(error, || {
                require_out(out)?;
                let $graph = graph_mut(graph)?;
                let $req = read_json_view::<$req_ty>(request)?;
                let value = $body;
                write_owned_json(out, &value)
            })
        }
    };
}

// -- writes ----------------------------------------------------------------

graph_verb!(lodedb_graph_add_episode_json, g, req: AddEpisodeReq, {
    g.add_episode_with_id(
        req.id.as_deref(),
        &req.source,
        &req.body,
        req.occurred_at,
        req.properties.unwrap_or(Value::Null),
        &req.mentions.unwrap_or_default(),
    )
    .map_err(graph_err)?
});

graph_verb!(lodedb_graph_upsert_entity_vec_json, g, req: UpsertEntityVecReq, {
    g.upsert_entity_vec_with_sources(
        &req.id,
        &req.entity_type,
        &req.label,
        req.properties.unwrap_or(Value::Null),
        &req.embedding,
        req.valid_at,
        req.invalid_at,
        &req.property_sources.unwrap_or_default(),
    )
    .map_err(graph_err)?
});

graph_verb!(lodedb_graph_add_fact_vec_json, g, req: AddFactVecReq, {
    g.add_fact_vec_with_id(
        req.id.as_deref(),
        &req.src,
        &req.relation,
        &req.dst,
        &req.fact,
        req.properties.unwrap_or(Value::Null),
        req.episodes.unwrap_or_default(),
        req.valid_at,
        &req.invalidates.unwrap_or_default(),
        &req.embedding,
    )
    .map_err(graph_err)?
});

graph_verb!(lodedb_graph_invalidate_fact_json, g, req: InvalidateFactReq, {
    g.invalidate_fact(&req.id, req.invalid_at).map_err(graph_err)?
});

graph_verb!(lodedb_graph_remove_entity_json, g, req: IdReq, {
    g.remove_entity(&req.id).map_err(graph_err)?
});

graph_verb!(lodedb_graph_remove_fact_json, g, req: IdReq, {
    g.remove_fact(&req.id).map_err(graph_err)?
});

graph_verb!(lodedb_graph_remove_episode_json, g, req: IdReq, {
    g.remove_episode(&req.id).map_err(graph_err)?
});

// -- reads -----------------------------------------------------------------

graph_verb!(lodedb_graph_get_entity_json, g, req: IdReq, {
    g.get_entity(&req.id).map_err(graph_err)?
});

graph_verb!(lodedb_graph_get_fact_json, g, req: IdReq, {
    g.get_fact(&req.id).map_err(graph_err)?
});

graph_verb!(lodedb_graph_get_episode_json, g, req: IdReq, {
    g.get_episode(&req.id).map_err(graph_err)?
});

graph_verb!(lodedb_graph_episodes_json, g, _req: Value, {
    g.episodes().map_err(graph_err)?
});

graph_verb!(lodedb_graph_facts_by_episode_json, g, req: IdReq, {
    g.facts_by_episode(&req.id).map_err(graph_err)?
});

graph_verb!(lodedb_graph_entity_property_history_json, g, req: PropertyHistoryReq, {
    g.entity_property_history(&req.id, req.key.as_deref())
        .map_err(graph_err)?
});

graph_verb!(lodedb_graph_entities_json, g, req: EntitiesReq, {
    g.entities(
        req.entity_type.as_deref(),
        resolve_as_of(
            req.as_of_ms,
            req.all_time,
            req.strict_now,
            req.known_at_ms,
        )?,
    )
        .map_err(graph_err)?
});

graph_verb!(lodedb_graph_history_json, g, req: IdReq, {
    g.history(&req.id).map_err(graph_err)?
});

graph_verb!(lodedb_graph_neighbors_json, g, req: NeighborsReq, {
    let dir = Direction::parse(&req.direction).map_err(graph_err)?;
    g.neighbors(
        &req.id,
        dir,
        req.relation.as_deref(),
        resolve_as_of(
            req.as_of_ms,
            req.all_time,
            req.strict_now,
            req.known_at_ms,
        )?,
    )
        .map_err(graph_err)?
});

graph_verb!(lodedb_graph_k_hop_json, g, req: KHopReq, {
    let dir = Direction::parse(&req.direction).map_err(graph_err)?;
    g.k_hop_filtered(
        &req.seeds,
        req.k,
        dir,
        resolve_as_of(
            req.as_of_ms,
            req.all_time,
            req.strict_now,
            req.known_at_ms,
        )?,
        req.predicate.as_ref(),
    )
        .map_err(graph_err)?
});

graph_verb!(lodedb_graph_semantic_entities_json, g, req: SemanticEntitiesReq, {
    g.semantic_entities_filtered(
        None,
        Some(req.embedding.as_slice()),
        req.k,
        req.entity_type.as_deref(),
        resolve_as_of(
            req.as_of_ms,
            req.all_time,
            req.strict_now,
            req.known_at_ms,
        )?,
        req.predicate.as_ref(),
    )
    .map_err(graph_err)?
});

graph_verb!(lodedb_graph_semantic_facts_json, g, req: SemanticFactsReq, {
    g.semantic_facts_filtered(
        None,
        Some(req.embedding.as_slice()),
        req.k,
        req.relation.as_deref(),
        resolve_as_of(
            req.as_of_ms,
            req.all_time,
            req.strict_now,
            req.known_at_ms,
        )?,
        req.predicate.as_ref(),
    )
    .map_err(graph_err)?
});

graph_verb!(lodedb_graph_search_subgraph_json, g, req: SearchSubgraphReq, {
    let dir = Direction::parse(&req.direction).map_err(graph_err)?;
    g.search_subgraph_filtered(
        None,
        Some(req.embedding.as_slice()),
        req.k,
        req.hops,
        dir,
        req.entity_type.as_deref(),
        req.relation.as_deref(),
        resolve_as_of(
            req.as_of_ms,
            req.all_time,
            req.strict_now,
            req.known_at_ms,
        )?,
        req.predicate.as_ref(),
        req.seed_kind.as_deref().unwrap_or("entity"),
    )
    .map_err(graph_err)?
});

// -- maintenance (no request payload) --------------------------------------

/// Rebuilds the semantic index; returns `{reindexed_entities, reindexed_facts,
/// removed_orphans}`.
///
/// # Safety
/// `out` and `error` must be valid pointers.
#[no_mangle]
pub unsafe extern "C" fn lodedb_graph_reindex_json(
    graph: *mut LodeGraph,
    out: *mut *mut LodeOwnedString,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        require_out(out)?;
        let g = graph_mut(graph)?;
        let s = g.reindex().map_err(graph_err)?;
        let value = json!({
            "reindexed_entities": s.reindexed_entities,
            "reindexed_facts": s.reindexed_facts,
            "removed_orphans": s.removed_orphans,
        });
        write_owned_json(out, &value)
    })
}

/// Returns `{entities, facts, indexed_documents}` counts.
///
/// # Safety
/// `out` and `error` must be valid pointers.
#[no_mangle]
pub unsafe extern "C" fn lodedb_graph_stats_json(
    graph: *const LodeGraph,
    out: *mut *mut LodeOwnedString,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        require_out(out)?;
        let g = graph_ref(graph)?;
        let s = g.stats().map_err(graph_err)?;
        let value = json!({
            "entities": s.entities,
            "facts": s.facts,
            "indexed_documents": s.indexed_documents,
        });
        write_owned_json(out, &value)
    })
}

/// Checkpoints the semantic index to disk.
///
/// # Safety
/// `error` must be valid.
#[no_mangle]
pub unsafe extern "C" fn lodedb_graph_persist(
    graph: *mut LodeGraph,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        let g = graph_mut(graph)?;
        g.persist().map_err(graph_err)
    })
}
