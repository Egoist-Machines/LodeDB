//! Python binding for `lodedb-graph`, exposed as the `graph` submodule of the
//! bundled `_turbovec` extension (surfaced to users as
//! `lodedb.graph.TemporalKnowledgeGraph`, via the pure-Python wrapper in
//! `src/lodedb/graph/temporal.py`).
//!
//! Mirrors `cloud.rs`'s "pure translator over a native Rust crate" pattern: it
//! marshals Python values to/from `lodedb_graph` and does no graph logic of its own.
//! Embedding stays in Python — a caller-supplied object with `dimension` and
//! `embed(texts, role)` is wrapped as a [`PyEmbedder`] the Rust engine calls back
//! into, matching how `lodedb-core` keeps embedding in the binding layer.

use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use lodedb_graph::{
    now_valid_frame, AsOf, Direction, EmbedRole, Embedder, Entity, EntityPropertyVersion,
    Episode, Fact, GraphConfig, GraphError, TemporalGraph,
};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};

/// Map a `GraphError` onto the closest Python exception.
fn to_py_err(error: GraphError) -> PyErr {
    match error {
        GraphError::InvalidArgument(m) => PyValueError::new_err(m),
        GraphError::NotFound(m) => PyKeyError::new_err(m),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

/// Resolve the `(as_of_ms, all_time)` wire pair into a temporal frame: `all_time`
/// wins (history); else `Some(t)` is as-of `t`; else the current view.
fn resolve_as_of(
    as_of_ms: Option<i64>,
    all_time: bool,
    strict_now: bool,
    known_at_ms: Option<i64>,
) -> PyResult<AsOf> {
    if all_time {
        Ok(AsOf::All)
    } else if strict_now {
        if as_of_ms.is_some() || known_at_ms.is_some() {
            return Err(PyValueError::new_err(
                "strict_now cannot be combined with as_of_ms or known_at_ms",
            ));
        }
        Ok(now_valid_frame())
    } else if let Some(known_at) = known_at_ms {
        let valid_at = as_of_ms.ok_or_else(|| {
            PyValueError::new_err("known_at_ms requires an event-time as_of_ms")
        })?;
        Ok(AsOf::AtKnown { valid_at, known_at })
    } else if let Some(t) = as_of_ms {
        Ok(AsOf::At(t))
    } else {
        Ok(AsOf::Now)
    }
}

/// Parse an optional JSON string into a `serde_json::Value` (properties travel as
/// JSON text across the boundary; the Python wrapper `json.dumps`/`loads` them).
fn parse_props(json: Option<&str>) -> PyResult<Value> {
    match json {
        None => Ok(Value::Null),
        Some(s) if s.trim().is_empty() => Ok(Value::Null),
        Some(s) => serde_json::from_str(s)
            .map_err(|e| PyValueError::new_err(format!("properties must be JSON: {e}"))),
    }
}

fn parse_optional_json(json: Option<&str>, what: &str) -> PyResult<Option<Value>> {
    match json {
        None => Ok(None),
        Some(s) if s.trim().is_empty() => Ok(None),
        Some(s) => serde_json::from_str(s)
            .map(Some)
            .map_err(|e| PyValueError::new_err(format!("{what} must be JSON: {e}"))),
    }
}

fn parse_property_sources(json: Option<&str>) -> PyResult<BTreeMap<String, String>> {
    match parse_optional_json(json, "property_sources")? {
        None => Ok(BTreeMap::new()),
        Some(value) => serde_json::from_value(value).map_err(|e| {
            PyValueError::new_err(format!(
                "property_sources must map property names to episode ids: {e}"
            ))
        }),
    }
}

fn props_to_string(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "null".to_string())
}

/// A caller-supplied Python embedder, wrapped as a Rust [`Embedder`]. Holds the
/// Python object and reattaches to the interpreter to call its
/// `embed(texts, role)`; the graph verbs run detached, so this is the one place
/// a native graph call re-enters Python.
struct PyEmbedder {
    obj: Py<PyAny>,
    dim: usize,
}

impl Embedder for PyEmbedder {
    fn dimension(&self) -> usize {
        self.dim
    }

    fn embed(&self, texts: &[String], role: EmbedRole) -> lodedb_graph::Result<Vec<Vec<f32>>> {
        let role_str = match role {
            EmbedRole::Document => "document",
            EmbedRole::Query => "query",
        };
        Python::attach(|py| {
            let bound = self.obj.bind(py);
            let result = bound
                .call_method1("embed", (texts.to_vec(), role_str))
                .map_err(|e| GraphError::Embedding(format!("python embedder failed: {e}")))?;
            let vectors: Vec<Vec<f32>> = result.extract().map_err(|e| {
                GraphError::Embedding(format!("embedder must return list[list[float]]: {e}"))
            })?;
            Ok(vectors)
        })
    }
}

fn entity_dict<'py>(py: Python<'py>, entity: &Entity) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("id", &entity.id)?;
    dict.set_item("type", &entity.entity_type)?;
    dict.set_item("label", &entity.label)?;
    dict.set_item("properties", props_to_string(&entity.properties))?;
    dict.set_item("valid_at", entity.valid_at)?;
    dict.set_item("invalid_at", entity.invalid_at)?;
    dict.set_item("created_at", entity.created_at)?;
    dict.set_item("expired_at", entity.expired_at)?;
    Ok(dict)
}

fn fact_dict<'py>(py: Python<'py>, fact: &Fact) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("id", &fact.id)?;
    dict.set_item("src", &fact.src)?;
    dict.set_item("relation", &fact.relation)?;
    dict.set_item("dst", &fact.dst)?;
    dict.set_item("fact", &fact.fact)?;
    dict.set_item("properties", props_to_string(&fact.properties))?;
    dict.set_item("episodes", fact.episodes.clone())?;
    dict.set_item("valid_at", fact.valid_at)?;
    dict.set_item("invalid_at", fact.invalid_at)?;
    dict.set_item("created_at", fact.created_at)?;
    dict.set_item("expired_at", fact.expired_at)?;
    dict.set_item("reference_time", fact.reference_time)?;
    Ok(dict)
}

fn episode_dict<'py>(py: Python<'py>, episode: &Episode) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("id", &episode.id)?;
    dict.set_item("source", &episode.source)?;
    dict.set_item("body", &episode.body)?;
    dict.set_item("occurred_at", episode.occurred_at)?;
    dict.set_item("created_at", episode.created_at)?;
    dict.set_item("properties", props_to_string(&episode.properties))?;
    Ok(dict)
}

fn property_version_dict<'py>(
    py: Python<'py>,
    version: &EntityPropertyVersion,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("entity_id", &version.entity_id)?;
    dict.set_item("key", &version.key)?;
    dict.set_item("value", props_to_string(&version.value))?;
    dict.set_item("episode_id", &version.episode_id)?;
    dict.set_item("valid_at", version.valid_at)?;
    dict.set_item("invalid_at", version.invalid_at)?;
    dict.set_item("created_at", version.created_at)?;
    dict.set_item("expired_at", version.expired_at)?;
    Ok(dict)
}

/// The native temporal knowledge graph handle. `unsendable` like `CoreEngine`,
/// which it owns via its semantic index.
#[pyclass(name = "TemporalKnowledgeGraph", unsendable)]
pub struct PyTemporalGraph {
    inner: TemporalGraph,
}

impl PyTemporalGraph {
    /// Runs a read-path native call with the GIL released. Traversal and
    /// semantic verbs scan SQL tables and the vector index (whose first query
    /// can trigger a lazy ANN build), which must not starve the host's Python
    /// threads.
    fn detached<R: Send>(
        &self,
        py: Python<'_>,
        call: impl FnOnce(&TemporalGraph) -> lodedb_graph::Result<R> + Send,
    ) -> PyResult<R> {
        let graph_addr = &self.inner as *const TemporalGraph as usize;
        py.detach(move || {
            // SAFETY: `self` outlives this synchronous call, the closure runs on
            // the graph-owning thread (the pyclass is unsendable, so no other
            // thread can borrow it), and no Python state is touched while
            // detached; the embedder callback reattaches on its own.
            let inner: &TemporalGraph = unsafe { &*(graph_addr as *const TemporalGraph) };
            call(inner)
        })
        .map_err(to_py_err)
    }

    /// [`Self::detached`] for the mutation verbs, which own the batch-shaped
    /// work (episode/fact ingest loops, reindex, persist).
    fn detached_mut<R: Send>(
        &mut self,
        py: Python<'_>,
        call: impl FnOnce(&mut TemporalGraph) -> lodedb_graph::Result<R> + Send,
    ) -> PyResult<R> {
        let graph_addr = &mut self.inner as *mut TemporalGraph as usize;
        py.detach(move || {
            // SAFETY: as for `detached`, plus exclusivity: the pycell's mutable
            // borrow stays held across the detached section, so a re-entrant
            // borrow from the embedder callback raises a Python borrow error
            // instead of aliasing `inner`.
            let inner: &mut TemporalGraph = unsafe { &mut *(graph_addr as *mut TemporalGraph) };
            call(inner)
        })
        .map_err(to_py_err)
    }
}

#[pymethods]
impl PyTemporalGraph {
    /// Open (or create) a graph. `path=None` is a fully in-memory graph. `embedder`
    /// is a Python object with `dimension` + `embed(texts, role)`; pass `None` for a
    /// vector-in graph (then use the `*_vec` verbs).
    #[new]
    #[pyo3(signature = (path=None, vector_dim=384, embedder=None, index_facts=true, index_text=true))]
    fn new(
        path: Option<&str>,
        vector_dim: usize,
        embedder: Option<Py<PyAny>>,
        index_facts: bool,
        index_text: bool,
    ) -> PyResult<Self> {
        let config = GraphConfig {
            vector_dim,
            index_text,
            index_facts,
        };
        let boxed: Option<Box<dyn Embedder>> = embedder
            .map(|obj| {
                let dim = Python::attach(|py| -> PyResult<usize> {
                    let value = obj.bind(py).getattr("dimension")?;
                    let value = if value.is_callable() {
                        value.call0()?
                    } else {
                        value
                    };
                    value.extract()
                })?;
                if dim != vector_dim {
                    return Err(PyValueError::new_err(format!(
                        "embedder dimension {dim} does not match vector_dim {vector_dim}"
                    )));
                }
                Ok(Box::new(PyEmbedder { obj, dim }) as Box<dyn Embedder>)
            })
            .transpose()?;
        let inner = match path {
            Some(p) => TemporalGraph::open(std::path::Path::new(p), config, boxed),
            None => TemporalGraph::open_in_memory(config, boxed),
        }
        .map_err(to_py_err)?;
        Ok(PyTemporalGraph { inner })
    }

    // -- episodes -----------------------------------------------------------

    #[pyo3(signature = (source, body, occurred_at, properties=None, mentions=Vec::new(), id=None))]
    #[allow(clippy::too_many_arguments)]
    fn add_episode(
        &mut self,
        py: Python<'_>,
        source: &str,
        body: &str,
        occurred_at: i64,
        properties: Option<&str>,
        mentions: Vec<String>,
        id: Option<&str>,
    ) -> PyResult<String> {
        let props = parse_props(properties)?;
        let source = source.to_owned();
        let body = body.to_owned();
        let id = id.map(str::to_owned);
        self.detached_mut(py, move |inner| {
            inner.add_episode_with_id(id.as_deref(), &source, &body, occurred_at, props, &mentions)
        })
    }

    fn get_episode<'py>(&self, py: Python<'py>, id: &str) -> PyResult<Option<Bound<'py, PyDict>>> {
        let id = id.to_owned();
        match self.detached(py, move |inner| inner.get_episode(&id))? {
            Some(ep) => Ok(Some(episode_dict(py, &ep)?)),
            None => Ok(None),
        }
    }

    fn episodes<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.detached(py, |inner| inner.episodes())?
            .iter()
            .map(|episode| episode_dict(py, episode))
            .collect()
    }

    fn facts_by_episode<'py>(
        &self,
        py: Python<'py>,
        id: &str,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let id = id.to_owned();
        self.detached(py, move |inner| inner.facts_by_episode(&id))?
            .iter()
            .map(|fact| fact_dict(py, fact))
            .collect()
    }

    fn remove_episode(&mut self, py: Python<'_>, id: &str) -> PyResult<bool> {
        let id = id.to_owned();
        self.detached_mut(py, move |inner| inner.remove_episode(&id))
    }

    // -- entities -----------------------------------------------------------

    #[pyo3(signature = (id, entity_type, label, properties=None, valid_at=None, invalid_at=None, property_sources=None))]
    fn upsert_entity(
        &mut self,
        py: Python<'_>,
        id: &str,
        entity_type: &str,
        label: &str,
        properties: Option<&str>,
        valid_at: Option<i64>,
        invalid_at: Option<i64>,
        property_sources: Option<&str>,
    ) -> PyResult<String> {
        let props = parse_props(properties)?;
        let sources = parse_property_sources(property_sources)?;
        let id = id.to_owned();
        let entity_type = entity_type.to_owned();
        let label = label.to_owned();
        self.detached_mut(py, move |inner| {
            inner.upsert_entity_with_sources(
                &id,
                &entity_type,
                &label,
                props,
                valid_at,
                invalid_at,
                &sources,
            )
        })
    }

    #[pyo3(signature = (id, entity_type, label, embedding, properties=None, valid_at=None, invalid_at=None, property_sources=None))]
    fn upsert_entity_vec(
        &mut self,
        py: Python<'_>,
        id: &str,
        entity_type: &str,
        label: &str,
        embedding: Vec<f32>,
        properties: Option<&str>,
        valid_at: Option<i64>,
        invalid_at: Option<i64>,
        property_sources: Option<&str>,
    ) -> PyResult<String> {
        let props = parse_props(properties)?;
        let sources = parse_property_sources(property_sources)?;
        let id = id.to_owned();
        let entity_type = entity_type.to_owned();
        let label = label.to_owned();
        self.detached_mut(py, move |inner| {
            inner.upsert_entity_vec_with_sources(
                &id,
                &entity_type,
                &label,
                props,
                &embedding,
                valid_at,
                invalid_at,
                &sources,
            )
        })
    }

    fn get_entity<'py>(&self, py: Python<'py>, id: &str) -> PyResult<Option<Bound<'py, PyDict>>> {
        let id = id.to_owned();
        match self.detached(py, move |inner| inner.get_entity(&id))? {
            Some(e) => Ok(Some(entity_dict(py, &e)?)),
            None => Ok(None),
        }
    }

    #[pyo3(signature = (id, key=None))]
    fn entity_property_history<'py>(
        &self,
        py: Python<'py>,
        id: &str,
        key: Option<&str>,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let id = id.to_owned();
        let key = key.map(str::to_owned);
        self.detached(py, move |inner| inner.entity_property_history(&id, key.as_deref()))?
            .iter()
            .map(|version| property_version_dict(py, version))
            .collect()
    }

    #[pyo3(signature = (entity_type=None, as_of_ms=None, all_time=false, strict_now=false, known_at_ms=None))]
    fn entities<'py>(
        &self,
        py: Python<'py>,
        entity_type: Option<&str>,
        as_of_ms: Option<i64>,
        all_time: bool,
        strict_now: bool,
        known_at_ms: Option<i64>,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let as_of = resolve_as_of(as_of_ms, all_time, strict_now, known_at_ms)?;
        let entity_type = entity_type.map(str::to_owned);
        let out =
            self.detached(py, move |inner| inner.entities(entity_type.as_deref(), as_of))?;
        out.iter().map(|e| entity_dict(py, e)).collect()
    }

    // -- facts --------------------------------------------------------------

    #[pyo3(signature = (src, relation, dst, fact, properties=None, episodes=Vec::new(), valid_at=None, invalidates=Vec::new(), id=None))]
    #[allow(clippy::too_many_arguments)]
    fn add_fact(
        &mut self,
        py: Python<'_>,
        src: &str,
        relation: &str,
        dst: &str,
        fact: &str,
        properties: Option<&str>,
        episodes: Vec<String>,
        valid_at: Option<i64>,
        invalidates: Vec<String>,
        id: Option<&str>,
    ) -> PyResult<String> {
        let props = parse_props(properties)?;
        let src = src.to_owned();
        let relation = relation.to_owned();
        let dst = dst.to_owned();
        let fact = fact.to_owned();
        let id = id.map(str::to_owned);
        self.detached_mut(py, move |inner| {
            inner.add_fact_with_id(
                id.as_deref(),
                &src,
                &relation,
                &dst,
                &fact,
                props,
                episodes,
                valid_at,
                &invalidates,
            )
        })
    }

    #[pyo3(signature = (src, relation, dst, fact, embedding, properties=None, episodes=Vec::new(), valid_at=None, invalidates=Vec::new(), id=None))]
    #[allow(clippy::too_many_arguments)]
    fn add_fact_vec(
        &mut self,
        py: Python<'_>,
        src: &str,
        relation: &str,
        dst: &str,
        fact: &str,
        embedding: Vec<f32>,
        properties: Option<&str>,
        episodes: Vec<String>,
        valid_at: Option<i64>,
        invalidates: Vec<String>,
        id: Option<&str>,
    ) -> PyResult<String> {
        let props = parse_props(properties)?;
        let src = src.to_owned();
        let relation = relation.to_owned();
        let dst = dst.to_owned();
        let fact = fact.to_owned();
        let id = id.map(str::to_owned);
        self.detached_mut(py, move |inner| {
            inner.add_fact_vec_with_id(
                id.as_deref(),
                &src,
                &relation,
                &dst,
                &fact,
                props,
                episodes,
                valid_at,
                &invalidates,
                &embedding,
            )
        })
    }

    #[pyo3(signature = (id, invalid_at=None))]
    fn invalidate_fact(
        &mut self,
        py: Python<'_>,
        id: &str,
        invalid_at: Option<i64>,
    ) -> PyResult<bool> {
        let id = id.to_owned();
        self.detached_mut(py, move |inner| inner.invalidate_fact(&id, invalid_at))
    }

    fn get_fact<'py>(&self, py: Python<'py>, id: &str) -> PyResult<Option<Bound<'py, PyDict>>> {
        let id = id.to_owned();
        match self.detached(py, move |inner| inner.get_fact(&id))? {
            Some(f) => Ok(Some(fact_dict(py, &f)?)),
            None => Ok(None),
        }
    }

    fn remove_entity(&mut self, py: Python<'_>, id: &str) -> PyResult<bool> {
        let id = id.to_owned();
        self.detached_mut(py, move |inner| inner.remove_entity(&id))
    }

    fn remove_fact(&mut self, py: Python<'_>, id: &str) -> PyResult<bool> {
        let id = id.to_owned();
        self.detached_mut(py, move |inner| inner.remove_fact(&id))
    }

    // -- traversal ----------------------------------------------------------

    #[pyo3(signature = (id, direction="out", relation=None, as_of_ms=None, all_time=false, strict_now=false, known_at_ms=None))]
    fn neighbors<'py>(
        &self,
        py: Python<'py>,
        id: &str,
        direction: &str,
        relation: Option<&str>,
        as_of_ms: Option<i64>,
        all_time: bool,
        strict_now: bool,
        known_at_ms: Option<i64>,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let dir = Direction::parse(direction).map_err(to_py_err)?;
        let as_of = resolve_as_of(as_of_ms, all_time, strict_now, known_at_ms)?;
        let id = id.to_owned();
        let relation = relation.map(str::to_owned);
        let facts = self.detached(py, move |inner| {
            inner.neighbors(&id, dir, relation.as_deref(), as_of)
        })?;
        facts.iter().map(|f| fact_dict(py, f)).collect()
    }

    #[pyo3(signature = (seeds, k=1, direction="both", as_of_ms=None, all_time=false, strict_now=false, known_at_ms=None, predicate=None))]
    fn k_hop<'py>(
        &self,
        py: Python<'py>,
        seeds: Vec<String>,
        k: usize,
        direction: &str,
        as_of_ms: Option<i64>,
        all_time: bool,
        strict_now: bool,
        known_at_ms: Option<i64>,
        predicate: Option<&str>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dir = Direction::parse(direction).map_err(to_py_err)?;
        let predicate = parse_optional_json(predicate, "predicate")?;
        let as_of = resolve_as_of(as_of_ms, all_time, strict_now, known_at_ms)?;
        let sub = self.detached(py, move |inner| {
            inner.k_hop_filtered(&seeds, k, dir, as_of, predicate.as_ref())
        })?;
        subgraph_dict(py, &sub)
    }

    // -- semantic retrieval -------------------------------------------------

    #[pyo3(signature = (query=None, embedding=None, k=10, entity_type=None, as_of_ms=None, all_time=false, strict_now=false, known_at_ms=None, predicate=None))]
    fn semantic_entities<'py>(
        &self,
        py: Python<'py>,
        query: Option<&str>,
        embedding: Option<Vec<f32>>,
        k: usize,
        entity_type: Option<&str>,
        as_of_ms: Option<i64>,
        all_time: bool,
        strict_now: bool,
        known_at_ms: Option<i64>,
        predicate: Option<&str>,
    ) -> PyResult<Vec<(f32, Bound<'py, PyDict>)>> {
        let predicate = parse_optional_json(predicate, "predicate")?;
        let as_of = resolve_as_of(as_of_ms, all_time, strict_now, known_at_ms)?;
        let query = query.map(str::to_owned);
        let entity_type = entity_type.map(str::to_owned);
        let hits = self.detached(py, move |inner| {
            inner.semantic_entities_filtered(
                query.as_deref(),
                embedding.as_deref(),
                k,
                entity_type.as_deref(),
                as_of,
                predicate.as_ref(),
            )
        })?;
        hits.iter()
            .map(|(score, e)| Ok((*score, entity_dict(py, e)?)))
            .collect()
    }

    #[pyo3(signature = (query=None, embedding=None, k=10, relation=None, as_of_ms=None, all_time=false, strict_now=false, known_at_ms=None, predicate=None))]
    fn semantic_facts<'py>(
        &self,
        py: Python<'py>,
        query: Option<&str>,
        embedding: Option<Vec<f32>>,
        k: usize,
        relation: Option<&str>,
        as_of_ms: Option<i64>,
        all_time: bool,
        strict_now: bool,
        known_at_ms: Option<i64>,
        predicate: Option<&str>,
    ) -> PyResult<Vec<(f32, Bound<'py, PyDict>)>> {
        let predicate = parse_optional_json(predicate, "predicate")?;
        let as_of = resolve_as_of(as_of_ms, all_time, strict_now, known_at_ms)?;
        let query = query.map(str::to_owned);
        let relation = relation.map(str::to_owned);
        let hits = self.detached(py, move |inner| {
            inner.semantic_facts_filtered(
                query.as_deref(),
                embedding.as_deref(),
                k,
                relation.as_deref(),
                as_of,
                predicate.as_ref(),
            )
        })?;
        hits.iter()
            .map(|(score, f)| Ok((*score, fact_dict(py, f)?)))
            .collect()
    }

    #[pyo3(signature = (query=None, embedding=None, k=5, hops=1, direction="both", entity_type=None, relation=None, as_of_ms=None, all_time=false, strict_now=false, known_at_ms=None, predicate=None, seed_kind="entity"))]
    #[allow(clippy::too_many_arguments)]
    fn search_subgraph<'py>(
        &self,
        py: Python<'py>,
        query: Option<&str>,
        embedding: Option<Vec<f32>>,
        k: usize,
        hops: usize,
        direction: &str,
        entity_type: Option<&str>,
        relation: Option<&str>,
        as_of_ms: Option<i64>,
        all_time: bool,
        strict_now: bool,
        known_at_ms: Option<i64>,
        predicate: Option<&str>,
        seed_kind: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dir = Direction::parse(direction).map_err(to_py_err)?;
        let predicate = parse_optional_json(predicate, "predicate")?;
        let as_of = resolve_as_of(as_of_ms, all_time, strict_now, known_at_ms)?;
        let query = query.map(str::to_owned);
        let entity_type = entity_type.map(str::to_owned);
        let relation = relation.map(str::to_owned);
        let seed_kind = seed_kind.to_owned();
        let sub = self.detached(py, move |inner| {
            inner.search_subgraph_filtered(
                query.as_deref(),
                embedding.as_deref(),
                k,
                hops,
                dir,
                entity_type.as_deref(),
                relation.as_deref(),
                as_of,
                predicate.as_ref(),
                &seed_kind,
            )
        })?;
        subgraph_dict(py, &sub)
    }

    #[pyo3(signature = (name, k=5, predicate=None))]
    fn resolve_entity<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        k: usize,
        predicate: Option<&str>,
    ) -> PyResult<Vec<(f32, Bound<'py, PyDict>)>> {
        let predicate = parse_optional_json(predicate, "predicate")?;
        let name = name.to_owned();
        let hits = self.detached(py, move |inner| {
            inner.resolve_entity_filtered(&name, k, predicate.as_ref())
        })?;
        hits.iter()
            .map(|(score, e)| Ok((*score, entity_dict(py, e)?)))
            .collect()
    }

    fn history<'py>(&self, py: Python<'py>, entity_id: &str) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let entity_id = entity_id.to_owned();
        let facts = self.detached(py, move |inner| inner.history(&entity_id))?;
        facts.iter().map(|f| fact_dict(py, f)).collect()
    }

    // -- maintenance --------------------------------------------------------

    fn reindex<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = self.detached_mut(py, |inner| inner.reindex())?;
        let dict = PyDict::new(py);
        dict.set_item("reindexed_entities", stats.reindexed_entities)?;
        dict.set_item("reindexed_facts", stats.reindexed_facts)?;
        dict.set_item("removed_orphans", stats.removed_orphans)?;
        Ok(dict)
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = self.detached(py, |inner| inner.stats())?;
        let dict = PyDict::new(py);
        dict.set_item("entities", stats.entities)?;
        dict.set_item("facts", stats.facts)?;
        dict.set_item("indexed_documents", stats.indexed_documents)?;
        Ok(dict)
    }

    fn persist(&mut self, py: Python<'_>) -> PyResult<()> {
        self.detached_mut(py, |inner| inner.persist())
    }
}

fn subgraph_dict<'py>(
    py: Python<'py>,
    sub: &lodedb_graph::Subgraph,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    let entities: Vec<Bound<'py, PyDict>> =
        sub.entities.values().map(|e| entity_dict(py, e)).collect::<PyResult<_>>()?;
    let facts: Vec<Bound<'py, PyDict>> =
        sub.facts.iter().map(|f| fact_dict(py, f)).collect::<PyResult<_>>()?;
    dict.set_item("entities", entities)?;
    dict.set_item("facts", facts)?;
    dict.set_item("seeds", sub.seeds.clone())?;
    Ok(dict)
}

#[pyfunction(name = "rrf", signature = (ranked_lists, k=1.0))]
fn py_rrf(ranked_lists: Vec<Vec<String>>, k: f32) -> Vec<(String, f32)> {
    lodedb_graph::rerank::rrf(&ranked_lists, k)
}

#[pyfunction(name = "maximal_marginal_relevance", signature = (query, candidates, lambda=0.5))]
fn py_maximal_marginal_relevance(
    query: Vec<f32>,
    candidates: Vec<(String, Vec<f32>)>,
    lambda: f32,
) -> Vec<String> {
    lodedb_graph::rerank::maximal_marginal_relevance(&query, &candidates, lambda)
}

#[pyfunction(name = "node_distance_reranker")]
fn py_node_distance_reranker(
    ids: Vec<String>,
    distances: HashMap<String, usize>,
) -> Vec<String> {
    lodedb_graph::rerank::node_distance_reranker(&ids, &distances)
}

#[pyfunction(name = "episode_mentions_reranker")]
fn py_episode_mentions_reranker(
    ids: Vec<String>,
    mention_counts: HashMap<String, usize>,
) -> Vec<String> {
    lodedb_graph::rerank::episode_mentions_reranker(&ids, &mention_counts)
}

/// Register the `graph` submodule on the parent `_turbovec` module.
pub(crate) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = PyModule::new(parent.py(), "graph")?;
    module.add_class::<PyTemporalGraph>()?;
    module.add_function(wrap_pyfunction!(py_rrf, &module)?)?;
    module.add_function(wrap_pyfunction!(py_maximal_marginal_relevance, &module)?)?;
    module.add_function(wrap_pyfunction!(py_node_distance_reranker, &module)?)?;
    module.add_function(wrap_pyfunction!(py_episode_mentions_reranker, &module)?)?;
    parent.add_submodule(&module)?;
    Ok(())
}
