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
    AsOf, Direction, EmbedRole, Embedder, Entity, Episode, Fact, GraphConfig, GraphError,
    TemporalGraph,
};
use serde_json::Value;

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
fn resolve_as_of(as_of_ms: Option<i64>, all_time: bool) -> AsOf {
    if all_time {
        AsOf::All
    } else if let Some(t) = as_of_ms {
        AsOf::At(t)
    } else {
        AsOf::Now
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

fn props_to_string(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "null".to_string())
}

/// A caller-supplied Python embedder, wrapped as a Rust [`Embedder`]. Holds the
/// Python object and calls its `embed(texts, role)` under the GIL.
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
        Python::with_gil(|py| {
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

/// The native temporal knowledge graph handle. `unsendable` like `CoreEngine`,
/// which it owns via its semantic index.
#[pyclass(name = "TemporalKnowledgeGraph", unsendable)]
pub struct PyTemporalGraph {
    inner: TemporalGraph,
}

#[pymethods]
impl PyTemporalGraph {
    /// Open (or create) a graph. `path=None` is a fully in-memory graph. `embedder`
    /// is a Python object with `dimension` + `embed(texts, role)`; pass `None` for a
    /// vector-in graph (then use the `*_vec` verbs).
    #[new]
    #[pyo3(signature = (path=None, vector_dim=384, embedder=None, index_facts=true))]
    fn new(
        path: Option<&str>,
        vector_dim: usize,
        embedder: Option<Py<PyAny>>,
        index_facts: bool,
    ) -> PyResult<Self> {
        let config = GraphConfig {
            vector_dim,
            index_text: true,
            index_facts,
        };
        let boxed: Option<Box<dyn Embedder>> = embedder.map(|obj| {
            Box::new(PyEmbedder { obj, dim: vector_dim }) as Box<dyn Embedder>
        });
        let inner = match path {
            Some(p) => TemporalGraph::open(std::path::Path::new(p), config, boxed),
            None => TemporalGraph::open_in_memory(config, boxed),
        }
        .map_err(to_py_err)?;
        Ok(PyTemporalGraph { inner })
    }

    // -- episodes -----------------------------------------------------------

    #[pyo3(signature = (source, body, occurred_at, properties=None, mentions=Vec::new()))]
    fn add_episode(
        &mut self,
        source: &str,
        body: &str,
        occurred_at: i64,
        properties: Option<&str>,
        mentions: Vec<String>,
    ) -> PyResult<String> {
        let props = parse_props(properties)?;
        self.inner
            .add_episode(source, body, occurred_at, props, &mentions)
            .map_err(to_py_err)
    }

    fn get_episode<'py>(&self, py: Python<'py>, id: &str) -> PyResult<Option<Bound<'py, PyDict>>> {
        match self.inner.get_episode(id).map_err(to_py_err)? {
            Some(ep) => Ok(Some(episode_dict(py, &ep)?)),
            None => Ok(None),
        }
    }

    // -- entities -----------------------------------------------------------

    #[pyo3(signature = (id, entity_type, label, properties=None, valid_at=None, invalid_at=None))]
    fn upsert_entity(
        &mut self,
        id: &str,
        entity_type: &str,
        label: &str,
        properties: Option<&str>,
        valid_at: Option<i64>,
        invalid_at: Option<i64>,
    ) -> PyResult<String> {
        let props = parse_props(properties)?;
        self.inner
            .upsert_entity(id, entity_type, label, props, valid_at, invalid_at)
            .map_err(to_py_err)
    }

    #[pyo3(signature = (id, entity_type, label, embedding, properties=None, valid_at=None, invalid_at=None))]
    fn upsert_entity_vec(
        &mut self,
        id: &str,
        entity_type: &str,
        label: &str,
        embedding: Vec<f32>,
        properties: Option<&str>,
        valid_at: Option<i64>,
        invalid_at: Option<i64>,
    ) -> PyResult<String> {
        let props = parse_props(properties)?;
        self.inner
            .upsert_entity_vec(id, entity_type, label, props, &embedding, valid_at, invalid_at)
            .map_err(to_py_err)
    }

    fn get_entity<'py>(&self, py: Python<'py>, id: &str) -> PyResult<Option<Bound<'py, PyDict>>> {
        match self.inner.get_entity(id).map_err(to_py_err)? {
            Some(e) => Ok(Some(entity_dict(py, &e)?)),
            None => Ok(None),
        }
    }

    #[pyo3(signature = (entity_type=None, as_of_ms=None, all_time=false))]
    fn entities<'py>(
        &self,
        py: Python<'py>,
        entity_type: Option<&str>,
        as_of_ms: Option<i64>,
        all_time: bool,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let out = self
            .inner
            .entities(entity_type, resolve_as_of(as_of_ms, all_time))
            .map_err(to_py_err)?;
        out.iter().map(|e| entity_dict(py, e)).collect()
    }

    // -- facts --------------------------------------------------------------

    #[pyo3(signature = (src, relation, dst, fact, properties=None, episodes=Vec::new(), valid_at=None, invalidates=Vec::new()))]
    #[allow(clippy::too_many_arguments)]
    fn add_fact(
        &mut self,
        src: &str,
        relation: &str,
        dst: &str,
        fact: &str,
        properties: Option<&str>,
        episodes: Vec<String>,
        valid_at: Option<i64>,
        invalidates: Vec<String>,
    ) -> PyResult<String> {
        let props = parse_props(properties)?;
        self.inner
            .add_fact(src, relation, dst, fact, props, episodes, valid_at, &invalidates)
            .map_err(to_py_err)
    }

    #[pyo3(signature = (id, invalid_at=None))]
    fn invalidate_fact(&mut self, id: &str, invalid_at: Option<i64>) -> PyResult<bool> {
        self.inner.invalidate_fact(id, invalid_at).map_err(to_py_err)
    }

    fn get_fact<'py>(&self, py: Python<'py>, id: &str) -> PyResult<Option<Bound<'py, PyDict>>> {
        match self.inner.get_fact(id).map_err(to_py_err)? {
            Some(f) => Ok(Some(fact_dict(py, &f)?)),
            None => Ok(None),
        }
    }

    fn remove_entity(&mut self, id: &str) -> PyResult<bool> {
        self.inner.remove_entity(id).map_err(to_py_err)
    }

    fn remove_fact(&mut self, id: &str) -> PyResult<bool> {
        self.inner.remove_fact(id).map_err(to_py_err)
    }

    // -- traversal ----------------------------------------------------------

    #[pyo3(signature = (id, direction="out", relation=None, as_of_ms=None, all_time=false))]
    fn neighbors<'py>(
        &self,
        py: Python<'py>,
        id: &str,
        direction: &str,
        relation: Option<&str>,
        as_of_ms: Option<i64>,
        all_time: bool,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let dir = Direction::parse(direction).map_err(to_py_err)?;
        let facts = self
            .inner
            .neighbors(id, dir, relation, resolve_as_of(as_of_ms, all_time))
            .map_err(to_py_err)?;
        facts.iter().map(|f| fact_dict(py, f)).collect()
    }

    #[pyo3(signature = (seeds, k=1, direction="both", as_of_ms=None, all_time=false))]
    fn k_hop<'py>(
        &self,
        py: Python<'py>,
        seeds: Vec<String>,
        k: usize,
        direction: &str,
        as_of_ms: Option<i64>,
        all_time: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dir = Direction::parse(direction).map_err(to_py_err)?;
        let sub = self
            .inner
            .k_hop(&seeds, k, dir, resolve_as_of(as_of_ms, all_time))
            .map_err(to_py_err)?;
        subgraph_dict(py, &sub)
    }

    // -- semantic retrieval -------------------------------------------------

    #[pyo3(signature = (query=None, embedding=None, k=10, entity_type=None, as_of_ms=None, all_time=false))]
    fn semantic_entities<'py>(
        &self,
        py: Python<'py>,
        query: Option<&str>,
        embedding: Option<Vec<f32>>,
        k: usize,
        entity_type: Option<&str>,
        as_of_ms: Option<i64>,
        all_time: bool,
    ) -> PyResult<Vec<(f32, Bound<'py, PyDict>)>> {
        let hits = self
            .inner
            .semantic_entities(
                query,
                embedding.as_deref(),
                k,
                entity_type,
                resolve_as_of(as_of_ms, all_time),
            )
            .map_err(to_py_err)?;
        hits.iter()
            .map(|(score, e)| Ok((*score, entity_dict(py, e)?)))
            .collect()
    }

    #[pyo3(signature = (query=None, embedding=None, k=10, relation=None, as_of_ms=None, all_time=false))]
    fn semantic_facts<'py>(
        &self,
        py: Python<'py>,
        query: Option<&str>,
        embedding: Option<Vec<f32>>,
        k: usize,
        relation: Option<&str>,
        as_of_ms: Option<i64>,
        all_time: bool,
    ) -> PyResult<Vec<(f32, Bound<'py, PyDict>)>> {
        let hits = self
            .inner
            .semantic_facts(
                query,
                embedding.as_deref(),
                k,
                relation,
                resolve_as_of(as_of_ms, all_time),
            )
            .map_err(to_py_err)?;
        hits.iter()
            .map(|(score, f)| Ok((*score, fact_dict(py, f)?)))
            .collect()
    }

    #[pyo3(signature = (query=None, embedding=None, k=5, hops=1, direction="both", entity_type=None, as_of_ms=None, all_time=false))]
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
        as_of_ms: Option<i64>,
        all_time: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dir = Direction::parse(direction).map_err(to_py_err)?;
        let sub = self
            .inner
            .search_subgraph(
                query,
                embedding.as_deref(),
                k,
                hops,
                dir,
                entity_type,
                resolve_as_of(as_of_ms, all_time),
            )
            .map_err(to_py_err)?;
        subgraph_dict(py, &sub)
    }

    #[pyo3(signature = (name, k=5))]
    fn resolve_entity<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        k: usize,
    ) -> PyResult<Vec<(f32, Bound<'py, PyDict>)>> {
        let hits = self.inner.resolve_entity(name, k).map_err(to_py_err)?;
        hits.iter()
            .map(|(score, e)| Ok((*score, entity_dict(py, e)?)))
            .collect()
    }

    fn history<'py>(&self, py: Python<'py>, entity_id: &str) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let facts = self.inner.history(entity_id).map_err(to_py_err)?;
        facts.iter().map(|f| fact_dict(py, f)).collect()
    }

    // -- maintenance --------------------------------------------------------

    fn reindex<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = self.inner.reindex().map_err(to_py_err)?;
        let dict = PyDict::new(py);
        dict.set_item("reindexed_entities", stats.reindexed_entities)?;
        dict.set_item("reindexed_facts", stats.reindexed_facts)?;
        dict.set_item("removed_orphans", stats.removed_orphans)?;
        Ok(dict)
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = self.inner.stats().map_err(to_py_err)?;
        let dict = PyDict::new(py);
        dict.set_item("entities", stats.entities)?;
        dict.set_item("facts", stats.facts)?;
        dict.set_item("indexed_documents", stats.indexed_documents)?;
        Ok(dict)
    }

    fn persist(&mut self) -> PyResult<()> {
        self.inner.persist().map_err(to_py_err)
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

/// Register the `graph` submodule on the parent `_turbovec` module.
pub(crate) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = PyModule::new(parent.py(), "graph")?;
    module.add_class::<PyTemporalGraph>()?;
    parent.add_submodule(&module)?;
    Ok(())
}
