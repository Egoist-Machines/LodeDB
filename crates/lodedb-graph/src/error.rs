//! Error type for the bi-temporal graph layer.
//!
//! Mirrors the shape of `lodedb_core::CoreError`: a small, matchable enum the
//! FFI and pyo3 bindings map onto their existing error surfaces (`CoreErrorCode`
//! and Python exceptions respectively).

use std::fmt;

/// A `lodedb-graph` operation error.
#[derive(Debug)]
pub enum GraphError {
    /// A caller passed an argument that cannot be honored (empty id, bad direction,
    /// a lexical/hybrid mode paired with a precomputed embedding, ...).
    InvalidArgument(String),
    /// A referenced entity, fact, or episode does not exist.
    NotFound(String),
    /// The SQL topology store (source of truth) failed.
    Topology(String),
    /// The `lodedb-core` semantic index failed.
    Index(String),
    /// The caller-supplied embedder failed or returned the wrong shape.
    Embedding(String),
    /// An invariant this crate owns was violated.
    Internal(String),
}

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, GraphError>;

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphError::InvalidArgument(m) => write!(f, "invalid argument: {m}"),
            GraphError::NotFound(m) => write!(f, "not found: {m}"),
            GraphError::Topology(m) => write!(f, "topology store error: {m}"),
            GraphError::Index(m) => write!(f, "semantic index error: {m}"),
            GraphError::Embedding(m) => write!(f, "embedding error: {m}"),
            GraphError::Internal(m) => write!(f, "internal error: {m}"),
        }
    }
}

impl std::error::Error for GraphError {}

impl From<rusqlite::Error> for GraphError {
    fn from(e: rusqlite::Error) -> Self {
        GraphError::Topology(e.to_string())
    }
}

impl From<lodedb_core::CoreError> for GraphError {
    fn from(e: lodedb_core::CoreError) -> Self {
        GraphError::Index(e.to_string())
    }
}

impl From<serde_json::Error> for GraphError {
    fn from(e: serde_json::Error) -> Self {
        GraphError::Internal(format!("json: {e}"))
    }
}
