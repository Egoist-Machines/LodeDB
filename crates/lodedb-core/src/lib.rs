//! Native Rust core for LodeDB.
//!
//! The Python engine remains authoritative during the migration. This crate
//! starts as a dependency-light home for deterministic core semantics that will
//! move from Python into Rust milestone by milestone.

pub mod error;
pub mod types;
pub mod version;

pub use error::{CoreError, CoreErrorCode};
pub use types::CoreApiVersion;
pub use version::{CORE_VERSION, STORAGE_SCHEMA_VERSION};
