//! Hidden PyO3 bindings for the native LodeDB core.
//!
//! This crate is intentionally not wired into the Python public API yet. Later
//! migration milestones can expose shadow-mode helpers without changing
//! `import lodedb` or importing heavy embedding dependencies.

use pyo3::prelude::*;

/// Returns the native-core crate version.
#[pyfunction]
fn native_core_version() -> &'static str {
    lodedb_core::CORE_VERSION
}

/// Returns the persisted-store schema version understood by the native core.
#[pyfunction]
fn storage_schema_version() -> u32 {
    lodedb_core::STORAGE_SCHEMA_VERSION
}

/// Hidden native-core Python module.
#[pymodule]
fn _native_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", lodedb_core::CORE_VERSION)?;
    module.add_function(wrap_pyfunction!(native_core_version, module)?)?;
    module.add_function(wrap_pyfunction!(storage_schema_version, module)?)?;
    Ok(())
}
