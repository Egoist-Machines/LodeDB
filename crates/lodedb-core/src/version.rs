//! Native-core version constants.

/// Crate version, kept in sync with the Python package for the migration branch.
pub const CORE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Current persisted-store schema version understood by the Python oracle.
pub const STORAGE_SCHEMA_VERSION: u32 = 1;

/// Stable native-core ABI version shared by C and Python extension shims.
///
/// Bumped to 2 when the C create ABI was consolidated: the positional
/// `lodedb_engine_create_index`/`_with_model` functions were retired and the
/// single `lodedb_engine_create_index_json` entry point now takes a minimal
/// create request. A binding built against the old ABI (missing symbols, or the
/// old full-options JSON) mismatches this, and the runtime version check catches
/// it. The Python create path (PyO3 `create_index_with_options`) is unchanged.
pub const NATIVE_CORE_ABI_VERSION: u32 = 2;

#[cfg(test)]
mod tests {
    use super::{CORE_VERSION, NATIVE_CORE_ABI_VERSION, STORAGE_SCHEMA_VERSION};

    #[test]
    fn exposes_version_constants() {
        // CORE_VERSION tracks the crate version, so check it is populated rather
        // than pinning a literal that goes stale on every release bump.
        assert!(!CORE_VERSION.is_empty());
        assert_eq!(STORAGE_SCHEMA_VERSION, 1);
        assert_eq!(NATIVE_CORE_ABI_VERSION, 2);
    }
}
