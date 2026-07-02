//! Native-core version constants.

/// Crate version, kept in sync with the Python package for the migration branch.
pub const CORE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Current persisted-store schema version understood by the Python oracle.
pub const STORAGE_SCHEMA_VERSION: u32 = 1;

/// Stable native-core ABI version shared by C and Python extension shims, so a
/// wrapper linked against a mismatched core fails the runtime version check
/// cleanly instead of a missing-symbol error or a silent shape mismatch.
///
/// Bumped to 2 for two changes in this release's ABI: the reader-freshness FFI
/// (`lodedb_engine_refresh` / `lodedb_engine_applied_lsn`) was added, and the C
/// create ABI was consolidated (the positional
/// `lodedb_engine_create_index`/`_with_model` functions were retired for the
/// single `lodedb_engine_create_index_json` entry point taking a minimal create
/// request). The Python create path (PyO3 `create_index_with_options`) is
/// unchanged.
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
