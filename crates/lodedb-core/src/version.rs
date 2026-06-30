//! Native-core version constants.

/// Crate version, kept in sync with the Python package for the migration branch.
pub const CORE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Current persisted-store schema version understood by the Python oracle.
pub const STORAGE_SCHEMA_VERSION: u32 = 1;

/// Stable native-core ABI version shared by C and Python extension shims.
pub const NATIVE_CORE_ABI_VERSION: u32 = 1;

#[cfg(test)]
mod tests {
    use super::{CORE_VERSION, NATIVE_CORE_ABI_VERSION, STORAGE_SCHEMA_VERSION};

    #[test]
    fn exposes_version_constants() {
        // CORE_VERSION tracks the crate version, so check it is populated rather
        // than pinning a literal that goes stale on every release bump.
        assert!(!CORE_VERSION.is_empty());
        assert_eq!(STORAGE_SCHEMA_VERSION, 1);
        assert_eq!(NATIVE_CORE_ABI_VERSION, 1);
    }
}
