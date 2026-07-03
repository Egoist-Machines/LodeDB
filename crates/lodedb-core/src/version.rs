//! Native-core version constants.

/// Crate version, kept in sync with the Python package for the migration branch.
pub const CORE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Current persisted-store schema version understood by the Python oracle.
pub const STORAGE_SCHEMA_VERSION: u32 = 1;

/// Stable native-core ABI version shared by C and Python extension shims, so a
/// wrapper linked against a mismatched core fails the runtime version check
/// cleanly instead of a missing-symbol error or a silent shape mismatch.
///
/// Bumped to 2 earlier this release for the reader-freshness FFI
/// (`lodedb_engine_refresh` / `lodedb_engine_applied_lsn`) and the consolidated C
/// create ABI. Bumped to 3 for the multi-producer text-append surface: the appender
/// gained `lodedb_appender_prepare_documents_json` /
/// `lodedb_appender_append_embedded_documents_json` (and the matching PyO3
/// `CoreAppender.prepare_documents` / `append_embedded_documents`), so a wrapper
/// built against the new header must not pair with an older core that lacks them.
/// Bumped to 4 for the running single-checkpointer: the new
/// `lodedb_checkpointer_open_json` / `lodedb_checkpointer_checkpoint` /
/// `lodedb_checkpointer_free` symbols (and the matching PyO3 `CoreCheckpointer`), so
/// a wrapper that drives a checkpointer must not pair with an older core lacking it.
pub const NATIVE_CORE_ABI_VERSION: u32 = 4;

#[cfg(test)]
mod tests {
    use super::{CORE_VERSION, NATIVE_CORE_ABI_VERSION, STORAGE_SCHEMA_VERSION};

    #[test]
    fn exposes_version_constants() {
        // CORE_VERSION tracks the crate version, so check it is populated rather
        // than pinning a literal that goes stale on every release bump.
        assert!(!CORE_VERSION.is_empty());
        assert_eq!(STORAGE_SCHEMA_VERSION, 1);
        assert_eq!(NATIVE_CORE_ABI_VERSION, 4);
    }
}
