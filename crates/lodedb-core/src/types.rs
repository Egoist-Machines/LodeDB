//! Shared native-core wire contracts.

use crate::version::{CORE_VERSION, STORAGE_SCHEMA_VERSION};

/// Minimal API-version record exposed before the full domain model lands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreApiVersion {
    /// Native-core crate version.
    pub core_version: &'static str,
    /// Persisted-store schema version.
    pub storage_schema_version: u32,
}

impl Default for CoreApiVersion {
    fn default() -> Self {
        Self {
            core_version: CORE_VERSION,
            storage_schema_version: STORAGE_SCHEMA_VERSION,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CoreApiVersion;

    #[test]
    fn default_api_version_matches_constants() {
        let version = CoreApiVersion::default();
        assert_eq!(version.core_version, "0.4.0");
        assert_eq!(version.storage_schema_version, 1);
    }
}
