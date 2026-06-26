//! Stable error codes for native-core bindings.

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Stable native-core error categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CoreErrorCode {
    /// A caller supplied an invalid argument.
    InvalidArgument,
    /// A requested object was not found.
    NotFound,
    /// A persisted store or sidecar is corrupt.
    CorruptStore,
    /// A prepared mutation plan no longer matches the open generation.
    PlanStale,
    /// The operation is not yet implemented in the native core.
    Unsupported,
    /// Any internal error that does not fit a more specific code.
    Internal,
}

impl CoreErrorCode {
    /// Returns the stable string representation used by bindings and diagnostics.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InvalidArgument => "INVALID_ARGUMENT",
            Self::NotFound => "NOT_FOUND",
            Self::CorruptStore => "CORRUPT_STORE",
            Self::PlanStale => "PLAN_STALE",
            Self::Unsupported => "UNSUPPORTED",
            Self::Internal => "INTERNAL",
        }
    }

    /// Returns the stable integer status code reserved for the future C ABI.
    pub fn ffi_status_code(self) -> u32 {
        match self {
            Self::InvalidArgument => 1,
            Self::NotFound => 2,
            Self::CorruptStore => 3,
            Self::PlanStale => 4,
            Self::Unsupported => 5,
            Self::Internal => 255,
        }
    }
}

/// Native-core error with a stable code and redacted message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoreError {
    code: CoreErrorCode,
    message: String,
}

impl CoreError {
    /// Creates a new native-core error.
    pub fn new(code: CoreErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    /// Returns the stable error code.
    pub fn code(&self) -> CoreErrorCode {
        self.code
    }

    /// Returns the redacted error message.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl Display for CoreError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{:?}: {}", self.code, self.message)
    }
}

impl Error for CoreError {}

#[cfg(test)]
mod tests {
    use super::{CoreError, CoreErrorCode};

    #[test]
    fn error_exposes_code_and_message() {
        let error = CoreError::new(CoreErrorCode::Unsupported, "native core is disabled");
        assert_eq!(error.code(), CoreErrorCode::Unsupported);
        assert_eq!(error.message(), "native core is disabled");
        assert_eq!(error.code().as_str(), "UNSUPPORTED");
        assert_eq!(error.code().ffi_status_code(), 5);
        assert_eq!(error.to_string(), "Unsupported: native core is disabled");
    }

    #[test]
    fn error_serializes_with_stable_code() {
        let error = CoreError::new(CoreErrorCode::PlanStale, "prepared plan is stale");
        let encoded = serde_json::to_string(&error).expect("serialize error");
        assert!(encoded.contains("\"code\":\"PLAN_STALE\""));
        let decoded: CoreError = serde_json::from_str(&encoded).expect("deserialize error");
        assert_eq!(decoded.code(), CoreErrorCode::PlanStale);
        assert_eq!(decoded.message(), "prepared plan is stale");
    }
}
