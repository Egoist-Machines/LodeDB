//! Stable error codes for native-core bindings.

use std::error::Error;
use std::fmt::{Display, Formatter};

/// Stable native-core error categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// Native-core error with a stable code and redacted message.
#[derive(Debug, Clone, PartialEq, Eq)]
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
        assert_eq!(error.to_string(), "Unsupported: native core is disabled");
    }
}
