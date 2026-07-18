//! Error type for OreCloud's artifact-store layer.
//!
//! Mirrors the failure surface the Python Milestone-1 modules expressed with
//! stdlib exceptions (`FileNotFoundError`, `RuntimeError`), mapped onto the
//! idiomatic Rust `Result`. The engine's own commit-manifest errors
//! (`lodedb_core::CoreError`) are wrapped rather than flattened, so a corrupt
//! root pointer keeps its diagnostic on the way up.

use lodedb_core::CoreError;
use std::fmt::{Display, Formatter};

/// The result type for every fallible artifact-store operation.
pub type Result<T> = std::result::Result<T, ArtifactStoreError>;

/// A failure raised by an [`ArtifactStore`](crate::ArtifactStore) operation.
#[derive(Debug)]
pub enum ArtifactStoreError {
    /// A requested artifact name is not present in the store (the analogue of
    /// the Python `FileNotFoundError`). Callers such as `export_generation`
    /// surface "no committed generation" as this variant.
    NotFound(String),
    /// A root-pointer compare-and-swap precondition failed: the committed
    /// generation was not the one the caller expected. Kept distinct because it
    /// is the *retryable* failure — a concurrent writer advanced the pointer.
    PointerConflict {
        key: String,
        expected: Option<u64>,
        found: Option<u64>,
    },
    /// The bytes or path do not match what the committed manifest promises: a
    /// checksum mismatch, an attempt to overwrite an immutable artifact with
    /// different bytes, a sub-manifest in an unsupported legacy layout, or a
    /// name/key that escapes the store root. All mean "do not trust these
    /// bytes / this path", so they collapse to one hard-failure category.
    Integrity(String),
    /// The engine's commit-manifest layer rejected a root-pointer read or write
    /// (schema-version or body-checksum failure).
    Core(CoreError),
    /// An underlying filesystem operation failed.
    Io(std::io::Error),
    /// A sync refused to transfer because the three-pointer classification
    /// requires an explicit force: the two ends diverged past the recorded
    /// base, or there is no trustworthy base at all. `classification` is the
    /// stable lowercase name (`diverged`/`unknown`); `hint` names the flag
    /// that overrides it. Distinct from [`PointerConflict`](Self::PointerConflict):
    /// that is a *race* (retryable), this is a *decision* only the caller can
    /// make.
    SyncConflict {
        classification: String,
        hint: String,
    },
    /// A restore refused to replace the destination pointer because the
    /// destination's WAL still holds acknowledged-but-uncheckpointed
    /// operations. Replaying those records onto a pulled lineage (or silently
    /// dropping them) would corrupt or lose acknowledged writes, so the caller
    /// must checkpoint the store first — or explicitly discard the records
    /// with a force-pull. Like [`SyncConflict`](Self::SyncConflict), this is a
    /// *decision*, not a race.
    PendingWal { ops: usize, hint: String },
    /// A storage backend (e.g. an object store) failed for a reason that is not a
    /// missing object or a pointer precondition — a network error, a permission
    /// denial, or any other transport-level failure. The message carries the
    /// backend's own diagnostic. Missing objects map to [`NotFound`](Self::NotFound)
    /// and failed conditional writes to [`PointerConflict`](Self::PointerConflict),
    /// so callers can still branch on those without inspecting this string.
    Backend(String),
}

impl Display for ArtifactStoreError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(name) => write!(formatter, "artifact {name:?} not found"),
            Self::PointerConflict {
                key,
                expected,
                found,
            } => write!(
                formatter,
                "pointer {key:?} compare-and-swap failed: expected generation {expected:?}, \
                 found {found:?}"
            ),
            Self::SyncConflict {
                classification,
                hint,
            } => write!(
                formatter,
                "sync refused: local and remote are {classification}; {hint}"
            ),
            Self::PendingWal { ops, hint } => write!(
                formatter,
                "the destination database holds {ops} uncheckpointed WAL operation(s); {hint}"
            ),
            Self::Integrity(message) => write!(formatter, "{message}"),
            Self::Core(error) => write!(formatter, "{error}"),
            Self::Io(error) => write!(formatter, "{error}"),
            Self::Backend(message) => write!(formatter, "{message}"),
        }
    }
}

impl std::error::Error for ArtifactStoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Core(error) => Some(error),
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<CoreError> for ArtifactStoreError {
    /// Lets `read_commit_manifest`/`write_commit_manifest` results propagate with
    /// `?`, preserving the engine's structured error.
    fn from(error: CoreError) -> Self {
        Self::Core(error)
    }
}

impl From<std::io::Error> for ArtifactStoreError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}
