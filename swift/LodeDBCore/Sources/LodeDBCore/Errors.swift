import Foundation

/// Errors surfaced by the native core, mapped from the C ABI `LodeStatus` codes.
public enum LodeDBError: Error, Equatable {
    /// `LODE_INVALID_ARGUMENT` (1)
    case invalidArgument(String)
    /// `LODE_NOT_FOUND` (2)
    case notFound(String)
    /// `LODE_CORRUPT_STORE` (3)
    case corruptStore(String)
    /// `LODE_PLAN_STALE` (4): an ingest/query plan was applied against a changed index.
    case planStale(String)
    /// `LODE_UNSUPPORTED` (5): the operation is not available in this build/configuration.
    case unsupported(String)
    /// `LODE_INTERNAL` (255) or an unrecognized status code.
    case internalError(String)
}
