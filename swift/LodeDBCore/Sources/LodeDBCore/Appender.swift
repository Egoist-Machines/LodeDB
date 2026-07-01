import Foundation

/// One pre-embedded document to append (vector-in). The appender logs the vector
/// and metadata only; raw text is never written to the WAL.
public struct LodeAppendDocument: Sendable, Equatable {
    public let id: String
    public let vector: [Float]
    public let metadata: [String: String]

    public init(id: String, vector: [Float], metadata: [String: String] = [:]) {
        self.id = id
        self.vector = vector
        self.metadata = metadata
    }
}

/// A shared-lock appender over a persisted store's single index.
///
/// Many processes can each open an appender at once and durably log vector-in
/// records to the store's WAL concurrently; the next exclusive writer (a `LodeDB`
/// open) folds them into the index. The store must hold exactly one index and be
/// operated in WAL commit mode: the appender logs to the WAL, and only a WAL-mode
/// writer replays it. A writer that opens the store in generation commit mode
/// never replays the WAL, so records appended here would be acknowledged yet never
/// folded in — hence `open` rejects generation-mode options outright. Like
/// `LodeDB`, a single instance is not thread-safe; serialize calls to it.
public final class LodeAppender {
    private let native: NativeAppender

    private init(native: NativeAppender) {
        self.native = native
    }

    /// Opens an appender over the store at `path` with `options`.
    ///
    /// `options.commitMode` must be `.wal` (the default); generation mode is
    /// rejected because it never replays the WAL, so appended records would be
    /// acknowledged yet never folded in. `options.durability` controls whether each
    /// append fsyncs before returning, and `options.acquireWriterLock` takes the
    /// shared `<dir>/.lodedb.lock` so appenders exclude an exclusive writer (pass a
    /// value with it `false` only when an outer caller owns exclusion). The
    /// retained-text options are inert here — the appender logs vector plus metadata
    /// only, never raw text.
    public static func open(
        at path: URL,
        options: LodeStoreOptions = LodeStoreOptions()
    ) throws -> LodeAppender {
        guard options.commitMode == .wal else {
            throw LodeDBError.unsupported(
                "the appender requires WAL commit mode; generation mode does not replay the WAL")
        }
        let optionsJSON = try options.coreOpenOptionsJSON(path: path.path, readOnly: false)
        return LodeAppender(native: try NativeAppender.open(optionsJSON: optionsJSON))
    }

    /// Durably logs one vector-in record and returns its log sequence number.
    @discardableResult
    public func append(id: String, vector: [Float], metadata: [String: String] = [:]) throws -> UInt64 {
        try append([LodeAppendDocument(id: id, vector: vector, metadata: metadata)])
    }

    /// Durably logs one record covering `documents` and returns its log sequence
    /// number. Appending an empty array throws.
    @discardableResult
    public func append(_ documents: [LodeAppendDocument]) throws -> UInt64 {
        let payload = documents.map {
            AppenderVectorDocumentJSON(documentID: $0.id, vector: $0.vector, metadata: $0.metadata)
        }
        return try native.appendVectorsJSON(try encodeJSON(payload))
    }

    /// Durably logs a delete of `ids` and returns its log sequence number.
    @discardableResult
    public func delete(ids: [String]) throws -> UInt64 {
        try native.appendDeletesJSON(try encodeJSON(ids))
    }
}

private struct AppenderVectorDocumentJSON: Encodable {
    let documentID: String
    let vector: [Float]
    let metadata: [String: String]
    // Always null: the appender logs vector plus metadata only, never raw text.
    let text: String? = nil

    enum CodingKeys: String, CodingKey {
        case documentID = "document_id"
        case vector
        case metadata
        case text
    }
}
