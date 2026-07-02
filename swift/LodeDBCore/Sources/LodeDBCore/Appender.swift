import Foundation

/// One pre-embedded document to append (vector-in). `text` is an optional caption
/// (e.g. for an image), retained only when the appender was opened with
/// `storeText`; it is never embedded or chunked.
public struct LodeAppendDocument: Sendable, Equatable {
    public let id: String
    public let vector: [Float]
    public let metadata: [String: String]
    public let text: String?

    public init(id: String, vector: [Float], metadata: [String: String] = [:], text: String? = nil) {
        self.id = id
        self.vector = vector
        self.metadata = metadata
        self.text = text
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
///
/// On Windows the shared lock degrades to an exclusive hold, so appenders exclude
/// each other there: a second concurrent `open` waits for the first appender to
/// close, then fails. On Unix appenders coexist freely.
public final class LodeAppender {
    private let native: NativeAppender

    private init(native: NativeAppender) {
        self.native = native
    }

    /// Opens an appender over the store at `path`.
    ///
    /// `commitMode` must be `.wal` (the default); generation mode is rejected because
    /// it never replays the WAL, so appended records would be acknowledged yet never
    /// folded in. `durability` defaults to `.fsync` (each append is fsynced before
    /// returning, matching `LodeDB`); pass `.buffered` to trade power-loss durability
    /// for ingest throughput. `acquireWriterLock` takes the shared
    /// `<dir>/.lodedb.lock` so appenders exclude an exclusive writer (pass `false`
    /// only when an outer caller owns exclusion).
    ///
    /// `storeText`/`indexText` default to `false` (privacy: no raw text reaches the
    /// WAL). These are appender-specific defaults, so customizing another argument
    /// never turns retention on by accident. To retain appended captions, pass
    /// `storeText: true`, and only for a store whose writer also retains text, or the
    /// writer drops the caption at checkpoint.
    public static func open(
        at path: URL,
        commitMode: CommitMode = .wal,
        durability: Durability = .fsync,
        storeText: Bool = false,
        indexText: Bool = false,
        acquireWriterLock: Bool = true
    ) throws -> LodeAppender {
        guard commitMode == .wal else {
            throw LodeDBError.unsupported(
                "the appender requires WAL commit mode; generation mode does not replay the WAL")
        }
        let options = LodeStoreOptions(
            durability: durability,
            commitMode: .wal,
            storeText: storeText,
            indexText: indexText,
            acquireWriterLock: acquireWriterLock
        )
        let optionsJSON = try options.coreOpenOptionsJSON(path: path.path, readOnly: false)
        return LodeAppender(native: try NativeAppender.open(optionsJSON: optionsJSON))
    }

    /// Durably logs one vector-in record and returns its log sequence number.
    ///
    /// `text` is an optional caption retained only when the appender was opened with
    /// `storeText` (see `open`); it is never embedded or chunked.
    @discardableResult
    public func append(
        id: String,
        vector: [Float],
        metadata: [String: String] = [:],
        text: String? = nil
    ) throws -> UInt64 {
        try append([LodeAppendDocument(id: id, vector: vector, metadata: metadata, text: text)])
    }

    /// Durably logs one record covering `documents` and returns its log sequence
    /// number. Appending an empty array throws.
    @discardableResult
    public func append(_ documents: [LodeAppendDocument]) throws -> UInt64 {
        let payload = documents.map {
            AppenderVectorDocumentJSON(
                documentID: $0.id, vector: $0.vector, metadata: $0.metadata, text: $0.text)
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
    // The optional caption. The native appender retains it only under storeText;
    // otherwise it is dropped, so no raw text reaches the WAL.
    let text: String?

    enum CodingKeys: String, CodingKey {
        case documentID = "document_id"
        case vector
        case metadata
        case text
    }
}
