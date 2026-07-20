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
/// folded in, hence `open` rejects generation-mode options outright. Like
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
    /// The appender always uses WAL commit mode: generation mode never replays the
    /// WAL, so an appended record would be acknowledged yet never folded in.
    /// `durability` defaults to `.fsync` (each append is fsynced before returning,
    /// matching `LodeDB`); pass `.buffered` to trade power-loss durability for
    /// ingest throughput. `acquireWriterLock` takes the shared `<dir>/.lodedb.lock`
    /// so appenders exclude an exclusive writer (pass `false` only when an outer
    /// caller owns exclusion).
    ///
    /// `storeText`/`indexText` default to `false` (privacy: no raw text reaches the
    /// WAL). These are appender-specific defaults, so customizing another argument
    /// never turns retention on by accident. To retain appended captions, pass
    /// `storeText: true`, and only for a store whose writer also retains text, or the
    /// writer drops the caption at checkpoint. `chunkCharacterLimit` (used only by
    /// `append(text:...)`) must match the store writer's (`LodeDB.addText`'s default
    /// is 8192) so appended text chunks identically.
    public static func open(
        at path: URL,
        durability: Durability = .fsync,
        storeText: Bool = false,
        indexText: Bool = false,
        acquireWriterLock: Bool = true,
        chunkCharacterLimit: Int = 8192
    ) throws -> LodeAppender {
        let options = LodeStoreOptions(
            durability: durability,
            commitMode: .wal,
            storeText: storeText,
            indexText: indexText,
            chunkCharacterLimit: chunkCharacterLimit,
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
        // Reuse the writer's vector-document shape so an appended record is
        // byte-identical to a `LodeDB.addVectors` one and replays the same way.
        let payload = documents.map {
            NativeVectorDocumentJSON(
                documentID: $0.id, vector: $0.vector, metadata: $0.metadata, text: $0.text)
        }
        return try native.appendVectorsJSON(try encodeJSON(payload))
    }

    /// Durably logs a delete of `ids` and returns its log sequence number.
    @discardableResult
    public func delete(ids: [String]) throws -> UInt64 {
        try native.appendDeletesJSON(try encodeJSON(ids))
    }

    /// Chunks and embeds `text` (via `embedder`), logs one post-embedding text record,
    /// and returns its log sequence number.
    ///
    /// The document is chunked by the native core exactly as `LodeDB.addText` chunks
    /// it (at the appender's `chunkCharacterLimit`), each chunk is embedded, and the
    /// record is logged for the next writable open to fold. Whether the raw text and
    /// lexical tokens are retained follows the `storeText`/`indexText` the appender was
    /// opened with (match the store's writer). The vector-in `append` never embeds and
    /// needs no embedder. `id` is required (auto-ids would collide across writers).
    @discardableResult
    public func append(
        text: String,
        id: String,
        metadata: [String: String] = [:],
        embedder: LodeEmbedder
    ) throws -> UInt64 {
        guard !id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw LodeDBError.invalidArgument("id is required")
        }
        let documentsJSON = try encodeJSON([
            NativeCoreDocumentJSON(documentID: id, text: text, metadata: metadata)
        ])
        let planJSON = try native.prepareDocumentsJSON(documentsJSON)
        let plan = try decodeJSON(NativeIngestPlanJSON.self, from: planJSON)
        // The native core validates the embedding dimension against the index; embed
        // the chunks it asked for and log the post-embedding record.
        let embeddings = try embedder.embed(texts: plan.chunksToEmbed.map(\.text), role: .document)
        return try native.appendEmbeddedDocumentsJSON(
            planJSON: planJSON,
            embeddingsJSON: try encodeJSON(embeddings)
        )
    }
}
