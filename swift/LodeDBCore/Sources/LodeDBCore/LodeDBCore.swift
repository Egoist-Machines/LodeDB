import Foundation

public protocol LodeEmbedder {
    var dimension: Int { get }
    /// A stable, public model identity (e.g. `"BAAI/bge-base-en-v1.5"`). When non-nil,
    /// `LodeMemory(embedder:)` binds the store to it so a later reopen with a different
    /// same-dimension model is rejected. Defaults to nil (no identity binding).
    var modelIdentity: String? { get }
    func embed(texts: [String]) throws -> [[Float]]
    /// Embeds texts for a specific role. Embedders with asymmetric query/document
    /// prefixes (e.g. BGE) must honor this; the default ignores the role. `LodeDB`
    /// requests `.document` when ingesting and `.query` when searching.
    func embed(texts: [String], role: EmbeddingRole) throws -> [[Float]]
}

public extension LodeEmbedder {
    var modelIdentity: String? { nil }

    func embed(texts: [String], role: EmbeddingRole) throws -> [[Float]] {
        try embed(texts: texts)
    }
}

public enum RetrievalMode: String, Sendable {
    case vector
    case lexical
    case hybrid
}

/// A LodeDB store backed by the native Rust core (statically linked via the
/// `LodeDBCoreFFI` XCFramework). All ranking, chunking, tokenization, scoring, and
/// durable storage run in the native engine; this type marshals values across the
/// C ABI and serializes access.
///
/// The native `CoreEngine` keeps interior-mutable state and is not thread-safe, so
/// every native call is serialized behind `lock`. A single `LodeDB` instance is safe
/// to share across threads, at the cost of serializing concurrent operations
/// (including the caller's embedding work inside `addText`/`search`).
public final class LodeDB {
    public let vectorDimension: Int
    private let engine: NativeEngine
    private let lock = NSLock()
    /// Set by `close()`; once true, every operation other than `close()` throws.
    private var closed = false
    /// The store's text-retention policy, applied to `addText`/`prepareTextUpsert` so
    /// a `storeText: false` / `indexText: false` store does not retain or index text.
    private let storesText: Bool
    private let indexesText: Bool

    /// Creates an ephemeral in-memory store (nothing is read from or written to disk).
    ///
    /// Pass `modelIdentity` (e.g. an embedder's `modelIdentity`) to bind the index to
    /// a model so a later durable reopen can reject a different same-dimension model.
    public init(vectorDimension: Int, modelIdentity: String? = nil, ann: LodeAnnOptions? = nil) throws {
        guard vectorDimension > 0 else {
            throw LodeDBError.invalidArgument("vectorDimension must be positive")
        }
        self.vectorDimension = vectorDimension
        self.engine = try NativeEngine.inMemory(
            vectorDimension: vectorDimension, model: modelIdentity, ann: ann)
        self.storesText = true
        self.indexesText = true
    }

    /// Opens (or creates) a durable, on-disk store at `path`. If the store already
    /// holds an index, its vector dimension must match `vectorDimension`, and (when
    /// `modelIdentity` is given) its persisted model identity must match too.
    public init(
        path: URL,
        vectorDimension: Int,
        options: LodeStoreOptions = LodeStoreOptions(),
        modelIdentity: String? = nil,
        ann: LodeAnnOptions? = nil
    ) throws {
        guard vectorDimension > 0 else {
            throw LodeDBError.invalidArgument("vectorDimension must be positive")
        }
        guard options.chunkCharacterLimit > 0 else {
            throw LodeDBError.invalidArgument("chunkCharacterLimit must be positive")
        }
        let optionsJSON = try options.coreOpenOptionsJSON(path: path.path, readOnly: false)
        let engine = try NativeEngine.open(optionsJSON: optionsJSON)
        // Create the index on a fresh store, or verify the identity of an existing one.
        // `ann` is a create-time choice: on reopen the persisted config is used, so a
        // reopen ignores this argument (the existing index keeps how it was created).
        let existing = try decodeJSON([String].self, from: engine.indexIdsJSON())
        if existing.contains(engine.indexID) {
            let stats = CollectionStats(try decodeJSON(CoreEngineStatsJSON.self, from: engine.statsJSON()))
            try LodeDB.validate(stats: stats, vectorDimension: vectorDimension, modelIdentity: modelIdentity)
        } else {
            try engine.createIndex(vectorDimension: vectorDimension, model: modelIdentity, ann: ann)
        }
        self.vectorDimension = vectorDimension
        self.engine = engine
        self.storesText = options.storeText
        self.indexesText = options.indexText
    }

    private init(engine: NativeEngine, vectorDimension: Int) {
        self.engine = engine
        self.vectorDimension = vectorDimension
        // Read-only snapshots do not ingest, so the retention policy is unused.
        self.storesText = true
        self.indexesText = true
    }

    /// Opens a persisted store read-only (a lock-free generation snapshot). The
    /// snapshot reflects the last committed generation; call `refresh()` to overlay
    /// the current WAL tail for reader freshness and read-your-writes. When
    /// `modelIdentity` is given, the store's persisted model must match.
    public static func openReadOnly(
        path: URL,
        options: LodeStoreOptions = LodeStoreOptions(),
        modelIdentity: String? = nil
    ) throws -> LodeDB {
        let optionsJSON = try options.coreOpenOptionsJSON(path: path.path, readOnly: true)
        let engine = try NativeEngine.openReadOnly(optionsJSON: optionsJSON)
        let ids = try decodeJSON([String].self, from: engine.indexIdsJSON())
        guard let indexID = ids.contains("default") ? "default" : ids.first else {
            throw LodeDBError.notFound("store contains no index")
        }
        engine.indexID = indexID
        let stats = CollectionStats(try decodeJSON(CoreEngineStatsJSON.self, from: engine.statsJSON()))
        if let modelIdentity {
            try validate(stats: stats, vectorDimension: stats.vectorDimension, modelIdentity: modelIdentity)
        }
        return LodeDB(engine: engine, vectorDimension: stats.vectorDimension)
    }

    /// Validates a reopened index against the requested dimension and (optionally)
    /// model identity, so a same-dimension different-model store fails closed.
    private static func validate(stats: CollectionStats, vectorDimension: Int, modelIdentity: String?) throws {
        guard stats.vectorDimension == vectorDimension else {
            throw LodeDBError.invalidArgument(
                "existing index dimension \(stats.vectorDimension) does not match requested \(vectorDimension)")
        }
        if let modelIdentity, stats.model != modelIdentity {
            throw LodeDBError.invalidArgument(
                "store model '\(stats.model)' does not match expected model '\(modelIdentity)'")
        }
    }

    // MARK: - Stats / enumeration

    /// Document count for the collection. Returns 0 if stats are unavailable.
    public var count: Int {
        (try? stats().documentCount) ?? 0
    }

    public func stats() throws -> CollectionStats {
        try lockedOpen {
            CollectionStats(try decodeJSON(CoreEngineStatsJSON.self, from: engine.statsJSON()))
        }
    }

    /// The index ids loaded in the underlying engine (collection enumeration).
    public func collections() throws -> [String] {
        try lockedOpen { try decodeJSON([String].self, from: engine.indexIdsJSON()) }
    }

    // MARK: - Ingest

    public func addVector(_ vector: [Float], id: String, metadata: [String: String] = [:]) throws {
        try lockedOpen {
            guard !id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw LodeDBError.invalidArgument("id is required")
            }
            guard vector.count == vectorDimension else {
                throw LodeDBError.invalidArgument("vector dimension does not match index")
            }
            let document = NativeVectorDocumentJSON(documentID: id, vector: vector, metadata: metadata, text: nil)
            try engine.upsertVectorsJSON(try encodeJSON([document]))
        }
    }

    public func addText(
        _ text: String,
        id: String,
        metadata: [String: String] = [:],
        embedder: LodeEmbedder,
        chunkCharacterLimit: Int = 8192
    ) throws {
        try lockedOpen {
            guard embedder.dimension == vectorDimension else {
                throw LodeDBError.invalidArgument("embedder dimension does not match index")
            }
            guard chunkCharacterLimit > 0 else {
                throw LodeDBError.invalidArgument("chunkCharacterLimit must be positive")
            }
            let documentsJSON = try encodeJSON([
                NativeCoreDocumentJSON(documentID: id, text: text, metadata: metadata)
            ])
            let planJSON = try engine.prepareTextUpsertJSON(
                documentsJSON,
                storeText: storesText,
                indexText: indexesText,
                chunkCharacterLimit: chunkCharacterLimit
            )
            let plan = try decodeJSON(NativeIngestPlanJSON.self, from: planJSON)
            let embeddings = try embedder.embed(texts: plan.chunksToEmbed.map(\.text), role: .document)
            guard embeddings.allSatisfy({ $0.count == vectorDimension }) else {
                throw LodeDBError.invalidArgument("embedding dimension does not match index")
            }
            _ = try engine.applyTextUpsertJSON(
                planJSON: planJSON,
                embeddingsJSON: try encodeJSON(embeddings),
                embeddingTimeMS: 0
            )
        }
    }

    public func prepareTextUpsert(
        _ text: String,
        id: String,
        metadata: [String: String] = [:],
        chunkCharacterLimit: Int = 8192
    ) throws -> TextIngestPlan {
        try lockedOpen {
            guard !id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw LodeDBError.invalidArgument("id is required")
            }
            guard chunkCharacterLimit > 0 else {
                throw LodeDBError.invalidArgument("chunkCharacterLimit must be positive")
            }
            let documentsJSON = try encodeJSON([
                NativeCoreDocumentJSON(documentID: id, text: text, metadata: metadata)
            ])
            let planJSON = try engine.prepareTextUpsertJSON(
                documentsJSON,
                storeText: storesText,
                indexText: indexesText,
                chunkCharacterLimit: chunkCharacterLimit
            )
            let plan = try decodeJSON(NativeIngestPlanJSON.self, from: planJSON)
            guard let document = plan.documents.first(where: { $0.documentID == id }) else {
                throw LodeDBError.internalError("native core returned no document plan")
            }
            let chunks = document.chunks.map { chunk in
                TextChunk(documentID: id, chunkID: chunk.chunkID, text: chunk.text, tokens: chunk.tokens)
            }
            return TextIngestPlan(
                id: id,
                metadata: document.metadata,
                text: text,
                chunks: chunks,
                nativePlanJSON: planJSON
            )
        }
    }

    public func applyTextUpsert(_ plan: TextIngestPlan, embeddings: [[Float]]) throws {
        try lockedOpen {
            guard embeddings.count == plan.chunks.count else {
                throw LodeDBError.invalidArgument("embedding count does not match plan")
            }
            if let first = embeddings.first {
                guard first.count == vectorDimension else {
                    throw LodeDBError.invalidArgument("embedding dimension does not match index")
                }
            }
            _ = try engine.applyTextUpsertJSON(
                planJSON: plan.nativePlanJSON,
                embeddingsJSON: try encodeJSON(embeddings),
                embeddingTimeMS: 0
            )
        }
    }

    // MARK: - Search

    public func search(
        text: String,
        k: Int,
        mode: RetrievalMode = .vector,
        embedder: LodeEmbedder? = nil,
        filter: MetadataFilter = MetadataFilter()
    ) throws -> [SearchHit] {
        try lockedOpen {
            guard k > 0 else {
                throw LodeDBError.invalidArgument("k must be positive")
            }
            let queryPlanJSON = try engine.prepareQueryTextJSON(text, mode: mode.rawValue)
            let queryEmbeddingJSON: String?
            if mode == .vector || mode == .hybrid {
                guard let embedder else {
                    throw LodeDBError.invalidArgument("embedder is required for vector search")
                }
                guard embedder.dimension == vectorDimension else {
                    throw LodeDBError.invalidArgument("embedder dimension does not match index")
                }
                let embeddings = try embedder.embed(texts: [text], role: .query)
                guard let query = embeddings.first, query.count == vectorDimension else {
                    throw LodeDBError.invalidArgument("embedder returned an invalid query embedding")
                }
                queryEmbeddingJSON = try encodeJSON(query)
            } else {
                queryEmbeddingJSON = nil
            }
            let resultsJSON = try engine.searchEmbeddedTextJSON(
                queryPlanJSON: queryPlanJSON,
                queryEmbeddingJSON: queryEmbeddingJSON,
                k: k,
                filterJSON: filter.encodedJSON
            )
            return try decodeSearchHits(resultsJSON)
        }
    }

    public func search(vector: [Float], k: Int, filter: MetadataFilter = MetadataFilter()) throws -> [SearchHit] {
        try lockedOpen {
            guard vector.count == vectorDimension else {
                throw LodeDBError.invalidArgument("query dimension does not match index")
            }
            guard k > 0 else {
                throw LodeDBError.invalidArgument("k must be positive")
            }
            let resultsJSON = try engine.queryVectorJSON(vector, k: k, filterJSON: filter.encodedJSON)
            return try decodeSearchHits(resultsJSON)
        }
    }

    /// Batched vector search: one result list per query vector, in input order.
    public func searchMany(vectors: [[Float]], k: Int, filter: MetadataFilter = MetadataFilter()) throws -> [[SearchHit]] {
        try lockedOpen {
            guard k > 0 else {
                throw LodeDBError.invalidArgument("k must be positive")
            }
            guard vectors.allSatisfy({ $0.count == vectorDimension }) else {
                throw LodeDBError.invalidArgument("query dimension does not match index")
            }
            let json = try engine.queryVectorsBatchJSON(
                queriesJSON: try encodeJSON(vectors),
                k: k,
                filterJSON: filter.encodedJSON
            )
            return try decodeJSON([NativeSearchResultsJSON].self, from: json).map(\.searchHits)
        }
    }

    /// Batched text search: one result list per query text, in input order.
    public func searchMany(
        texts: [String],
        k: Int,
        mode: RetrievalMode = .vector,
        embedder: LodeEmbedder? = nil,
        filter: MetadataFilter = MetadataFilter()
    ) throws -> [[SearchHit]] {
        try lockedOpen {
            guard k > 0 else {
                throw LodeDBError.invalidArgument("k must be positive")
            }
            let plans = try texts.map { try engine.prepareQueryTextJSON($0, mode: mode.rawValue) }
            // Each plan is a JSON object; concatenating them is a valid JSON array.
            let plansJSON = "[" + plans.joined(separator: ",") + "]"
            let embeddingsJSON: String?
            if mode == .vector || mode == .hybrid {
                guard let embedder else {
                    throw LodeDBError.invalidArgument("embedder is required for vector search")
                }
                guard embedder.dimension == vectorDimension else {
                    throw LodeDBError.invalidArgument("embedder dimension does not match index")
                }
                let embeddings = try embedder.embed(texts: texts, role: .query)
                guard embeddings.allSatisfy({ $0.count == vectorDimension }) else {
                    throw LodeDBError.invalidArgument("embedder returned an invalid query embedding")
                }
                embeddingsJSON = try encodeJSON(embeddings)
            } else {
                embeddingsJSON = nil
            }
            let json = try engine.searchEmbeddedTextBatchJSON(
                queryPlansJSON: plansJSON,
                queryEmbeddingsJSON: embeddingsJSON,
                k: k,
                filterJSON: filter.encodedJSON
            )
            return try decodeJSON([NativeSearchResultsJSON].self, from: json).map(\.searchHits)
        }
    }

    // MARK: - CRUD / retrieval

    /// Deletes a document by id. Returns true if a document was removed.
    @discardableResult
    public func remove(_ id: String) throws -> Bool {
        try lockedOpen {
            let resultJSON = try engine.deleteDocumentsJSON(try encodeJSON([id]))
            let result = try decodeJSON(CoreMutationResultJSON.self, from: resultJSON)
            return result.documentsDeleted > 0
        }
    }

    /// Returns a document's retained text, or nil if absent or text was not stored.
    public func get(_ id: String) throws -> String? {
        try lockedOpen {
            let json = try engine.getDocumentTextJSON(documentID: id)
            if isJSONNull(json) { return nil }
            return try decodeJSON(String.self, from: json)
        }
    }

    /// Returns retained text for several documents (ids without stored text are omitted).
    public func getTexts(_ ids: [String]) throws -> [String: String] {
        try lockedOpen {
            try decodeJSON([String: String].self, from: engine.getDocumentTextsJSON(try encodeJSON(ids)))
        }
    }

    /// Returns a payload-free document record, or nil if the document does not exist.
    public func getDocument(_ id: String) throws -> DocumentRecord? {
        try lockedOpen {
            let json = try engine.getDocumentJSON(documentID: id)
            if isJSONNull(json) { return nil }
            return DocumentRecord(try decodeJSON(DocumentRecordJSON.self, from: json))
        }
    }

    /// Lists payload-free document records, optionally filtered, paged with an `after`
    /// id cursor, and capped at `limit`.
    public func listDocuments(
        filter: MetadataFilter = MetadataFilter(),
        after: String? = nil,
        limit: Int? = nil
    ) throws -> [DocumentRecord] {
        try lockedOpen {
            if let limit, limit < 0 {
                throw LodeDBError.invalidArgument("limit must be non-negative")
            }
            let json = try engine.listDocumentsJSON(filterJSON: filter.encodedJSON, after: after, limit: limit)
            return try decodeJSON([DocumentRecordJSON].self, from: json).map(DocumentRecord.init)
        }
    }

    /// Updates a document's metadata and/or retained text.
    public func updateDocument(id: String, metadata: [String: String]? = nil, text: TextUpdate = .unchanged) throws {
        try lockedOpen {
            // Nothing to change: skip the FFI call so we do not bump the generation
            // or append a WAL record for a no-op update.
            if metadata == nil, case .unchanged = text { return }
            let metadataJSON = try metadata.map { try encodeJSON($0) }
            let textJSON: String?
            switch text {
            case .unchanged: textJSON = nil
            case .clear: textJSON = "null"
            case .set(let value): textJSON = try encodeJSON(value)
            }
            _ = try engine.updateDocumentPayloadJSON(documentID: id, metadataJSON: metadataJSON, textJSON: textJSON)
        }
    }

    // MARK: - Durability

    /// Flushes pending writes to durable storage. No-op for in-memory stores.
    public func persist() throws {
        try lockedOpen { try engine.persist() }
    }

    /// Overlays the current write-ahead log tail into this handle's in-memory view
    /// without checkpointing.
    ///
    /// A `readOnly` handle loads the last committed generation on open and is
    /// otherwise a stable snapshot; call this to fold in records other processes
    /// appended since (e.g. via `LodeAppender`), and to reach read-your-writes for
    /// an appended LSN (see `appliedLSN()`). A no-op on a writable handle, which
    /// folds the WAL when it opens.
    public func refresh() throws {
        try lockedOpen { try engine.refresh() }
    }

    /// The highest log sequence number reflected in this handle's view. Compare it
    /// to the LSN a `LodeAppender` returned for read-your-writes: the appended
    /// record is visible here once `appliedLSN() >= that LSN`. On a read-only handle
    /// call `refresh()` first to fold the current WAL tail into the view.
    public func appliedLSN() throws -> UInt64 {
        try lockedOpen { try engine.appliedLSN() }
    }

    /// Closes the writable generation (a final checkpoint) and marks the handle
    /// closed: every subsequent operation throws `.unsupported`. Idempotent.
    ///
    /// Native `close()` drops the engine's persistence and writer lock, so without
    /// this guard the same instance would keep accepting writes into a detached
    /// in-memory copy that is never persisted (and silently lost).
    public func close() throws {
        lock.lock()
        defer { lock.unlock() }
        guard !closed else { return }
        try engine.close()
        closed = true
    }

    // MARK: - Helpers

    /// Locks, then runs `body` unless the store has been closed.
    private func lockedOpen<T>(_ body: () throws -> T) throws -> T {
        lock.lock()
        defer { lock.unlock() }
        guard !closed else { throw LodeDBError.unsupported("store is closed") }
        return try body()
    }

    private func decodeSearchHits(_ resultsJSON: String) throws -> [SearchHit] {
        try decodeJSON(NativeSearchResultsJSON.self, from: resultsJSON).searchHits
    }
}

public struct TextIngestPlan: Equatable, Sendable {
    public let id: String
    public let metadata: [String: String]
    public let text: String
    public let chunks: [TextChunk]
    /// The native `IngestPlan` JSON, carried so `applyTextUpsert` can hand the exact
    /// plan back to the core (the source of truth for chunk ids and ordering).
    let nativePlanJSON: String
}

public struct TextChunk: Equatable, Sendable {
    public let documentID: String
    public let chunkID: String
    public let text: String
    public let tokens: [String]
}

/// The `upsert_vectors`/append vector-document JSON shape. Shared by
/// `LodeDB.addVectors` and `LodeAppender`, whose records must be byte-identical so
/// an appended record replays exactly like a writer-authored one. The optional
/// caption is retained by the native core only under `storeText`.
struct NativeVectorDocumentJSON: Encodable {
    let documentID: String
    let vector: [Float]
    let metadata: [String: String]
    let text: String?

    enum CodingKeys: String, CodingKey {
        case documentID = "document_id"
        case vector
        case metadata
        case text
    }
}

// Internal (not private): also encoded/decoded by `LodeAppender`'s text-append path
// in Appender.swift, which reuses the exact writer shapes.
struct NativeCoreDocumentJSON: Encodable {
    let documentID: String
    let text: String
    let metadata: [String: String]

    enum CodingKeys: String, CodingKey {
        case documentID = "document_id"
        case text
        case metadata
    }
}

struct NativeIngestPlanJSON: Decodable {
    let documents: [NativePlanDocumentJSON]
    let chunksToEmbed: [NativePlanEmbeddingChunkJSON]

    enum CodingKeys: String, CodingKey {
        case documents
        case chunksToEmbed = "chunks_to_embed"
    }
}

struct NativePlanDocumentJSON: Decodable {
    let documentID: String
    let metadata: [String: String]
    let text: String?
    let chunks: [NativePlanDocumentChunkJSON]

    enum CodingKeys: String, CodingKey {
        case documentID = "document_id"
        case metadata
        case text
        case chunks
    }
}

struct NativePlanDocumentChunkJSON: Decodable {
    let chunkID: String
    let text: String
    let tokens: [String]

    enum CodingKeys: String, CodingKey {
        case chunkID = "chunk_id"
        case text
        case tokens
    }
}

struct NativePlanEmbeddingChunkJSON: Decodable {
    let text: String
}

struct NativeSearchResultsJSON: Decodable {
    let hits: [NativeSearchHitJSON]

    var searchHits: [SearchHit] {
        hits.map { SearchHit(id: $0.documentID, chunkID: $0.chunkID, score: $0.score, metadata: $0.metadata) }
    }
}

struct NativeSearchHitJSON: Decodable {
    let documentID: String
    let chunkID: String
    let score: Float
    let metadata: [String: String]

    enum CodingKeys: String, CodingKey {
        case documentID = "document_id"
        case chunkID = "chunk_id"
        case score
        case metadata
    }
}

/// True when the native core returned a bare JSON `null` (an absent `Option`).
private func isJSONNull(_ json: String) -> Bool {
    json.trimmingCharacters(in: .whitespacesAndNewlines) == "null"
}

func encodeJSON<T: Encodable>(_ value: T) throws -> String {
    let data = try JSONEncoder().encode(value)
    guard let text = String(data: data, encoding: .utf8) else {
        throw LodeDBError.invalidArgument("failed to encode JSON as UTF-8")
    }
    return text
}

func decodeJSON<T: Decodable>(_ type: T.Type, from text: String) throws -> T {
    guard let data = text.data(using: .utf8) else {
        throw LodeDBError.internalError("native core returned JSON that is not valid UTF-8")
    }
    return try JSONDecoder().decode(type, from: data)
}
