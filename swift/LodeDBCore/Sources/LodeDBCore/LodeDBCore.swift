import Foundation

public protocol LodeEmbedder {
    var dimension: Int { get }
    func embed(texts: [String]) throws -> [[Float]]
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

    /// Creates an ephemeral in-memory store (nothing is read from or written to disk).
    public init(vectorDimension: Int) throws {
        guard vectorDimension > 0 else {
            throw LodeDBError.invalidArgument("vectorDimension must be positive")
        }
        self.vectorDimension = vectorDimension
        self.engine = try NativeEngine.inMemory(vectorDimension: vectorDimension)
    }

    /// Opens (or creates) a durable, on-disk store at `path`. If the store already
    /// holds an index, its vector dimension must match `vectorDimension`.
    public init(path: URL, vectorDimension: Int, options: LodeStoreOptions = LodeStoreOptions()) throws {
        guard vectorDimension > 0 else {
            throw LodeDBError.invalidArgument("vectorDimension must be positive")
        }
        let optionsJSON = try options.coreOpenOptionsJSON(path: path.path, readOnly: false)
        let engine = try NativeEngine.open(optionsJSON: optionsJSON)
        // Create the index on a fresh store, or verify the dimension of an existing one.
        let existing = try decodeJSON([String].self, from: engine.indexIdsJSON())
        if existing.contains(engine.indexID) {
            let stats = CollectionStats(try decodeJSON(CoreEngineStatsJSON.self, from: engine.statsJSON()))
            guard stats.vectorDimension == vectorDimension else {
                throw LodeDBError.invalidArgument(
                    "existing index dimension \(stats.vectorDimension) does not match requested \(vectorDimension)")
            }
        } else {
            try engine.createIndex(vectorDimension: vectorDimension)
        }
        self.vectorDimension = vectorDimension
        self.engine = engine
    }

    private init(engine: NativeEngine, vectorDimension: Int) {
        self.engine = engine
        self.vectorDimension = vectorDimension
    }

    /// Opens a persisted store read-only (a lock-free generation snapshot). WAL tails
    /// are ignored; the snapshot reflects the last committed generation.
    public static func openReadOnly(path: URL, options: LodeStoreOptions = LodeStoreOptions()) throws -> LodeDB {
        let optionsJSON = try options.coreOpenOptionsJSON(path: path.path, readOnly: true)
        let engine = try NativeEngine.openReadOnly(optionsJSON: optionsJSON)
        let ids = try decodeJSON([String].self, from: engine.indexIdsJSON())
        guard let indexID = ids.contains("default") ? "default" : ids.first else {
            throw LodeDBError.notFound("store contains no index")
        }
        engine.indexID = indexID
        let stats = CollectionStats(try decodeJSON(CoreEngineStatsJSON.self, from: engine.statsJSON()))
        return LodeDB(engine: engine, vectorDimension: stats.vectorDimension)
    }

    // MARK: - Stats / enumeration

    /// Document count for the collection. Returns 0 if stats are unavailable.
    public var count: Int {
        (try? stats().documentCount) ?? 0
    }

    public func stats() throws -> CollectionStats {
        try locked {
            CollectionStats(try decodeJSON(CoreEngineStatsJSON.self, from: engine.statsJSON()))
        }
    }

    /// The index ids loaded in the underlying engine (collection enumeration).
    public func collections() throws -> [String] {
        try locked { try decodeJSON([String].self, from: engine.indexIdsJSON()) }
    }

    // MARK: - Ingest

    public func addVector(_ vector: [Float], id: String, metadata: [String: String] = [:]) throws {
        try locked {
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
        try locked {
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
                storeText: true,
                indexText: true,
                chunkCharacterLimit: chunkCharacterLimit
            )
            let plan = try decodeJSON(NativeIngestPlanJSON.self, from: planJSON)
            let embeddings = try embedder.embed(texts: plan.chunksToEmbed.map(\.text))
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
        try locked {
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
                storeText: true,
                indexText: true,
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
        try locked {
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
        try locked {
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
                let embeddings = try embedder.embed(texts: [text])
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
        try locked {
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

    // MARK: - CRUD / retrieval

    /// Deletes a document by id. Returns true if a document was removed.
    @discardableResult
    public func remove(_ id: String) throws -> Bool {
        try locked {
            let resultJSON = try engine.deleteDocumentsJSON(try encodeJSON([id]))
            let result = try decodeJSON(CoreMutationResultJSON.self, from: resultJSON)
            return result.documentsDeleted > 0
        }
    }

    /// Returns a document's retained text, or nil if absent or text was not stored.
    public func get(_ id: String) throws -> String? {
        try locked {
            let json = try engine.getDocumentTextJSON(documentID: id)
            if isJSONNull(json) { return nil }
            return try decodeJSON(String.self, from: json)
        }
    }

    /// Returns retained text for several documents (ids without stored text are omitted).
    public func getTexts(_ ids: [String]) throws -> [String: String] {
        try locked {
            try decodeJSON([String: String].self, from: engine.getDocumentTextsJSON(try encodeJSON(ids)))
        }
    }

    /// Returns a payload-free document record, or nil if the document does not exist.
    public func getDocument(_ id: String) throws -> DocumentRecord? {
        try locked {
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
        try locked {
            let json = try engine.listDocumentsJSON(filterJSON: filter.encodedJSON, after: after, limit: limit)
            return try decodeJSON([DocumentRecordJSON].self, from: json).map(DocumentRecord.init)
        }
    }

    /// Updates a document's metadata and/or retained text.
    public func updateDocument(id: String, metadata: [String: String]? = nil, text: TextUpdate = .unchanged) throws {
        try locked {
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
        try locked { try engine.persist() }
    }

    /// Closes the writable generation (a final checkpoint). No-op for read-only / in-memory.
    public func close() throws {
        try locked { try engine.close() }
    }

    // MARK: - Helpers

    private func locked<T>(_ body: () throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try body()
    }

    private func decodeSearchHits(_ resultsJSON: String) throws -> [SearchHit] {
        let results = try decodeJSON(NativeSearchResultsJSON.self, from: resultsJSON)
        return results.hits.map { hit in
            SearchHit(id: hit.documentID, chunkID: hit.chunkID, score: hit.score, metadata: hit.metadata)
        }
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

private struct NativeVectorDocumentJSON: Encodable {
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

private struct NativeCoreDocumentJSON: Encodable {
    let documentID: String
    let text: String
    let metadata: [String: String]

    enum CodingKeys: String, CodingKey {
        case documentID = "document_id"
        case text
        case metadata
    }
}

private struct NativeIngestPlanJSON: Decodable {
    let documents: [NativePlanDocumentJSON]
    let chunksToEmbed: [NativePlanEmbeddingChunkJSON]

    enum CodingKeys: String, CodingKey {
        case documents
        case chunksToEmbed = "chunks_to_embed"
    }
}

private struct NativePlanDocumentJSON: Decodable {
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

private struct NativePlanDocumentChunkJSON: Decodable {
    let chunkID: String
    let text: String
    let tokens: [String]

    enum CodingKeys: String, CodingKey {
        case chunkID = "chunk_id"
        case text
        case tokens
    }
}

private struct NativePlanEmbeddingChunkJSON: Decodable {
    let text: String
}

private struct NativeSearchResultsJSON: Decodable {
    let hits: [NativeSearchHitJSON]
}

private struct NativeSearchHitJSON: Decodable {
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
