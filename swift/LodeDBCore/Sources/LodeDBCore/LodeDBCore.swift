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
/// `LodeDBCoreFFI` XCFramework). All ranking, chunking, tokenization, and scoring
/// run in the native engine; this type only marshals values across the C ABI.
///
/// The native `CoreEngine` keeps interior-mutable state (the lazily built TurboVec
/// index lives in a `RefCell`) and is not thread-safe, so every native call and
/// every mutation of local state is serialized behind `lock`. A single `LodeDB`
/// instance is therefore safe to share across threads, at the cost of serializing
/// concurrent operations (including the caller's embedding work inside `addText`).
public final class LodeDB {
    let vectorDimension: Int

    /// The native engine. Present for writable in-memory stores. `nil` only for a
    /// read-only snapshot opened with `openReadOnly`, which currently reports
    /// `count` from the manifest; native read-only open lands in a later phase.
    private let engine: NativeEngine?

    /// Document ids known to this store, the source for `count` until the native
    /// `stats()` call is exposed over the C ABI in the durable-storage phase.
    private var documentIDs: Set<String>

    /// Serializes access to the non-thread-safe native engine and to `documentIDs`.
    private let lock = NSLock()

    public init(vectorDimension: Int) throws {
        guard vectorDimension > 0 else {
            throw LodeDBError.invalidArgument("vectorDimension must be positive")
        }
        self.vectorDimension = vectorDimension
        self.engine = try NativeEngine(vectorDimension: vectorDimension)
        self.documentIDs = []
    }

    /// Backing initializer for the read-only manifest snapshot (`openReadOnly`).
    private init(snapshotVectorDimension: Int, documentIDs: Set<String>) {
        self.vectorDimension = snapshotVectorDimension
        self.engine = nil
        self.documentIDs = documentIDs
    }

    public var count: Int {
        locked { documentIDs.count }
    }

    public func addVector(_ vector: [Float], id: String, metadata: [String: String] = [:]) throws {
        try locked {
            let engine = try requireEngine()
            guard !id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw LodeDBError.invalidArgument("id is required")
            }
            guard vector.count == vectorDimension else {
                throw LodeDBError.invalidArgument("vector dimension does not match index")
            }
            let document = NativeVectorDocumentJSON(documentID: id, vector: vector, metadata: metadata, text: nil)
            try engine.upsertVectorsJSON(try encodeJSON([document]))
            documentIDs.insert(id)
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
            let engine = try requireEngine()
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
            documentIDs.insert(id)
        }
    }

    public func search(
        text: String,
        k: Int,
        mode: RetrievalMode = .vector,
        embedder: LodeEmbedder? = nil,
        filter: MetadataFilter = MetadataFilter()
    ) throws -> [SearchHit] {
        try locked {
            let engine = try requireEngine()
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
            let engine = try requireEngine()
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

    public func prepareTextUpsert(
        _ text: String,
        id: String,
        metadata: [String: String] = [:],
        chunkCharacterLimit: Int = 8192
    ) throws -> TextIngestPlan {
        try locked {
            let engine = try requireEngine()
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
            let engine = try requireEngine()
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
            documentIDs.insert(plan.id)
        }
    }

    /// Opens a persisted store read-only by reading its commit manifest.
    ///
    /// This is an interim manifest reader that reports `count`; the durable-storage
    /// phase replaces it with the native `open_readonly` path so snapshots can serve
    /// real native search.
    public static func openReadOnly(path: URL) throws -> LodeDB {
        let fileManager = FileManager.default
        let entries = try fileManager.contentsOfDirectory(at: path, includingPropertiesForKeys: nil)
        guard let commit = entries.first(where: { $0.lastPathComponent.hasSuffix(".commit.json") }) else {
            throw LodeDBError.notFound("commit manifest is missing")
        }
        let wrapper = try readJSONObject(commit)
        guard let body = wrapper["body"] as? [String: Any],
              let indexKey = body["index_key"] as? String,
              let baseEpoch = body["base_epoch"] as? Int else {
            throw LodeDBError.corruptStore("commit manifest body is malformed")
        }
        let generationPath = path
            .appendingPathComponent("\(indexKey).gen")
            .appendingPathComponent("g\(baseEpoch).json")
        let state = try readJSONObject(generationPath)
        let vectorDimension = state["native_dim"] as? Int ?? 1
        let hashes = state["document_hashes"] as? [String: Any] ?? [:]
        return LodeDB(
            snapshotVectorDimension: vectorDimension,
            documentIDs: Set(hashes.keys)
        )
    }

    private func locked<T>(_ body: () throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try body()
    }

    private func requireEngine() throws -> NativeEngine {
        guard let engine else {
            throw LodeDBError.unsupported(
                "operation requires a writable store; read-only snapshots gain native search in a later phase")
        }
        return engine
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

private func encodeJSON<T: Encodable>(_ value: T) throws -> String {
    let data = try JSONEncoder().encode(value)
    guard let text = String(data: data, encoding: .utf8) else {
        throw LodeDBError.invalidArgument("failed to encode JSON as UTF-8")
    }
    return text
}

private func decodeJSON<T: Decodable>(_ type: T.Type, from text: String) throws -> T {
    guard let data = text.data(using: .utf8) else {
        throw LodeDBError.internalError("native core returned JSON that is not valid UTF-8")
    }
    return try JSONDecoder().decode(type, from: data)
}

private func readJSONObject(_ url: URL) throws -> [String: Any] {
    let data = try Data(contentsOf: url)
    guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        throw LodeDBError.corruptStore("\(url.lastPathComponent) is not a JSON object")
    }
    return object
}
