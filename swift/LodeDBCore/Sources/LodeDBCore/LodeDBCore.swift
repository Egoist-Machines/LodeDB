import Foundation
import CryptoKit

public protocol LodeEmbedder {
    var dimension: Int { get }
    func embed(texts: [String]) throws -> [[Float]]
}

public enum RetrievalMode: String, Sendable {
    case vector
    case lexical
    case hybrid
}

public final class LodeDB {
    private let vectorDimension: Int
    private let nativeTextCore: NativeTextCore?
    private var nativeVectorSearchComplete: Bool
    private var documents: [String: VectorDocument]

    public init(vectorDimension: Int) throws {
        guard vectorDimension > 0 else {
            throw LodeDBError.invalidArgument("vectorDimension must be positive")
        }
        self.vectorDimension = vectorDimension
        let nativeTextCore = try Self.nativeCoreFromEnvironment(vectorDimension: vectorDimension)
        self.nativeTextCore = nativeTextCore
        self.nativeVectorSearchComplete = nativeTextCore != nil
        self.documents = [:]
    }

    private init(vectorDimension: Int, documents: [String: VectorDocument]) {
        self.vectorDimension = vectorDimension
        self.nativeTextCore = nil
        self.nativeVectorSearchComplete = false
        self.documents = documents
    }

    var nativeCoreEnabled: Bool {
        nativeTextCore != nil
    }

    var nativeVectorSearchReady: Bool {
        nativeTextCore != nil && nativeVectorSearchComplete
    }

    public var count: Int {
        documents.count
    }

    public func addVector(_ vector: [Float], id: String, metadata: [String: String] = [:]) throws {
        guard !id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw LodeDBError.invalidArgument("id is required")
        }
        guard vector.count == vectorDimension else {
            throw LodeDBError.invalidArgument("vector dimension does not match index")
        }
        if nativeTextCore != nil {
            nativeVectorSearchComplete = false
        }
        documents[id] = VectorDocument(vector: vector, metadata: metadata)
    }

    public func addText(
        _ text: String,
        id: String,
        metadata: [String: String] = [:],
        embedder: LodeEmbedder,
        chunkCharacterLimit: Int = 8192
    ) throws {
        guard embedder.dimension == vectorDimension else {
            throw LodeDBError.invalidArgument("embedder dimension does not match index")
        }
        if let nativeTextCore {
            try addTextWithNativeCore(
                nativeTextCore,
                text: text,
                id: id,
                metadata: metadata,
                embedder: embedder,
                chunkCharacterLimit: chunkCharacterLimit
            )
            return
        }
        let plan = try prepareTextUpsert(
            text,
            id: id,
            metadata: metadata,
            chunkCharacterLimit: chunkCharacterLimit
        )
        let embeddings = try embedder.embed(texts: plan.chunks.map(\.text))
        try applyTextUpsert(plan, embeddings: embeddings)
    }

    public func search(
        text: String,
        k: Int,
        mode: RetrievalMode = .vector,
        embedder: LodeEmbedder? = nil,
        filter: MetadataFilter = MetadataFilter()
    ) throws -> [SearchHit] {
        if let nativeTextCore, nativeVectorSearchComplete {
            return try searchTextWithNativeCore(
                nativeTextCore,
                text: text,
                k: k,
                mode: mode,
                embedder: embedder,
                filter: filter
            )
        }
        let queryTokens = tokenize(text)
        let vectorHits: [SearchHit]
        if mode == .vector || mode == .hybrid {
            guard let embedder else {
                throw LodeDBError.invalidArgument("embedder is required for vector search")
            }
            guard embedder.dimension == vectorDimension else {
                throw LodeDBError.invalidArgument("embedder dimension does not match index")
            }
            let embeddings = try embedder.embed(texts: [text])
            guard let query = embeddings.first else {
                throw LodeDBError.invalidArgument("embedder returned no query embedding")
            }
            vectorHits = try search(vector: query, k: k, filter: filter)
        } else {
            vectorHits = []
        }
        if mode == .vector {
            return vectorHits
        }
        let lexicalHits = lexicalSearch(tokens: queryTokens, k: k, filter: filter)
        if mode == .lexical {
            return lexicalHits
        }
        return reciprocalRankFusion(vectorHits, lexicalHits, k: k)
    }

    public func search(vector: [Float], k: Int, filter: MetadataFilter = MetadataFilter()) throws -> [SearchHit] {
        guard vector.count == vectorDimension else {
            throw LodeDBError.invalidArgument("query dimension does not match index")
        }
        guard k > 0 else {
            throw LodeDBError.invalidArgument("k must be positive")
        }
        if let nativeTextCore, nativeVectorSearchComplete, filter.isEmpty {
            return try nativeTextCore.queryVector(vector, k: k).map { hit in
                SearchHit(
                    id: hit.id,
                    chunkID: hit.chunkID,
                    score: hit.score,
                    metadata: documents[hit.id]?.metadata ?? [:]
                )
            }
        }
        return documents
            .filter { filter.matches($0.value.metadata) }
            .map { id, document in
                SearchHit(
                    id: id,
                    chunkID: document.chunkID.isEmpty ? nil : document.chunkID,
                    score: dot(vector, document.vector),
                    metadata: document.metadata
                )
            }
            .sorted { left, right in
                if left.score == right.score {
                    return left.id < right.id
                }
                return left.score > right.score
            }
            .prefix(k)
            .map { $0 }
    }

    public func prepareTextUpsert(
        _ text: String,
        id: String,
        metadata: [String: String] = [:],
        chunkCharacterLimit: Int = 8192
    ) throws -> TextIngestPlan {
        guard !id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw LodeDBError.invalidArgument("id is required")
        }
        guard chunkCharacterLimit > 0 else {
            throw LodeDBError.invalidArgument("chunkCharacterLimit must be positive")
        }
        let chunks = chunkText(text, limit: chunkCharacterLimit).enumerated().map { offset, chunk in
            let hash = normalizedSHA256(chunk)
            return TextChunk(
                documentID: id,
                chunkID: "\(id):\(hash.prefix(12)):\(String(format: "%04d", offset))",
                text: chunk,
                tokens: tokenize(chunk)
            )
        }
        return TextIngestPlan(id: id, metadata: metadata, text: text, chunks: chunks)
    }

    public func applyTextUpsert(_ plan: TextIngestPlan, embeddings: [[Float]]) throws {
        guard embeddings.count == plan.chunks.count else {
            throw LodeDBError.invalidArgument("embedding count does not match plan")
        }
        guard let first = embeddings.first else {
            throw LodeDBError.invalidArgument("text produced no chunks")
        }
        guard first.count == vectorDimension else {
            throw LodeDBError.invalidArgument("embedding dimension does not match index")
        }
        if nativeTextCore != nil {
            nativeVectorSearchComplete = false
        }
        documents[plan.id] = VectorDocument(
            vector: first,
            metadata: plan.metadata,
            text: plan.text,
            chunkID: plan.chunks.first?.chunkID ?? plan.id,
            tokens: plan.chunks.flatMap(\.tokens)
        )
    }

    private func lexicalSearch(tokens: [String], k: Int, filter: MetadataFilter) -> [SearchHit] {
        let query = Set(tokens)
        return documents
            .filter { filter.matches($0.value.metadata) }
            .compactMap { id, document -> SearchHit? in
                let overlap = document.tokens.filter { query.contains($0) }.count
                guard overlap > 0 else {
                    return nil
                }
                return SearchHit(
                    id: id,
                    chunkID: document.chunkID.isEmpty ? nil : document.chunkID,
                    score: Float(overlap),
                    metadata: document.metadata
                )
            }
            .sorted { left, right in
                if left.score == right.score {
                    return left.id < right.id
                }
                return left.score > right.score
            }
            .prefix(k)
            .map { $0 }
    }

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
        let metadata = state["document_metadata"] as? [String: Any] ?? [:]
        var documents: [String: VectorDocument] = [:]
        for documentID in hashes.keys {
            documents[documentID] = VectorDocument(
                vector: Array(repeating: 0, count: vectorDimension),
                metadata: normalizeMetadata(metadata[documentID])
            )
        }
        return LodeDB(vectorDimension: vectorDimension, documents: documents)
    }

    private static func nativeCoreFromEnvironment(vectorDimension: Int) throws -> NativeTextCore? {
        guard let dylib = ProcessInfo.processInfo.environment["LODEDB_FFI_DYLIB"] else {
            return nil
        }
        let library = try NativeCoreLibrary(path: dylib)
        return try NativeTextCore(library: library, vectorDimension: vectorDimension)
    }

    private func addTextWithNativeCore(
        _ nativeTextCore: NativeTextCore,
        text: String,
        id: String,
        metadata: [String: String],
        embedder: LodeEmbedder,
        chunkCharacterLimit: Int
    ) throws {
        guard chunkCharacterLimit > 0 else {
            throw LodeDBError.invalidArgument("chunkCharacterLimit must be positive")
        }
        let documentsJSON = try encodeJSON([
            NativeCoreDocumentJSON(documentID: id, text: text, metadata: metadata)
        ])
        let planJSON = try nativeTextCore.prepareTextUpsertJSON(
            documentsJSON,
            storeText: true,
            indexText: true,
            chunkCharacterLimit: chunkCharacterLimit
        )
        let plan = try decodeJSON(NativeIngestPlanJSON.self, from: planJSON)
        let chunkTexts = plan.chunksToEmbed.map(\.text)
        let embeddingStarted = Date()
        let embeddings = try embedder.embed(texts: chunkTexts)
        let embeddingTimeMS = Date().timeIntervalSince(embeddingStarted) * 1000
        guard embeddings.allSatisfy({ $0.count == vectorDimension }) else {
            throw LodeDBError.invalidArgument("embedding dimension does not match index")
        }
        _ = try nativeTextCore.applyTextUpsertJSON(
            planJSON: planJSON,
            embeddingsJSON: try encodeJSON(embeddings),
            embeddingTimeMS: embeddingTimeMS
        )
        guard let document = plan.documents.first(where: { $0.documentID == id }) else {
            throw LodeDBError.invalidArgument("native core returned no document plan")
        }
        guard let vector = embeddings.first ?? documents[id]?.vector else {
            throw LodeDBError.invalidArgument("text produced no chunks")
        }
        documents[id] = VectorDocument(
            vector: vector,
            metadata: document.metadata,
            text: document.text,
            chunkID: document.chunks.first?.chunkID ?? id,
            tokens: document.chunks.flatMap(\.tokens)
        )
    }

    private func searchTextWithNativeCore(
        _ nativeTextCore: NativeTextCore,
        text: String,
        k: Int,
        mode: RetrievalMode,
        embedder: LodeEmbedder?,
        filter: MetadataFilter
    ) throws -> [SearchHit] {
        let queryPlanJSON = try nativeTextCore.prepareQueryTextJSON(text, mode: mode.rawValue)
        let queryEmbeddingJSON: String?
        if mode == .vector || mode == .hybrid {
            guard let embedder else {
                throw LodeDBError.invalidArgument("embedder is required for vector search")
            }
            guard embedder.dimension == vectorDimension else {
                throw LodeDBError.invalidArgument("embedder dimension does not match index")
            }
            let embeddings = try embedder.embed(texts: [text])
            guard let query = embeddings.first else {
                throw LodeDBError.invalidArgument("embedder returned no query embedding")
            }
            guard query.count == vectorDimension else {
                throw LodeDBError.invalidArgument("query dimension does not match index")
            }
            queryEmbeddingJSON = try encodeJSON(query)
        } else {
            queryEmbeddingJSON = nil
        }
        let filterJSON = filter.isEmpty ? nil : try encodeJSON(["metadata": filter.exactMatches])
        let resultsJSON = try nativeTextCore.searchEmbeddedTextJSON(
            queryPlanJSON: queryPlanJSON,
            queryEmbeddingJSON: queryEmbeddingJSON,
            k: k,
            filterJSON: filterJSON
        )
        let results = try decodeJSON(NativeSearchResultsJSON.self, from: resultsJSON)
        return results.hits.map { hit in
            SearchHit(
                id: hit.documentID,
                chunkID: hit.chunkID,
                score: hit.score,
                metadata: hit.metadata
            )
        }
    }
}

public struct TextIngestPlan: Equatable, Sendable {
    public let id: String
    public let metadata: [String: String]
    public let text: String
    public let chunks: [TextChunk]
}

public struct TextChunk: Equatable, Sendable {
    public let documentID: String
    public let chunkID: String
    public let text: String
    public let tokens: [String]
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

private struct VectorDocument {
    let vector: [Float]
    let metadata: [String: String]
    var text: String?
    var chunkID: String
    var tokens: [String]

    init(
        vector: [Float],
        metadata: [String: String],
        text: String? = nil,
        chunkID: String? = nil,
        tokens: [String] = []
    ) {
        self.vector = vector
        self.metadata = metadata
        self.text = text
        self.chunkID = chunkID ?? ""
        self.tokens = tokens
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
        throw LodeDBError.invalidArgument("JSON is not valid UTF-8")
    }
    return try JSONDecoder().decode(type, from: data)
}

private func dot(_ left: [Float], _ right: [Float]) -> Float {
    zip(left, right).map(*).reduce(0, +)
}

private func readJSONObject(_ url: URL) throws -> [String: Any] {
    let data = try Data(contentsOf: url)
    guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        throw LodeDBError.corruptStore("\(url.lastPathComponent) is not a JSON object")
    }
    return object
}

private func chunkText(_ text: String, limit: Int) -> [String] {
    let stripped = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !stripped.isEmpty else {
        return []
    }
    var result: [String] = []
    var current = ""
    for character in stripped {
        current.append(character)
        if current.count == limit {
            result.append(current)
            current = ""
        }
    }
    if !current.isEmpty {
        result.append(current)
    }
    return result
}

private func tokenize(_ text: String) -> [String] {
    text.lowercased()
        .split { !$0.isLetter && !$0.isNumber && $0 != "-" && $0 != "_" }
        .map(String.init)
}

private func normalizedSHA256(_ text: String) -> String {
    let normalized = text.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    let digest = SHA256.hash(data: Data(normalized.utf8))
    return digest.map { String(format: "%02x", $0) }.joined()
}

private func reciprocalRankFusion(_ left: [SearchHit], _ right: [SearchHit], k: Int) -> [SearchHit] {
    var scores: [String: (SearchHit, Float)] = [:]
    for (offset, hit) in left.enumerated() {
        scores[hit.id] = (hit, (scores[hit.id]?.1 ?? 0) + 1 / Float(60 + offset + 1))
    }
    for (offset, hit) in right.enumerated() {
        scores[hit.id] = (hit, (scores[hit.id]?.1 ?? 0) + 1 / Float(60 + offset + 1))
    }
    return scores.values
        .map { hit, score in SearchHit(id: hit.id, chunkID: hit.chunkID, score: score, metadata: hit.metadata) }
        .sorted { left, right in
            if left.score == right.score {
                return left.id < right.id
            }
            return left.score > right.score
        }
        .prefix(k)
        .map { $0 }
}

private func normalizeMetadata(_ value: Any?) -> [String: String] {
    guard let object = value as? [String: Any] else {
        return [:]
    }
    var result: [String: String] = [:]
    for (key, value) in object {
        result[key] = String(describing: value)
    }
    return result
}
