import Foundation

/// The temporal frame a read resolves under (Graphiti's as-of): the current view,
/// as of an event-time instant (epoch ms), or every version (history).
public enum GraphAsOf: Sendable, Equatable {
    case now
    case at(Int64)
    case all

    var ms: Int64? { if case let .at(t) = self { return t } else { return nil } }
    var allTime: Bool { if case .all = self { return true } else { return false } }
}

// MARK: - Result types (decoded from the native JSON)

/// A resolved thing in the world. `properties` are not decoded here (v1); read
/// from `label`, `type`, and the relations.
public struct GraphEntity: Decodable, Equatable, Sendable {
    public let id: String
    public let type: String
    public let label: String
    public let validAt: Int64?
    public let invalidAt: Int64?
    public let createdAt: Int64
    public let expiredAt: Int64?

    enum CodingKeys: String, CodingKey {
        case id, label
        case type = "entity_type"
        case validAt = "valid_at"
        case invalidAt = "invalid_at"
        case createdAt = "created_at"
        case expiredAt = "expired_at"
    }
}

/// A typed, directed, labeled, bi-temporal relationship.
public struct GraphFact: Decodable, Equatable, Sendable {
    public let id: String
    public let src: String
    public let relation: String
    public let dst: String
    public let fact: String
    public let episodes: [String]
    public let validAt: Int64?
    public let invalidAt: Int64?
    public let createdAt: Int64
    public let expiredAt: Int64?
    public let referenceTime: Int64?

    /// Whether the fact is currently live (not superseded, not ended).
    public var isLive: Bool { expiredAt == nil && invalidAt == nil }

    enum CodingKeys: String, CodingKey {
        case id, src, relation, dst, fact, episodes
        case validAt = "valid_at"
        case invalidAt = "invalid_at"
        case createdAt = "created_at"
        case expiredAt = "expired_at"
        case referenceTime = "reference_time"
    }
}

/// A raw observation (a note, a chat turn) the graph was built from.
public struct GraphEpisode: Decodable, Equatable, Sendable {
    public let id: String
    public let source: String
    public let body: String
    public let occurredAt: Int64
    public let createdAt: Int64

    enum CodingKeys: String, CodingKey {
        case id, source, body
        case occurredAt = "occurred_at"
        case createdAt = "created_at"
    }
}

/// An entity with its relevance score (native `[score, entity]` pair).
public struct GraphScoredEntity: Decodable, Sendable {
    public let score: Float
    public let entity: GraphEntity
    public init(from decoder: Decoder) throws {
        var c = try decoder.unkeyedContainer()
        score = try c.decode(Float.self)
        entity = try c.decode(GraphEntity.self)
    }
}

/// A fact with its relevance score (native `[score, fact]` pair).
public struct GraphScoredFact: Decodable, Sendable {
    public let score: Float
    public let fact: GraphFact
    public init(from decoder: Decoder) throws {
        var c = try decoder.unkeyedContainer()
        score = try c.decode(Float.self)
        fact = try c.decode(GraphFact.self)
    }
}

/// A semantic seed entity (native `[id, score]` pair).
public struct GraphSeed: Decodable, Sendable {
    public let id: String
    public let score: Float
    public init(from decoder: Decoder) throws {
        var c = try decoder.unkeyedContainer()
        id = try c.decode(String.self)
        score = try c.decode(Float.self)
    }
}

/// A retrieved neighbourhood: entities, the facts among them, and the seeds.
public struct GraphSubgraph: Decodable, Sendable {
    public let entities: [GraphEntity]
    public let facts: [GraphFact]
    public let seeds: [GraphSeed]

    enum CodingKeys: String, CodingKey { case entities, facts, seeds }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let map = try c.decode([String: GraphEntity].self, forKey: .entities)
        entities = Array(map.values)
        facts = try c.decode([GraphFact].self, forKey: .facts)
        seeds = (try? c.decode([GraphSeed].self, forKey: .seeds)) ?? []
    }
}

public struct GraphStats: Decodable, Equatable, Sendable {
    public let entities: Int
    public let facts: Int
    public let indexedDocuments: Int
    enum CodingKeys: String, CodingKey {
        case entities, facts
        case indexedDocuments = "indexed_documents"
    }
}

public struct GraphReindexStats: Decodable, Equatable, Sendable {
    public let reindexedEntities: Int
    public let reindexedFacts: Int
    public let removedOrphans: Int
    enum CodingKeys: String, CodingKey {
        case reindexedEntities = "reindexed_entities"
        case reindexedFacts = "reindexed_facts"
        case removedOrphans = "removed_orphans"
    }
}

// MARK: - LodeGraph

/// A bi-temporal knowledge graph on device (the native `lodedb-graph`).
///
/// Raw observations go in as **episodes**; a caller-side extractor turns them into
/// typed **entities** and **facts**, which apps read back by enumerating entities
/// and traversing facts, "as of" now or any past instant. The graph holds no
/// embedder — this type embeds label/fact/query text on device (via the supplied
/// `LodeEmbedder`) and feeds the native store vectors. Thread-safe (serialized).
public final class LodeGraph {
    private let native: NativeTemporalGraph
    private let embedder: LodeEmbedder
    private let lock = NSLock()

    /// Open (or create) a graph. `path == nil` is in-memory. The embedder's
    /// `dimension` sets the index dimension. `indexText: false` keeps the semantic
    /// index vector-only (no label/fact text retained on the index side).
    public init(path: URL? = nil, embedder: LodeEmbedder, indexFacts: Bool = true,
                indexText: Bool = true) throws {
        self.embedder = embedder
        let req = OpenRequest(path: path?.path, vector_dim: embedder.dimension,
                              index_facts: indexFacts, index_text: indexText)
        self.native = try NativeTemporalGraph.open(requestJSON: try encodeGraphJSON(req))
    }

    // -- episodes ------------------------------------------------------------

    /// Store a raw observation (no extraction, no embedding). Returns its id.
    @discardableResult
    public func addEpisode(source: String, body: String, occurredAt: Int64,
                           mentions: [String] = []) throws -> String {
        try locked {
            let req = AddEpisodeRequest(source: source, body: body, occurred_at: occurredAt, mentions: mentions)
            return try decodeGraphJSON(String.self, from: try native.addEpisode(try encodeGraphJSON(req)))
        }
    }

    // -- entities & facts ----------------------------------------------------

    /// Create or replace an entity (upsert by id); its label is embedded on device.
    @discardableResult
    public func upsertEntity(id: String, type: String, label: String,
                             validAt: Int64? = nil, invalidAt: Int64? = nil) throws -> String {
        try locked {
            let vec = try embedOne(label, role: .document)
            let req = UpsertEntityVecRequest(id: id, type: type, label: label, embedding: vec,
                                             valid_at: validAt, invalid_at: invalidAt)
            return try decodeGraphJSON(String.self, from: try native.upsertEntityVec(try encodeGraphJSON(req)))
        }
    }

    /// Assert a fact; its text is embedded on device. `invalidates` closes prior
    /// facts (Graphiti's rule). Returns the fact id.
    @discardableResult
    public func addFact(src: String, relation: String, dst: String, fact: String,
                        episodes: [String] = [], validAt: Int64? = nil,
                        invalidates: [String] = []) throws -> String {
        try locked {
            let vec = try embedOne(fact, role: .document)
            let req = AddFactVecRequest(src: src, relation: relation, dst: dst, fact: fact,
                                        embedding: vec, episodes: episodes, valid_at: validAt,
                                        invalidates: invalidates)
            return try decodeGraphJSON(String.self, from: try native.addFactVec(try encodeGraphJSON(req)))
        }
    }

    @discardableResult
    public func invalidateFact(id: String, invalidAt: Int64? = nil) throws -> Bool {
        try locked {
            let req = InvalidateFactRequest(id: id, invalid_at: invalidAt)
            return try decodeGraphJSON(Bool.self, from: try native.invalidateFact(try encodeGraphJSON(req)))
        }
    }

    // -- reads ---------------------------------------------------------------

    /// Every entity of a type (nil = all), in a temporal frame.
    public func entities(type: String? = nil, asOf: GraphAsOf = .now) throws -> [GraphEntity] {
        try locked {
            let req = EntitiesRequest(type: type, as_of_ms: asOf.ms, all_time: asOf.allTime)
            return try decodeGraphJSON([GraphEntity].self, from: try native.entities(try encodeGraphJSON(req)))
        }
    }

    /// Facts incident to an entity (out/in/both, optional relation), as-of a frame.
    public func neighbors(id: String, direction: String = "out", relation: String? = nil,
                          asOf: GraphAsOf = .now) throws -> [GraphFact] {
        try locked {
            let req = NeighborsRequest(id: id, direction: direction, relation: relation,
                                       as_of_ms: asOf.ms, all_time: asOf.allTime)
            return try decodeGraphJSON([GraphFact].self, from: try native.neighbors(try encodeGraphJSON(req)))
        }
    }

    /// Deterministic k-hop neighbourhood around `seeds`, in a temporal frame.
    public func kHop(seeds: [String], k: Int = 1, direction: String = "both",
                     asOf: GraphAsOf = .now) throws -> GraphSubgraph {
        try locked {
            let req = KHopRequest(seeds: seeds, k: k, direction: direction,
                                  as_of_ms: asOf.ms, all_time: asOf.allTime)
            return try decodeGraphJSON(GraphSubgraph.self, from: try native.kHop(try encodeGraphJSON(req)))
        }
    }

    /// Top-`k` entities semantically matching `query` (embedded on device), as-of a frame.
    public func semanticEntities(_ query: String, k: Int = 10, type: String? = nil,
                                 asOf: GraphAsOf = .now) throws -> [GraphScoredEntity] {
        try locked {
            let vec = try embedOne(query, role: .query)
            let req = SemanticEntitiesRequest(embedding: vec, k: k, type: type,
                                              as_of_ms: asOf.ms, all_time: asOf.allTime)
            return try decodeGraphJSON([GraphScoredEntity].self, from: try native.semanticEntities(try encodeGraphJSON(req)))
        }
    }

    /// Top-`k` facts semantically matching `query`, as-of a frame.
    public func semanticFacts(_ query: String, k: Int = 10, relation: String? = nil,
                              asOf: GraphAsOf = .now) throws -> [GraphScoredFact] {
        try locked {
            let vec = try embedOne(query, role: .query)
            let req = SemanticFactsRequest(embedding: vec, k: k, relation: relation,
                                           as_of_ms: asOf.ms, all_time: asOf.allTime)
            return try decodeGraphJSON([GraphScoredFact].self, from: try native.semanticFacts(try encodeGraphJSON(req)))
        }
    }

    /// Semantic seed entities + k-hop expansion, the headline retrieval query.
    public func searchSubgraph(_ query: String, k: Int = 5, hops: Int = 1, direction: String = "both",
                               type: String? = nil, asOf: GraphAsOf = .now) throws -> GraphSubgraph {
        try locked {
            let vec = try embedOne(query, role: .query)
            let req = SearchSubgraphRequest(embedding: vec, k: k, hops: hops, direction: direction,
                                            type: type, as_of_ms: asOf.ms, all_time: asOf.allTime)
            return try decodeGraphJSON(GraphSubgraph.self, from: try native.searchSubgraph(try encodeGraphJSON(req)))
        }
    }

    public func getEntity(_ id: String) throws -> GraphEntity? {
        try locked { try decodeGraphJSON(GraphEntity?.self, from: try native.getEntity(try encodeGraphJSON(IdRequest(id: id)))) }
    }
    public func getFact(_ id: String) throws -> GraphFact? {
        try locked { try decodeGraphJSON(GraphFact?.self, from: try native.getFact(try encodeGraphJSON(IdRequest(id: id)))) }
    }
    public func getEpisode(_ id: String) throws -> GraphEpisode? {
        try locked { try decodeGraphJSON(GraphEpisode?.self, from: try native.getEpisode(try encodeGraphJSON(IdRequest(id: id)))) }
    }

    /// Every fact ever touching an entity, all frames (history).
    public func history(entityID: String) throws -> [GraphFact] {
        try locked { try decodeGraphJSON([GraphFact].self, from: try native.history(try encodeGraphJSON(IdRequest(id: entityID)))) }
    }

    @discardableResult public func removeEntity(_ id: String) throws -> Bool {
        try locked { try decodeGraphJSON(Bool.self, from: try native.removeEntity(try encodeGraphJSON(IdRequest(id: id)))) }
    }
    @discardableResult public func removeFact(_ id: String) throws -> Bool {
        try locked { try decodeGraphJSON(Bool.self, from: try native.removeFact(try encodeGraphJSON(IdRequest(id: id)))) }
    }

    @discardableResult public func reindex() throws -> GraphReindexStats {
        try locked { try decodeGraphJSON(GraphReindexStats.self, from: try native.reindex()) }
    }
    public func stats() throws -> GraphStats {
        try locked { try decodeGraphJSON(GraphStats.self, from: try native.stats()) }
    }
    public func persist() throws { try locked { try native.persist() } }

    // -- internals ---------------------------------------------------------

    private func locked<T>(_ body: () throws -> T) rethrows -> T {
        lock.lock(); defer { lock.unlock() }
        return try body()
    }

    private func embedOne(_ text: String, role: EmbeddingRole) throws -> [Float] {
        let vectors = try embedder.embed(texts: [text], role: role)
        guard let vec = vectors.first, vec.count == embedder.dimension else {
            throw LodeDBError.invalidArgument("embedder returned an invalid embedding")
        }
        return vec
    }
}

// MARK: - JSON helpers + serde-shaped request payloads

private func encodeGraphJSON<T: Encodable>(_ value: T) throws -> String {
    let data = try JSONEncoder().encode(value)
    guard let text = String(data: data, encoding: .utf8) else {
        throw LodeDBError.invalidArgument("failed to encode request JSON")
    }
    return text
}

private func decodeGraphJSON<T: Decodable>(_ type: T.Type, from text: String) throws -> T {
    guard let data = text.data(using: .utf8) else {
        throw LodeDBError.internalError("native core returned invalid UTF-8")
    }
    return try JSONDecoder().decode(type, from: data)
}

private struct OpenRequest: Encodable { let path: String?; let vector_dim: Int; let index_facts: Bool; let index_text: Bool }
private struct AddEpisodeRequest: Encodable { let source: String; let body: String; let occurred_at: Int64; let mentions: [String] }
private struct UpsertEntityVecRequest: Encodable { let id: String; let type: String; let label: String; let embedding: [Float]; let valid_at: Int64?; let invalid_at: Int64? }
private struct AddFactVecRequest: Encodable { let src: String; let relation: String; let dst: String; let fact: String; let embedding: [Float]; let episodes: [String]; let valid_at: Int64?; let invalidates: [String] }
private struct InvalidateFactRequest: Encodable { let id: String; let invalid_at: Int64? }
private struct IdRequest: Encodable { let id: String }
private struct EntitiesRequest: Encodable { let type: String?; let as_of_ms: Int64?; let all_time: Bool }
private struct NeighborsRequest: Encodable { let id: String; let direction: String; let relation: String?; let as_of_ms: Int64?; let all_time: Bool }
private struct KHopRequest: Encodable { let seeds: [String]; let k: Int; let direction: String; let as_of_ms: Int64?; let all_time: Bool }
private struct SemanticEntitiesRequest: Encodable { let embedding: [Float]; let k: Int; let type: String?; let as_of_ms: Int64?; let all_time: Bool }
private struct SemanticFactsRequest: Encodable { let embedding: [Float]; let k: Int; let relation: String?; let as_of_ms: Int64?; let all_time: Bool }
private struct SearchSubgraphRequest: Encodable { let embedding: [Float]; let k: Int; let hops: Int; let direction: String; let type: String?; let as_of_ms: Int64?; let all_time: Bool }
