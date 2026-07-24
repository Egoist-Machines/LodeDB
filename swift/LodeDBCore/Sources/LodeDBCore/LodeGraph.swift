import Foundation

/// The temporal frame a read resolves under (Graphiti's as-of): the current view,
/// as of an event-time instant (epoch ms), or every version (history).
public enum GraphAsOf: Sendable, Equatable {
    case now
    case nowValid
    case at(Int64)
    case atKnown(validAt: Int64, knownAt: Int64)
    case all

    var ms: Int64? {
        switch self {
        case let .at(t): return t
        case let .atKnown(validAt, _): return validAt
        default: return nil
        }
    }
    var allTime: Bool { if case .all = self { return true } else { return false } }
    var strictNow: Bool { if case .nowValid = self { return true } else { return false } }
    var knownAt: Int64? {
        if case let .atKnown(_, knownAt) = self { return knownAt } else { return nil }
    }
}

/// A property predicate evaluated inside the semantic index before candidate
/// ranking. Use the same scope fields on entities and facts when expanding a
/// protected subgraph.
public indirect enum GraphPropertyPredicate: Encodable, Sendable {
    case equals(String, String)
    case oneOf(String, [String])
    case exists(String, Bool)
    case all([GraphPropertyPredicate])
    case any([GraphPropertyPredicate])
    case not(GraphPropertyPredicate)

    private struct Key: CodingKey {
        let stringValue: String
        let intValue: Int? = nil
        init(_ value: String) { stringValue = value }
        init?(stringValue: String) { self.init(stringValue) }
        init?(intValue: Int) { return nil }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: Key.self)
        switch self {
        case let .equals(key, value):
            try container.encode(value, forKey: Key(key))
        case let .oneOf(key, values):
            try container.encode(["$in": values], forKey: Key(key))
        case let .exists(key, exists):
            try container.encode(["$exists": exists], forKey: Key(key))
        case let .all(predicates):
            try container.encode(predicates, forKey: Key("$and"))
        case let .any(predicates):
            try container.encode(predicates, forKey: Key("$or"))
        case let .not(predicate):
            try container.encode(predicate, forKey: Key("$not"))
        }
    }
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

/// A JSON value retained in an independently versioned entity property.
public indirect enum GraphPropertyValue: Codable, Equatable, Sendable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case object([String: GraphPropertyValue])
    case array([GraphPropertyValue])
    case null

    public init(from decoder: Decoder) throws {
        let value = try decoder.singleValueContainer()
        if value.decodeNil() {
            self = .null
        } else if let decoded = try? value.decode(Bool.self) {
            self = .bool(decoded)
        } else if let decoded = try? value.decode(Double.self) {
            self = .number(decoded)
        } else if let decoded = try? value.decode(String.self) {
            self = .string(decoded)
        } else if let decoded = try? value.decode([String: GraphPropertyValue].self) {
            self = .object(decoded)
        } else {
            self = .array(try value.decode([GraphPropertyValue].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var value = encoder.singleValueContainer()
        switch self {
        case let .string(decoded): try value.encode(decoded)
        case let .number(decoded): try value.encode(decoded)
        case let .bool(decoded): try value.encode(decoded)
        case let .object(decoded): try value.encode(decoded)
        case let .array(decoded): try value.encode(decoded)
        case .null: try value.encodeNil()
        }
    }
}

/// One version of one entity property, including its optional source episode.
public struct GraphPropertyVersion: Decodable, Equatable, Sendable {
    public let entityID: String
    public let key: String
    public let value: GraphPropertyValue
    public let episodeID: String?
    public let validAt: Int64?
    public let invalidAt: Int64?
    public let createdAt: Int64
    public let expiredAt: Int64?

    enum CodingKeys: String, CodingKey {
        case key, value
        case entityID = "entity_id"
        case episodeID = "episode_id"
        case validAt = "valid_at"
        case invalidAt = "invalid_at"
        case createdAt = "created_at"
        case expiredAt = "expired_at"
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
                           properties: [String: String]? = nil,
                           mentions: [String] = [], id: String? = nil) throws -> String {
        try locked {
            let req = AddEpisodeRequest(id: id, source: source, body: body,
                                        occurred_at: occurredAt, properties: properties,
                                        mentions: mentions)
            return try decodeGraphJSON(String.self, from: try native.addEpisode(try encodeGraphJSON(req)))
        }
    }

    // -- entities & facts ----------------------------------------------------

    /// Create or replace an entity (upsert by id); its label is embedded on device.
    @discardableResult
    public func upsertEntity(id: String, type: String, label: String,
                             properties: [String: String]? = nil,
                             propertySources: [String: String]? = nil,
                             validAt: Int64? = nil, invalidAt: Int64? = nil) throws -> String {
        try locked {
            let vec = try embedOne(label, role: .document)
            let req = UpsertEntityVecRequest(id: id, type: type, label: label, embedding: vec,
                                             properties: properties,
                                             valid_at: validAt, invalid_at: invalidAt,
                                             property_sources: propertySources)
            return try decodeGraphJSON(String.self, from: try native.upsertEntityVec(try encodeGraphJSON(req)))
        }
    }

    /// Assert a fact; its text is embedded on device. `invalidates` closes prior
    /// facts (Graphiti's rule). Returns the fact id.
    @discardableResult
    public func addFact(src: String, relation: String, dst: String, fact: String,
                        properties: [String: String]? = nil,
                        episodes: [String] = [], validAt: Int64? = nil,
                        invalidates: [String] = [], id: String? = nil) throws -> String {
        try locked {
            let vec = try embedOne(fact, role: .document)
            let req = AddFactVecRequest(id: id, src: src, relation: relation, dst: dst, fact: fact,
                                        embedding: vec, properties: properties,
                                        episodes: episodes, valid_at: validAt,
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
            let req = EntitiesRequest(type: type, as_of_ms: asOf.ms, all_time: asOf.allTime,
                                      strict_now: asOf.strictNow, known_at_ms: asOf.knownAt)
            return try decodeGraphJSON([GraphEntity].self, from: try native.entities(try encodeGraphJSON(req)))
        }
    }

    /// Facts incident to an entity (out/in/both, optional relation), as-of a frame.
    public func neighbors(id: String, direction: String = "out", relation: String? = nil,
                          asOf: GraphAsOf = .now) throws -> [GraphFact] {
        try locked {
            let req = NeighborsRequest(id: id, direction: direction, relation: relation,
                                       as_of_ms: asOf.ms, all_time: asOf.allTime,
                                       strict_now: asOf.strictNow, known_at_ms: asOf.knownAt)
            return try decodeGraphJSON([GraphFact].self, from: try native.neighbors(try encodeGraphJSON(req)))
        }
    }

    /// Deterministic k-hop neighbourhood around `seeds`, in a temporal frame.
    public func kHop(seeds: [String], k: Int = 1, direction: String = "both",
                     asOf: GraphAsOf = .now,
                     predicate: GraphPropertyPredicate? = nil) throws -> GraphSubgraph {
        try locked {
            let req = KHopRequest(seeds: seeds, k: k, direction: direction,
                                  as_of_ms: asOf.ms, all_time: asOf.allTime,
                                  strict_now: asOf.strictNow, known_at_ms: asOf.knownAt,
                                  predicate: predicate)
            return try decodeGraphJSON(GraphSubgraph.self, from: try native.kHop(try encodeGraphJSON(req)))
        }
    }

    /// Top-`k` entities semantically matching `query` (embedded on device), as-of a frame.
    public func semanticEntities(_ query: String, k: Int = 10, type: String? = nil,
                                 asOf: GraphAsOf = .now,
                                 predicate: GraphPropertyPredicate? = nil) throws -> [GraphScoredEntity] {
        try locked {
            let vec = try embedOne(query, role: .query)
            let req = SemanticEntitiesRequest(embedding: vec, k: k, type: type,
                                              as_of_ms: asOf.ms, all_time: asOf.allTime,
                                              strict_now: asOf.strictNow,
                                              known_at_ms: asOf.knownAt, predicate: predicate)
            return try decodeGraphJSON([GraphScoredEntity].self, from: try native.semanticEntities(try encodeGraphJSON(req)))
        }
    }

    /// Top-`k` facts semantically matching `query`, as-of a frame.
    public func semanticFacts(_ query: String, k: Int = 10, relation: String? = nil,
                              asOf: GraphAsOf = .now,
                              predicate: GraphPropertyPredicate? = nil) throws -> [GraphScoredFact] {
        try locked {
            let vec = try embedOne(query, role: .query)
            let req = SemanticFactsRequest(embedding: vec, k: k, relation: relation,
                                           as_of_ms: asOf.ms, all_time: asOf.allTime,
                                           strict_now: asOf.strictNow,
                                           known_at_ms: asOf.knownAt, predicate: predicate)
            return try decodeGraphJSON([GraphScoredFact].self, from: try native.semanticFacts(try encodeGraphJSON(req)))
        }
    }

    /// Semantic seed entities + k-hop expansion, the headline retrieval query.
    public func searchSubgraph(_ query: String, k: Int = 5, hops: Int = 1, direction: String = "both",
                               type: String? = nil, relation: String? = nil,
                               seedKind: String = "entity",
                               asOf: GraphAsOf = .now,
                               predicate: GraphPropertyPredicate? = nil) throws -> GraphSubgraph {
        try locked {
            let vec = try embedOne(query, role: .query)
            let req = SearchSubgraphRequest(embedding: vec, k: k, hops: hops, direction: direction,
                                            type: type, relation: relation,
                                            as_of_ms: asOf.ms, all_time: asOf.allTime,
                                            strict_now: asOf.strictNow,
                                            known_at_ms: asOf.knownAt, predicate: predicate,
                                            seed_kind: seedKind)
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
    public func resolveEntity(_ name: String, k: Int = 5,
                              predicate: GraphPropertyPredicate? = nil) throws -> [GraphScoredEntity] {
        try locked {
            let vec = try embedOne(name, role: .query)
            let req = SemanticEntitiesRequest(embedding: vec, k: k, type: nil,
                                              as_of_ms: nil, all_time: true,
                                              strict_now: false, known_at_ms: nil,
                                              predicate: predicate)
            return try decodeGraphJSON(
                [GraphScoredEntity].self,
                from: try native.semanticEntities(try encodeGraphJSON(req))
            )
        }
    }
    public func episodes() throws -> [GraphEpisode] {
        try locked { try decodeGraphJSON([GraphEpisode].self, from: try native.episodes("{}")) }
    }
    public func factsByEpisode(_ id: String) throws -> [GraphFact] {
        try locked {
            try decodeGraphJSON([GraphFact].self,
                                from: try native.factsByEpisode(try encodeGraphJSON(IdRequest(id: id))))
        }
    }
    public func entityPropertyHistory(_ id: String, key: String? = nil) throws
        -> [GraphPropertyVersion] {
        try locked {
            let request = PropertyHistoryRequest(id: id, key: key)
            return try decodeGraphJSON(
                [GraphPropertyVersion].self,
                from: try native.entityPropertyHistory(try encodeGraphJSON(request))
            )
        }
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
    @discardableResult public func removeEpisode(_ id: String) throws -> Bool {
        try locked {
            try decodeGraphJSON(Bool.self,
                                from: try native.removeEpisode(try encodeGraphJSON(IdRequest(id: id))))
        }
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
private struct AddEpisodeRequest: Encodable { let id: String?; let source: String; let body: String; let occurred_at: Int64; let properties: [String: String]?; let mentions: [String] }
private struct UpsertEntityVecRequest: Encodable { let id: String; let type: String; let label: String; let embedding: [Float]; let properties: [String: String]?; let valid_at: Int64?; let invalid_at: Int64?; let property_sources: [String: String]? }
private struct AddFactVecRequest: Encodable { let id: String?; let src: String; let relation: String; let dst: String; let fact: String; let embedding: [Float]; let properties: [String: String]?; let episodes: [String]; let valid_at: Int64?; let invalidates: [String] }
private struct InvalidateFactRequest: Encodable { let id: String; let invalid_at: Int64? }
private struct IdRequest: Encodable { let id: String }
private struct PropertyHistoryRequest: Encodable { let id: String; let key: String? }
private struct EntitiesRequest: Encodable { let type: String?; let as_of_ms: Int64?; let all_time: Bool; let strict_now: Bool; let known_at_ms: Int64? }
private struct NeighborsRequest: Encodable { let id: String; let direction: String; let relation: String?; let as_of_ms: Int64?; let all_time: Bool; let strict_now: Bool; let known_at_ms: Int64? }
private struct KHopRequest: Encodable { let seeds: [String]; let k: Int; let direction: String; let as_of_ms: Int64?; let all_time: Bool; let strict_now: Bool; let known_at_ms: Int64?; let predicate: GraphPropertyPredicate? }
private struct SemanticEntitiesRequest: Encodable { let embedding: [Float]; let k: Int; let type: String?; let as_of_ms: Int64?; let all_time: Bool; let strict_now: Bool; let known_at_ms: Int64?; let predicate: GraphPropertyPredicate? }
private struct SemanticFactsRequest: Encodable { let embedding: [Float]; let k: Int; let relation: String?; let as_of_ms: Int64?; let all_time: Bool; let strict_now: Bool; let known_at_ms: Int64?; let predicate: GraphPropertyPredicate? }
private struct SearchSubgraphRequest: Encodable { let embedding: [Float]; let k: Int; let hops: Int; let direction: String; let type: String?; let relation: String?; let as_of_ms: Int64?; let all_time: Bool; let strict_now: Bool; let known_at_ms: Int64?; let predicate: GraphPropertyPredicate?; let seed_kind: String }
