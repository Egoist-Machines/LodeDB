import Foundation

/// One recalled memory: the document id, its similarity score, the stored text (when
/// text was retained), and its metadata.
public struct MemoryHit: Sendable, Equatable {
    public let id: String
    public let score: Float
    public let text: String?
    public let metadata: [String: String]

    public init(id: String, score: Float, text: String?, metadata: [String: String]) {
        self.id = id
        self.score = score
        self.text = text
        self.metadata = metadata
    }
}

/// An agent-facing memory store: a small `save` / `recall` / `forget` contract over a
/// `LodeDB` collection, mirroring the verbs the Python MCP server and agent-memory
/// skill expose. Everything stays on device; nothing is sent off the machine.
///
/// This is the surface an in-app LLM agent uses for long-term memory:
/// - `save` ingests a memory (auto-generating an id when none is given);
/// - `recall` does a hybrid (vector + lexical) search by default and returns the
///   stored text alongside each hit (search results are otherwise payload-free);
/// - `forget` removes a memory by id.
public final class LodeMemory {
    private let db: LodeDB
    private let embedder: LodeEmbedder

    /// Wraps an existing store and embedder (e.g. a durable `LodeDB`).
    public init(db: LodeDB, embedder: LodeEmbedder) {
        self.db = db
        self.embedder = embedder
    }

    /// An ephemeral in-memory memory store sized to `embedder`. Binds to the explicit
    /// `modelIdentity` when given, otherwise the embedder's own `modelIdentity` (nil
    /// for embedders that do not declare one, in which case no identity is recorded).
    public convenience init(embedder: LodeEmbedder, modelIdentity: String? = nil) throws {
        let identity = modelIdentity ?? embedder.modelIdentity
        let db = try LodeDB(vectorDimension: embedder.dimension, modelIdentity: identity)
        self.init(db: db, embedder: embedder)
    }

    /// Saves a memory and returns its id (a fresh UUID when `id` is nil).
    @discardableResult
    public func save(_ text: String, id: String? = nil, metadata: [String: String] = [:]) throws -> String {
        let documentID = id ?? UUID().uuidString
        try db.addText(text, id: documentID, metadata: metadata, embedder: embedder)
        return documentID
    }

    /// Recalls the most relevant memories for `query`. Hybrid (vector + lexical) by
    /// default; the stored text is attached to each hit.
    ///
    /// Returns up to `k` distinct memories: the underlying text search is chunk-level,
    /// so a memory split into several chunks is deduplicated to its best-ranked hit.
    public func recall(
        _ query: String,
        k: Int = 5,
        mode: RetrievalMode = .hybrid,
        filter: MetadataFilter = MetadataFilter()
    ) throws -> [MemoryHit] {
        let hits = try db.search(text: query, k: k, mode: mode, embedder: embedder, filter: filter)
        // Hits arrive best-first; keep the first (highest-scored) hit per document id.
        var seen = Set<String>()
        let unique = hits.filter { seen.insert($0.id).inserted }
        let texts = try db.getTexts(unique.map(\.id))
        return unique.map { hit in
            MemoryHit(id: hit.id, score: hit.score, text: texts[hit.id], metadata: hit.metadata)
        }
    }

    /// Forgets a memory by id. Returns true if it existed.
    @discardableResult
    public func forget(_ id: String) throws -> Bool {
        try db.remove(id)
    }

    /// The number of stored memories.
    public var count: Int { db.count }

    /// Flushes a durable store to disk. No-op for an in-memory store.
    public func persist() throws { try db.persist() }
}
