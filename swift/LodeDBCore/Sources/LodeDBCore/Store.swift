import Foundation

/// Whether each commit flushes to disk (`fsync`) or relies on the OS page cache
/// (`buffered`). Maps to the native `CoreOpenOptions.durability`.
public enum Durability: String, Sendable {
    case buffered
    case fsync
}

/// How durable writes are committed: a write-ahead log (default, low per-write
/// latency) or generation-addressed snapshots. Maps to `CoreOpenOptions.commit_mode`.
public enum CommitMode: String, Sendable {
    case wal
    case generation
}

/// Options for opening a durable, on-disk LodeDB store.
public struct LodeStoreOptions: Sendable, Equatable {
    public var durability: Durability
    public var commitMode: CommitMode
    /// Retain raw document text (required for `get`/`getTexts`).
    public var storeText: Bool
    /// Index document text for lexical / hybrid search.
    public var indexText: Bool
    /// zstd-compress newly written retained text.
    public var compressText: Bool
    public var chunkCharacterLimit: Int
    /// Take the shared single-writer lock on `<dir>/.lodedb.lock` (default true).
    public var acquireWriterLock: Bool

    public init(
        durability: Durability = .fsync,
        commitMode: CommitMode = .wal,
        storeText: Bool = true,
        indexText: Bool = true,
        compressText: Bool = true,
        chunkCharacterLimit: Int = 8192,
        acquireWriterLock: Bool = true
    ) {
        self.durability = durability
        self.commitMode = commitMode
        self.storeText = storeText
        self.indexText = indexText
        self.compressText = compressText
        self.chunkCharacterLimit = chunkCharacterLimit
        self.acquireWriterLock = acquireWriterLock
    }

    func coreOpenOptionsJSON(path: String, readOnly: Bool) throws -> String {
        let payload = CoreOpenOptionsJSON(
            path: path,
            read_only: readOnly,
            durability: durability.rawValue,
            commit_mode: commitMode.rawValue,
            store_text: storeText,
            index_text: indexText,
            compress_text: compressText,
            chunk_character_limit: chunkCharacterLimit,
            acquire_writer_lock: acquireWriterLock
        )
        return try encodeJSON(payload)
    }
}

/// Metrics-only statistics for a collection. Carries no document text or vectors.
public struct CollectionStats: Sendable, Equatable {
    public let documentCount: Int
    public let chunkCount: Int
    public let generation: UInt64
    public let storageSchemaVersion: UInt32
    public let nativeCoreVersion: String
    public let vectorDimension: Int
    public let bitWidth: Int
    /// The persisted model identity the index was created with.
    public let model: String

    init(_ json: CoreEngineStatsJSON) {
        self.documentCount = json.documentCount
        self.chunkCount = json.chunkCount
        self.generation = json.generation
        self.storageSchemaVersion = json.storageSchemaVersion
        self.nativeCoreVersion = json.nativeCoreVersion
        self.vectorDimension = json.vectorDim
        self.bitWidth = json.bitWidth
        self.model = json.model
    }
}

/// A payload-free document record returned by `getDocument` / `listDocuments`.
/// Retained text is fetched separately via `get` (search hits and listings never
/// carry raw text).
public struct DocumentRecord: Sendable, Equatable {
    public let id: String
    public let metadata: [String: String]
    public let chunkCount: Int
    public let contentHash: String?

    init(_ json: DocumentRecordJSON) {
        self.id = json.documentID
        self.metadata = json.metadata
        self.chunkCount = json.chunkCount
        self.contentHash = json.contentHash
    }
}

/// How `updateDocument` should treat a document's retained text.
public enum TextUpdate: Sendable, Equatable {
    /// Leave the stored text unchanged.
    case unchanged
    /// Remove any stored text.
    case clear
    /// Replace the stored text.
    case set(String)
}

struct CoreOpenOptionsJSON: Encodable {
    let path: String
    let read_only: Bool
    let durability: String
    let commit_mode: String
    let store_text: Bool
    let index_text: Bool
    let compress_text: Bool
    let chunk_character_limit: Int
    let acquire_writer_lock: Bool
}

struct CoreEngineStatsJSON: Decodable {
    let documentCount: Int
    let chunkCount: Int
    let generation: UInt64
    let storageSchemaVersion: UInt32
    let nativeCoreVersion: String
    let vectorDim: Int
    let bitWidth: Int
    let model: String

    enum CodingKeys: String, CodingKey {
        case documentCount = "document_count"
        case chunkCount = "chunk_count"
        case generation
        case storageSchemaVersion = "storage_schema_version"
        case nativeCoreVersion = "native_core_version"
        case vectorDim = "vector_dim"
        case bitWidth = "bit_width"
        case model
    }
}

struct DocumentRecordJSON: Decodable {
    let documentID: String
    let metadata: [String: String]
    let chunkCount: Int
    let contentHash: String?

    enum CodingKeys: String, CodingKey {
        case documentID = "document_id"
        case metadata
        case chunkCount = "chunk_count"
        case contentHash = "content_hash"
    }
}

struct CoreMutationResultJSON: Decodable {
    let documentsUpserted: Int
    let documentsDeleted: Int

    enum CodingKeys: String, CodingKey {
        case documentsUpserted = "documents_upserted"
        case documentsDeleted = "documents_deleted"
    }
}
