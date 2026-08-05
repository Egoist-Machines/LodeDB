import Foundation

/// Stable results of the managed three-pointer classifier.
public enum SyncClassification: String, Decodable, Sendable, Equatable {
    case inSync = "in_sync"
    case localAhead = "local_ahead"
    case remoteAhead = "remote_ahead"
    case diverged
    case republish
    case unknown
}

/// One content-addressed artifact a managed pull needs the app to download.
public struct PullRequirement: Decodable, Sendable, Equatable {
    public let name: String
    public let sha256: String
    public let sizeBytes: UInt64
    public let kind: String
    public let epoch: UInt64
    public let isBase: Bool

    enum CodingKeys: String, CodingKey {
        case name, sha256, kind, epoch
        case sizeBytes = "size_bytes"
        case isBase = "is_base"
    }
}

/// A managed generation identity and the payload stores it includes.
public struct PullPlanSide: Decodable, Sendable, Equatable {
    public let snapshotID: String
    public let logicalID: String
    public let generation: UInt64
    public let hasText: Bool
    public let hasLexical: Bool

    enum CodingKeys: String, CodingKey {
        case generation
        case snapshotID = "snapshot_id"
        case logicalID = "logical_id"
        case hasText = "has_text"
        case hasLexical = "has_lexical"
    }
}

/// The base identity recorded by the local managed-sync sidecar.
public struct PullPlanBase: Decodable, Sendable, Equatable {
    public let snapshotID: String
    public let logicalID: String
    public let generation: UInt64

    enum CodingKeys: String, CodingKey {
        case generation
        case snapshotID = "snapshot_id"
        case logicalID = "logical_id"
    }
}

/// The local committed generation a managed plan found.
public struct PullPlanLocal: Decodable, Sendable, Equatable {
    public let side: PullPlanSide
    public let legacyRedactedID: String
    /// Canonical manifest body passed to the app's control-plane request.
    public let bodyJSON: String
    /// Engine-written pointer document the control plane may store verbatim.
    public let pointerDocument: String
    public let artifacts: [PullRequirement]

    enum CodingKeys: String, CodingKey {
        case legacyRedactedID = "legacy_redacted_id"
        case bodyJSON = "body_json"
        case pointerDocument = "pointer_document"
        case artifacts
        case snapshotID = "snapshot_id"
        case logicalID = "logical_id"
        case generation
        case hasText = "has_text"
        case hasLexical = "has_lexical"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        side = PullPlanSide(
            snapshotID: try container.decode(String.self, forKey: .snapshotID),
            logicalID: try container.decode(String.self, forKey: .logicalID),
            generation: try container.decode(UInt64.self, forKey: .generation),
            hasText: try container.decode(Bool.self, forKey: .hasText),
            hasLexical: try container.decode(Bool.self, forKey: .hasLexical)
        )
        legacyRedactedID = try container.decode(String.self, forKey: .legacyRedactedID)
        bodyJSON = try container.decode(String.self, forKey: .bodyJSON)
        pointerDocument = try container.decode(String.self, forKey: .pointerDocument)
        artifacts = try container.decode([PullRequirement].self, forKey: .artifacts)
    }
}

/// A read-only managed-cloud synchronization decision.
public struct PullPlan: Decodable, Sendable, Equatable {
    public let indexKey: String
    public let localGeneration: UInt64?
    public let remoteGeneration: UInt64?
    public let localDocumentCount: UInt64?
    public let remoteDocumentCount: UInt64?
    public let localChunkCount: UInt64?
    public let remoteChunkCount: UInt64?
    public let artifactsToUpload: Int
    public let bytesToUpload: UInt64
    public let shipsBase: Bool
    public let inSync: Bool
    public let sidecarPresent: Bool
    public let sidecarCorrupt: Bool
    public let baseGeneration: UInt64?
    public let classification: SyncClassification?
    public let local: PullPlanLocal?
    public let remote: PullPlanSide?
    public let base: PullPlanBase?
    public let baseIsCurrent: Bool
    /// The raw local snapshot that a later materialization can pin against.
    public let localRawSnapshotID: String?

    enum CodingKeys: String, CodingKey {
        case indexKey = "index_key"
        case localGeneration = "local_generation"
        case remoteGeneration = "remote_generation"
        case localDocumentCount = "local_document_count"
        case remoteDocumentCount = "remote_document_count"
        case localChunkCount = "local_chunk_count"
        case remoteChunkCount = "remote_chunk_count"
        case artifactsToUpload = "artifacts_to_upload"
        case bytesToUpload = "bytes_to_upload"
        case shipsBase = "ships_base"
        case inSync = "in_sync"
        case sidecarPresent = "sidecar_present"
        case sidecarCorrupt = "sidecar_corrupt"
        case baseGeneration = "base_generation"
        case classification, local, remote, base
        case baseIsCurrent = "base_is_current"
        case localRawSnapshotID = "local_raw_snapshot_id"
    }
}

/// Metrics from the staging transfer that materialized a managed pull.
public struct PullTransfer: Decodable, Sendable, Equatable {
    public let indexKey: String
    public let generation: UInt64
    public let artifactsWritten: Int
    public let artifactsSkipped: Int
    public let bytesWritten: UInt64
    public let pointerPublished: Bool

    enum CodingKeys: String, CodingKey {
        case generation
        case indexKey = "index_key"
        case artifactsWritten = "artifacts_written"
        case artifactsSkipped = "artifacts_skipped"
        case bytesWritten = "bytes_written"
        case pointerPublished = "pointer_published"
    }
}

/// Counts proved by opening the restored generation before publishing it.
public struct PullOpen: Decodable, Sendable, Equatable {
    public let documentCount: UInt64
    public let chunkCount: UInt64

    enum CodingKeys: String, CodingKey {
        case documentCount = "document_count"
        case chunkCount = "chunk_count"
    }
}

/// The transfer and verify-open reports produced by a managed materialization.
public struct PullOutcome: Decodable, Sendable, Equatable {
    public let transfer: PullTransfer
    public let open: PullOpen
}

/// Managed cloud replication with application-owned HTTP.
///
/// This type deliberately has no networking API. Call `managedPlan`, download
/// each returned SHA-256 blob with the app's HTTP client into `stagingDirectory`,
/// then call `materialize`. Native code validates every staged byte before the
/// destination pointer can move. Calls are serialized with an `NSLock`.
public final class LodeReplica {
    private let native: NativeCloudReplica
    private let lock = NSLock()

    public init() throws {
        native = try NativeCloudReplica.open()
    }

    /// Builds a read-only plan for one local index and a control-plane remote head.
    public func managedPlan(
        directory: URL,
        indexKey: String,
        remoteID: String,
        remoteBody: String? = nil,
        includeText: Bool = false,
        includeLexical: Bool = false
    ) throws -> PullPlan {
        try locked {
            let request = ManagedPlanRequest(
                dir: directory.path,
                index_key: indexKey,
                remote_id: remoteID,
                remote_body: remoteBody,
                include_text: includeText,
                include_lexical: includeLexical
            )
            return try decodeJSON(PullPlan.self, from: native.managedPlan(try encodeJSON(request)))
        }
    }

    /// Returns the body-pinned blobs absent from the local directory.
    public func pullRequirements(
        directory: URL,
        indexKey: String,
        body: String
    ) throws -> [PullRequirement] {
        try locked {
            let request = PullRequirementsRequest(dir: directory.path, index_key: indexKey, body: body)
            let response = try decodeJSON(PullRequirementsResponse.self, from: native.pullRequirements(try encodeJSON(request)))
            return response.artifacts
        }
    }

    /// Verifies blobs named by SHA-256 under `stagingDirectory`, restores the generation,
    /// verify-opens it, and records the remote body as the local sidecar base.
    public func materialize(
        directory: URL,
        indexKey: String,
        remoteID: String,
        body: String,
        stagingDirectory: URL,
        discardPendingWAL: Bool = false,
        expectedLocalSnapshotID: String? = nil
    ) throws -> PullOutcome {
        try locked {
            let request = MaterializeRequest(
                dir: directory.path,
                index_key: indexKey,
                remote_id: remoteID,
                body: body,
                staging_dir: stagingDirectory.path,
                discard_pending_wal: discardPendingWAL,
                expected_local_snapshot_id: expectedLocalSnapshotID
            )
            return try decodeJSON(PullOutcome.self, from: native.materialize(try encodeJSON(request)))
        }
    }

    /// Records a body as the remote's trusted sidecar base after app-owned transfer work.
    public func recordBase(directory: URL, indexKey: String, remoteID: String, body: String) throws {
        try locked {
            let request = RecordBaseRequest(
                dir: directory.path,
                index_key: indexKey,
                remote_id: remoteID,
                body: body
            )
            _ = try native.recordBase(try encodeJSON(request))
        }
    }

    private func locked<T>(_ body: () throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try body()
    }
}

private struct ManagedPlanRequest: Encodable {
    let dir: String
    let index_key: String
    let remote_id: String
    let remote_body: String?
    let include_text: Bool
    let include_lexical: Bool
}

private struct PullRequirementsRequest: Encodable {
    let dir: String
    let index_key: String
    let body: String
}

private struct PullRequirementsResponse: Decodable {
    let artifacts: [PullRequirement]
}

private struct MaterializeRequest: Encodable {
    let dir: String
    let index_key: String
    let remote_id: String
    let body: String
    let staging_dir: String
    let discard_pending_wal: Bool
    let expected_local_snapshot_id: String?
}

private struct RecordBaseRequest: Encodable {
    let dir: String
    let index_key: String
    let remote_id: String
    let body: String
}
