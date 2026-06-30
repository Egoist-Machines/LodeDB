import Foundation

/// A late-interaction (multi-vector / MaxSim) index. Each document is a set of
/// per-token vectors ("patches"); a query is also a set of vectors, and documents
/// are scored by summing, over each query vector, its maximum similarity to any of
/// the document's patches (the ColBERT MaxSim operator), all in the native core.
///
/// Patches are stored as `float32`; the caller supplies the vectors and this type
/// handles encoding. Like `LodeDB`, access is serialized behind a lock.
public final class LodeLateInteractionIndex {
    public let vectorDimension: Int
    private let engine: NativeEngine
    private let lock = NSLock()

    /// Creates an ephemeral in-memory late-interaction index.
    public init(vectorDimension: Int) throws {
        guard vectorDimension > 0 else {
            throw LodeDBError.invalidArgument("vectorDimension must be positive")
        }
        self.vectorDimension = vectorDimension
        self.engine = try NativeEngine.inMemory(vectorDimension: vectorDimension)
    }

    /// Adds (or replaces) a document from its patch vectors. Each patch must have
    /// `vectorDimension` elements. A mean-pooled anchor vector is stored alongside so
    /// the document is also reachable by ordinary vector search.
    public func addDocument(id: String, patches: [[Float]], metadata: [String: String] = [:]) throws {
        try locked {
            guard !id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw LodeDBError.invalidArgument("id is required")
            }
            guard !patches.isEmpty else {
                throw LodeDBError.invalidArgument("patches must not be empty")
            }
            guard patches.allSatisfy({ $0.count == vectorDimension }) else {
                throw LodeDBError.invalidArgument("patch dimension does not match index")
            }
            let anchor = meanPool(patches)
            let bytes = littleEndianFloat32Bytes(patches.flatMap { $0 })
            let sidecar = [
                MultiVecSidecarJSON(
                    documentID: id,
                    metadata: metadata,
                    dtype: "float32",
                    patchCount: patches.count,
                    nbytes: bytes.count
                )
            ]
            _ = try engine.upsertMultivectorJSON(
                vectors: anchor,
                rows: 1,
                dim: vectorDimension,
                patchBytes: bytes,
                sidecarJSON: try encodeJSON(sidecar)
            )
        }
    }

    /// Scores documents against the multi-vector query by MaxSim and returns the top `k`.
    public func search(queryPatches: [[Float]], k: Int, filter: MetadataFilter = MetadataFilter()) throws -> [SearchHit] {
        try locked {
            guard k > 0 else {
                throw LodeDBError.invalidArgument("k must be positive")
            }
            guard !queryPatches.isEmpty else {
                throw LodeDBError.invalidArgument("queryPatches must not be empty")
            }
            guard queryPatches.allSatisfy({ $0.count == vectorDimension }) else {
                throw LodeDBError.invalidArgument("query patch dimension does not match index")
            }
            let json = try engine.queryMultivectorJSON(
                query: queryPatches.flatMap { $0 },
                nQuery: queryPatches.count,
                k: k,
                filterJSON: filter.encodedJSON
            )
            return try decodeJSON(NativeSearchResultsJSON.self, from: json).searchHits
        }
    }

    private func meanPool(_ patches: [[Float]]) -> [Float] {
        var sum = [Float](repeating: 0, count: vectorDimension)
        for patch in patches {
            for index in 0..<vectorDimension {
                sum[index] += patch[index]
            }
        }
        let count = Float(patches.count)
        return sum.map { $0 / count }
    }

    private func locked<T>(_ body: () throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try body()
    }
}

struct MultiVecSidecarJSON: Encodable {
    let documentID: String
    let metadata: [String: String]
    let dtype: String
    let patchCount: Int
    let nbytes: Int

    enum CodingKeys: String, CodingKey {
        case documentID = "document_id"
        case metadata
        case dtype
        case patchCount = "patch_count"
        case nbytes
    }
}

/// Encodes f32 values as a contiguous little-endian byte buffer, the `float32`
/// layout the native multi-vector store decodes (`f32::from_le_bytes`).
func littleEndianFloat32Bytes(_ values: [Float]) -> [UInt8] {
    var bytes = [UInt8]()
    bytes.reserveCapacity(values.count * 4)
    for value in values {
        let bits = value.bitPattern
        bytes.append(UInt8(bits & 0xff))
        bytes.append(UInt8((bits >> 8) & 0xff))
        bytes.append(UInt8((bits >> 16) & 0xff))
        bytes.append(UInt8((bits >> 24) & 0xff))
    }
    return bytes
}
