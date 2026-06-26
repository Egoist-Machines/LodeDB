import Foundation

public final class LodeDB {
    private let vectorDimension: Int
    private var documents: [String: VectorDocument]

    public init(vectorDimension: Int) throws {
        guard vectorDimension > 0 else {
            throw LodeDBError.invalidArgument("vectorDimension must be positive")
        }
        self.vectorDimension = vectorDimension
        self.documents = [:]
    }

    private init(vectorDimension: Int, documents: [String: VectorDocument]) {
        self.vectorDimension = vectorDimension
        self.documents = documents
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
        documents[id] = VectorDocument(vector: vector, metadata: metadata)
    }

    public func search(vector: [Float], k: Int, filter: MetadataFilter = MetadataFilter()) throws -> [SearchHit] {
        guard vector.count == vectorDimension else {
            throw LodeDBError.invalidArgument("query dimension does not match index")
        }
        guard k > 0 else {
            throw LodeDBError.invalidArgument("k must be positive")
        }
        return documents
            .filter { filter.matches($0.value.metadata) }
            .map { id, document in
                SearchHit(id: id, score: dot(vector, document.vector), metadata: document.metadata)
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
}

private struct VectorDocument {
    let vector: [Float]
    let metadata: [String: String]
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
