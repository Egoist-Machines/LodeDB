import Foundation

/// A metadata filter applied natively during search.
///
/// Phase 0 supports flat exact-match equality (`["topic": "ops"]`). The full
/// Mongo-style predicate grammar the engine accepts ($gt/$in/$or/...) is exposed
/// in the retrieval-parity phase.
public struct MetadataFilter: Equatable, Sendable {
    private let expected: [String: String]

    public init(_ expected: [String: String] = [:]) {
        self.expected = expected
    }

    var isEmpty: Bool {
        expected.isEmpty
    }

    var exactMatches: [String: String] {
        expected
    }

    /// The filter encoded as the `{"metadata": {...}}` JSON the native core consumes,
    /// or `nil` when empty (so the search call passes `has_filter = 0`).
    var encodedJSON: String? {
        guard !expected.isEmpty else { return nil }
        guard let data = try? JSONEncoder().encode(["metadata": expected]),
              let text = String(data: data, encoding: .utf8) else {
            return nil
        }
        return text
    }

    public func matches(_ metadata: [String: String]) -> Bool {
        for (key, value) in expected where metadata[key] != value {
            return false
        }
        return true
    }
}
