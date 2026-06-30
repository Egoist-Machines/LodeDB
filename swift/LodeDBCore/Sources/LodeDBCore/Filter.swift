import Foundation

/// A metadata predicate, mirroring the Mongo-style grammar the native filter
/// planner accepts. Values are strings (metadata is `[String: String]`); ordered
/// comparisons (`$gt`/`$gte`/`$lt`/`$lte`) compare numerically when both sides parse
/// as numbers and lexically otherwise, matching the engine.
public indirect enum FilterPredicate: Sendable, Equatable {
    case equals(String, String)
    case notEquals(String, String)
    case greaterThan(String, String)
    case greaterThanOrEqual(String, String)
    case lessThan(String, String)
    case lessThanOrEqual(String, String)
    case inSet(String, [String])
    case notInSet(String, [String])
    case exists(String, Bool)
    case and([FilterPredicate])
    case or([FilterPredicate])
    case not(FilterPredicate)

    /// The predicate as a JSON-serializable object tree (`JSONSerialization` value).
    var jsonObject: Any {
        switch self {
        case .equals(let field, let value): return [field: value]
        case .notEquals(let field, let value): return [field: ["$ne": value]]
        case .greaterThan(let field, let value): return [field: ["$gt": value]]
        case .greaterThanOrEqual(let field, let value): return [field: ["$gte": value]]
        case .lessThan(let field, let value): return [field: ["$lt": value]]
        case .lessThanOrEqual(let field, let value): return [field: ["$lte": value]]
        case .inSet(let field, let values): return [field: ["$in": values]]
        case .notInSet(let field, let values): return [field: ["$nin": values]]
        case .exists(let field, let present): return [field: ["$exists": present]]
        case .and(let predicates): return ["$and": predicates.map(\.jsonObject)]
        case .or(let predicates): return ["$or": predicates.map(\.jsonObject)]
        case .not(let predicate): return ["$not": predicate.jsonObject]
        }
    }
}

/// A metadata filter applied natively during search and listing. Supports flat
/// exact-match equality (`["topic": "ops"]`), the full Mongo-style predicate grammar
/// (via `FilterPredicate`), and an optional `document_ids` allowlist.
public struct MetadataFilter: Equatable, Sendable {
    private let predicate: FilterPredicate?
    private let documentIDs: [String]?

    /// Flat exact-match equality. An empty dictionary matches everything.
    public init(_ exactMatches: [String: String] = [:]) {
        self.predicate = MetadataFilter.exactPredicate(exactMatches)
        self.documentIDs = nil
    }

    /// A full predicate, optionally restricted to a `document_ids` allowlist.
    public init(predicate: FilterPredicate, documentIDs: [String]? = nil) {
        self.predicate = predicate
        self.documentIDs = documentIDs
    }

    /// A `document_ids` allowlist, optionally combined with a predicate.
    public init(documentIDs: [String], predicate: FilterPredicate? = nil) {
        self.predicate = predicate
        self.documentIDs = documentIDs
    }

    private static func exactPredicate(_ matches: [String: String]) -> FilterPredicate? {
        if matches.isEmpty { return nil }
        if matches.count == 1, let (field, value) = matches.first {
            return .equals(field, value)
        }
        return .and(matches.sorted { $0.key < $1.key }.map { .equals($0.key, $0.value) })
    }

    /// The filter encoded as the `{"metadata": ..., "document_ids": [...]}` envelope
    /// the native core consumes, or `nil` when empty (so the call passes no filter).
    var encodedJSON: String? {
        var envelope: [String: Any] = [:]
        if let predicate {
            envelope["metadata"] = predicate.jsonObject
        }
        if let documentIDs {
            envelope["document_ids"] = documentIDs
        }
        guard !envelope.isEmpty else { return nil }
        guard let data = try? JSONSerialization.data(withJSONObject: envelope),
              let text = String(data: data, encoding: .utf8) else {
            return nil
        }
        return text
    }
}
