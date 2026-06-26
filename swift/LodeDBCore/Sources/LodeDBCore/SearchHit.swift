public struct SearchHit: Equatable, Sendable {
    public let id: String
    public let score: Float
    public let metadata: [String: String]

    public init(id: String, score: Float, metadata: [String: String]) {
        self.id = id
        self.score = score
        self.metadata = metadata
    }
}
