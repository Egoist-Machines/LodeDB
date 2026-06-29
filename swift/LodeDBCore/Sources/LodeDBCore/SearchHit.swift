public struct SearchHit: Equatable, Sendable {
    public let id: String
    public let chunkID: String
    public let score: Float
    public let metadata: [String: String]

    public init(id: String, chunkID: String? = nil, score: Float, metadata: [String: String]) {
        self.id = id
        self.chunkID = chunkID ?? id
        self.score = score
        self.metadata = metadata
    }
}
