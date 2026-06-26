public struct MetadataFilter: Equatable, Sendable {
    private let expected: [String: String]

    public init(_ expected: [String: String] = [:]) {
        self.expected = expected
    }

    public func matches(_ metadata: [String: String]) -> Bool {
        for (key, value) in expected where metadata[key] != value {
            return false
        }
        return true
    }
}
