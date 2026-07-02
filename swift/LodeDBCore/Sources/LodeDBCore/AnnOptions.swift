/// Opt-in approximate-nearest-neighbor tuning for an index.
///
/// ANN is disabled by default (exact scan, full recall). Enable it at index
/// creation to trade a little recall for speed on large corpora: a query scores
/// cluster centroids, scans only the nearest clusters, and the exact TurboVec scan
/// re-scores those candidates. Returned scores are therefore exact, but the result
/// set is approximate (a true neighbor in an unprobed cluster can be missed, so
/// recall is below 100%). Probing every cluster reproduces the exact result. Only
/// `"cluster"` (IVF-style cluster pruning) is supported today.
public struct LodeAnnOptions: Sendable, Equatable {
    /// ANN algorithm; the only supported value is `"cluster"`.
    public var algorithm: String
    /// Partition count. `nil` uses a corpus-derived default (about `sqrt(n)`).
    public var clusters: Int?
    /// Clusters probed per query. `nil` uses a default (about `sqrt(clusters)`).
    public var nprobe: Int?

    public init(algorithm: String = "cluster", clusters: Int? = nil, nprobe: Int? = nil) {
        self.algorithm = algorithm
        self.clusters = clusters
        self.nprobe = nprobe
    }

    /// IVF-style cluster pruning with corpus-derived default tuning.
    public static let cluster = LodeAnnOptions()
}
