import Foundation

/// One result of a brute-force vector scan: the row index and its similarity score.
public struct VectorScanHit: Sendable, Equatable {
    public let index: Int
    public let score: Float

    public init(index: Int, score: Float) {
        self.index = index
        self.score = score
    }
}

/// Which backend a scan ran on.
public enum ScanBackend: String, Sendable {
    case metal
    case cpu
}

/// An OPT-IN, exact (full-precision f32) brute-force top-k dot-product scanner with a
/// GPU (Metal) path and a CPU fallback.
///
/// This is NOT the default LodeDB search path. `LodeDB`/`search` use the native
/// TurboVec NEON scan over the quantized index, which on-device benchmarks show is
/// faster than a GPU/MPS scan (and CUDA is irrelevant on Apple platforms). This
/// scanner is for callers that hold raw f32 vectors and want a GPU-accelerated exact
/// scan — e.g. exact reranking of a candidate set, or scanning app-managed vectors —
/// where the Metal path can win at large `count`. Accelerating the core's *quantized*
/// scan with Metal is deferred pending device benchmarks that show NEON is the
/// bottleneck.
///
/// Results are identical to the CPU reference (same f32 dot products); the backend is
/// selected once at init and reported via `backend`.
public final class MetalVectorScanner {
    public let backend: ScanBackend
    private let metal: MetalContext?

    /// Whether a Metal device is present on this system.
    public static var isMetalAvailable: Bool {
        MetalContext.makeDefault() != nil
    }

    /// The default Metal device's name, or nil if no Metal device is available.
    public static var metalDeviceName: String? {
        MetalContext.makeDefault()?.deviceName
    }

    /// Creates a scanner. Uses Metal when available and `preferMetal` is true (the
    /// default), otherwise the CPU path.
    public init(preferMetal: Bool = true) {
        if preferMetal, let metal = MetalContext.makeDefault() {
            self.metal = metal
            self.backend = .metal
        } else {
            self.metal = nil
            self.backend = .cpu
        }
    }

    /// Returns the top `k` rows of `vectors` (flat row-major `count * dim` f32) by dot
    /// product with `query` (`dim` f32), highest score first, ties broken by index.
    public func topK(query: [Float], vectors: [Float], count: Int, dim: Int, k: Int) throws -> [VectorScanHit] {
        guard dim > 0 else { throw LodeDBError.invalidArgument("dim must be positive") }
        guard k > 0 else { throw LodeDBError.invalidArgument("k must be positive") }
        guard query.count == dim else {
            throw LodeDBError.invalidArgument("query length \(query.count) does not match dim \(dim)")
        }
        guard vectors.count == count * dim else {
            throw LodeDBError.invalidArgument("vectors length \(vectors.count) does not match count * dim")
        }
        if count == 0 { return [] }
        let scores: [Float]
        if let metal {
            scores = try metal.dotScores(query: query, vectors: vectors, count: count, dim: dim)
        } else {
            scores = MetalVectorScanner.cpuScores(query: query, vectors: vectors, count: count, dim: dim)
        }
        return MetalVectorScanner.topK(fromScores: scores, k: k)
    }

    static func cpuScores(query: [Float], vectors: [Float], count: Int, dim: Int) -> [Float] {
        var scores = [Float](repeating: 0, count: count)
        query.withUnsafeBufferPointer { queryBuffer in
            vectors.withUnsafeBufferPointer { vectorBuffer in
                for row in 0..<count {
                    var accumulator: Float = 0
                    let base = row * dim
                    for component in 0..<dim {
                        accumulator += queryBuffer[component] * vectorBuffer[base + component]
                    }
                    scores[row] = accumulator
                }
            }
        }
        return scores
    }

    static func topK(fromScores scores: [Float], k: Int) -> [VectorScanHit] {
        var hits = [VectorScanHit]()
        hits.reserveCapacity(scores.count)
        for (index, score) in scores.enumerated() {
            hits.append(VectorScanHit(index: index, score: score))
        }
        hits.sort { left, right in
            left.score == right.score ? left.index < right.index : left.score > right.score
        }
        if hits.count > k {
            hits.removeLast(hits.count - k)
        }
        return hits
    }
}

#if canImport(Metal)
import Metal

/// Holds the Metal device, pipeline, and queue for the dot-product kernel.
private final class MetalContext {
    let deviceName: String
    private let device: MTLDevice
    private let pipeline: MTLComputePipelineState
    private let queue: MTLCommandQueue

    private static let kernelSource = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void lodedb_dot_scores(
        device const float* query   [[buffer(0)]],
        device const float* vectors [[buffer(1)]],
        device float* scores        [[buffer(2)]],
        constant uint& dim          [[buffer(3)]],
        uint row [[thread_position_in_grid]])
    {
        float acc = 0.0;
        uint base = row * dim;
        for (uint j = 0; j < dim; j++) {
            acc += query[j] * vectors[base + j];
        }
        scores[row] = acc;
    }
    """

    static func makeDefault() -> MetalContext? {
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        return try? MetalContext(device: device)
    }

    init(device: MTLDevice) throws {
        self.device = device
        self.deviceName = device.name
        let library = try device.makeLibrary(source: MetalContext.kernelSource, options: nil)
        guard let function = library.makeFunction(name: "lodedb_dot_scores") else {
            throw LodeDBError.internalError("Metal kernel function not found")
        }
        self.pipeline = try device.makeComputePipelineState(function: function)
        guard let queue = device.makeCommandQueue() else {
            throw LodeDBError.internalError("could not create a Metal command queue")
        }
        self.queue = queue
    }

    func dotScores(query: [Float], vectors: [Float], count: Int, dim: Int) throws -> [Float] {
        let floatSize = MemoryLayout<Float>.stride
        guard
            let queryBuffer = device.makeBuffer(bytes: query, length: dim * floatSize, options: .storageModeShared),
            let vectorBuffer = device.makeBuffer(bytes: vectors, length: count * dim * floatSize, options: .storageModeShared),
            let scoreBuffer = device.makeBuffer(length: count * floatSize, options: .storageModeShared)
        else {
            throw LodeDBError.internalError("could not allocate Metal buffers")
        }
        guard
            let commandBuffer = queue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw LodeDBError.internalError("could not create a Metal command encoder")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(queryBuffer, offset: 0, index: 0)
        encoder.setBuffer(vectorBuffer, offset: 0, index: 1)
        encoder.setBuffer(scoreBuffer, offset: 0, index: 2)
        var dimensions = UInt32(dim)
        encoder.setBytes(&dimensions, length: MemoryLayout<UInt32>.stride, index: 3)

        let threadsPerThreadgroup = min(pipeline.maxTotalThreadsPerThreadgroup, count)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: max(threadsPerThreadgroup, 1), height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let pointer = scoreBuffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }
}
#else
private final class MetalContext {
    let deviceName: String = ""
    static func makeDefault() -> MetalContext? { nil }
}
#endif
