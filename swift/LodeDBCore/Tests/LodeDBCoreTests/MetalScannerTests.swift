import Foundation
import Testing

@testable import LodeDBCore

@Test func cpuScannerReturnsTopKByDotProduct() throws {
    let scanner = MetalVectorScanner(preferMetal: false)
    #expect(scanner.backend == .cpu)

    let vectors: [Float] = [
        1, 0, 0, 0, // index 0  -> dot 1.0
        0, 1, 0, 0, // index 1  -> dot 0.0
        0.5, 0.5, 0, 0, // index 2 -> dot 0.5
    ]
    let hits = try scanner.topK(query: [1, 0, 0, 0], vectors: vectors, count: 3, dim: 4, k: 2)
    #expect(hits.map(\.index) == [0, 2])
    #expect(abs(hits[0].score - 1.0) < 1e-6)
    #expect(abs(hits[1].score - 0.5) < 1e-6)
}

@Test func vectorScannerValidatesInput() throws {
    let scanner = MetalVectorScanner(preferMetal: false)
    #expect(throws: LodeDBError.self) {
        _ = try scanner.topK(query: [1, 0], vectors: [1, 0, 0, 0], count: 1, dim: 4, k: 1) // query length != dim
    }
    #expect(throws: LodeDBError.self) {
        _ = try scanner.topK(query: [1, 0, 0, 0], vectors: [1, 0, 0], count: 1, dim: 4, k: 1) // vectors length != count*dim
    }
    // An empty corpus returns no hits rather than erroring.
    #expect(try scanner.topK(query: [1, 0, 0, 0], vectors: [], count: 0, dim: 4, k: 5).isEmpty)
}

@Test func metalScannerScoresMatchCPUReference() throws {
    // Skip where no Metal device is present (e.g. some headless CI).
    guard MetalVectorScanner.isMetalAvailable else { return }
    let scanner = MetalVectorScanner(preferMetal: true)
    #expect(scanner.backend == .metal)

    let count = 256
    let dimension = 16
    var vectors = [Float](repeating: 0, count: count * dimension)
    for i in 0..<vectors.count {
        vectors[i] = Float((i * 131 + 7) % 251) / 251.0 - 0.5 // deterministic, spread
    }
    var query = [Float](repeating: 0, count: dimension)
    for j in 0..<dimension {
        query[j] = Float((j * 97 + 3) % 251) / 251.0 - 0.5
    }

    let reference = MetalVectorScanner.cpuScores(query: query, vectors: vectors, count: count, dim: dimension)
    let hits = try scanner.topK(query: query, vectors: vectors, count: count, dim: dimension, k: count)
    #expect(hits.count == count)
    // Every GPU score matches the CPU dot product for that row.
    for hit in hits {
        #expect(abs(hit.score - reference[hit.index]) < 1e-3)
    }
    // The GPU's top score matches the true maximum (robust to float-order ties).
    #expect(abs((hits.first?.score ?? 0) - (reference.max() ?? 0)) < 1e-3)
}
