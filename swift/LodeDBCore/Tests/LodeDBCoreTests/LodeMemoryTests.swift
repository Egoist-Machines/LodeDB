import Foundation
import Testing

@testable import LodeDBCore

@Test func lodeMemorySaveRecallForget() throws {
    let memory = try LodeMemory(embedder: BucketEmbedder(dimension: 8))
    let runbookID = try memory.save(
        "The deployment runbook for the payments service.",
        metadata: ["topic": "ops"]
    )
    try memory.save("Chocolate cake recipe with vanilla frosting.", id: "recipe")
    #expect(memory.count == 2)

    let hits = try memory.recall("payments service runbook", k: 1, mode: .lexical)
    #expect(hits.first?.id == runbookID)
    #expect(hits.first?.text == "The deployment runbook for the payments service.")
    #expect(hits.first?.metadata["topic"] == "ops")

    #expect(try memory.forget("recipe") == true)
    #expect(try memory.forget("recipe") == false)
    #expect(memory.count == 1)
}

@Test func lodeMemoryRecallHonorsMetadataFilter() throws {
    let memory = try LodeMemory(embedder: BucketEmbedder(dimension: 8))
    try memory.save("incident review", id: "a", metadata: ["team": "sre"])
    try memory.save("incident review", id: "b", metadata: ["team": "ml"])

    let hits = try memory.recall(
        "incident review", k: 5, mode: .lexical, filter: MetadataFilter(["team": "sre"]))
    #expect(hits.map(\.id) == ["a"])
}

@Test func recallDeduplicatesMultiChunkMemories() throws {
    let memory = try LodeMemory(embedder: BucketEmbedder(dimension: 8))
    // Longer than the default chunk limit (8192 chars), so it is stored as several chunks.
    let longText = String(repeating: "alpha beta gamma delta epsilon ", count: 400)
    let id = try memory.save(longText)
    let hits = try memory.recall("alpha beta gamma", k: 5, mode: .lexical)
    // One memory, not one result slot per chunk.
    #expect(hits.map(\.id) == [id])
}

private struct BucketEmbedder: LodeEmbedder {
    let dimension: Int

    func embed(texts: [String]) throws -> [[Float]] {
        texts.map { text in
            var vector = Array(repeating: Float(0), count: dimension)
            vector[abs(text.hashValue) % dimension] = 1
            return vector
        }
    }
}
