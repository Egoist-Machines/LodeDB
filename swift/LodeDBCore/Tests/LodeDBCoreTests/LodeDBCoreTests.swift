import Foundation
import Testing

@testable import LodeDBCore

@Test func vectorOnlySearchRanksAndFilters() throws {
    let db = try LodeDB(vectorDimension: 8)
    try db.addVector([1, 0, 0, 0, 0, 0, 0, 0], id: "a", metadata: ["topic": "ops"])
    try db.addVector([0, 1, 0, 0, 0, 0, 0, 0], id: "b", metadata: ["topic": "ml"])
    #expect(db.count == 2)

    let hits = try db.search(vector: [0, 1, 0, 0, 0, 0, 0, 0], k: 2)
    #expect(hits.map(\.id) == ["b", "a"])
    #expect(hits.first?.metadata["topic"] == "ml")

    let filtered = try db.search(
        vector: [1, 0, 0, 0, 0, 0, 0, 0],
        k: 2,
        filter: MetadataFilter(["topic": "ops"])
    )
    #expect(filtered.map(\.id) == ["a"])
}

@Test func opensPythonGenerationFixtureReadOnly() throws {
    let package = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let fixture = package
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("tests/fixtures/persisted/v0_4_generation")
    let db = try LodeDB.openReadOnly(path: fixture)
    #expect(db.count == 3)
}

@Test func textPrepareApplyUsesEmbedderAndMatchesChunkIDFixture() throws {
    let db = try LodeDB(vectorDimension: 8)
    let embedder = HashTestEmbedder(dimension: 8)
    let text = "Alpha launch notes mention error code E-1001 and a blue widget."
    let plan = try db.prepareTextUpsert(text, id: "doc-alpha", metadata: ["topic": "ops"])
    #expect(plan.chunks.first?.chunkID == "doc-alpha:6ed29ed824c2:0000")
    let embeddings = try embedder.embed(texts: plan.chunks.map(\.text))
    try db.applyTextUpsert(plan, embeddings: embeddings)
    #expect(db.count == 1)

    let vectorHits = try db.search(text: "blue widget", k: 1, mode: .vector, embedder: embedder)
    #expect(vectorHits.first?.id == "doc-alpha")
}

@Test func textLexicalAndHybridSearchUsePayloadTokens() throws {
    let db = try LodeDB(vectorDimension: 8)
    let embedder = HashTestEmbedder(dimension: 8)
    try db.addText(
        "Beta incident report for serial AX-42 on 2024-06-13.",
        id: "doc-beta",
        embedder: embedder
    )
    try db.addText(
        "Gamma handbook explains offline vector search and local recovery.",
        id: "doc-gamma",
        embedder: embedder
    )

    let lexical = try db.search(text: "serial AX-42", k: 1, mode: .lexical)
    #expect(lexical.map(\.id) == ["doc-beta"])

    let hybrid = try db.search(text: "local recovery", k: 2, mode: .hybrid, embedder: embedder)
    #expect(hybrid.contains { $0.id == "doc-gamma" })
}

@Test func metadataFilterIsAppliedNativelyBeforeTopK() throws {
    let db = try LodeDB(vectorDimension: 8)
    // The best vector match is excluded by the filter, so a correct native filter
    // (applied before top-k) must still return the lower-scoring in-filter doc.
    try db.addVector([1, 0, 0, 0, 0, 0, 0, 0], id: "best", metadata: ["topic": "ml"])
    try db.addVector([0.6, 0.4, 0, 0, 0, 0, 0, 0], id: "second", metadata: ["topic": "ops"])

    let hits = try db.search(
        vector: [1, 0, 0, 0, 0, 0, 0, 0],
        k: 1,
        filter: MetadataFilter(["topic": "ops"])
    )
    #expect(hits.map(\.id) == ["second"])
}

@Test func nativeEngineExposesABIVersionAndFFITextProtocol() throws {
    #expect(NativeEngine.abiVersion() == 1)

    let engine = try NativeEngine(vectorDimension: 8)
    let documentsJSON = """
    [{"document_id":"doc-text","text":"Alpha launch notes mention error code E-1001.","metadata":{"topic":"ops"}}]
    """
    let planJSON = try engine.prepareTextUpsertJSON(
        documentsJSON,
        storeText: true,
        indexText: true,
        chunkCharacterLimit: 900
    )
    #expect(planJSON.contains(#""chunk_id":"doc-text:d9041255442c:0000""#))
    #expect(planJSON.contains(#""chunks_to_embed""#))

    let resultJSON = try engine.applyTextUpsertJSON(
        planJSON: planJSON,
        embeddingsJSON: "[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]]",
        embeddingTimeMS: 2.5
    )
    #expect(resultJSON.contains(#""embedded_chunks":1"#))
    #expect(resultJSON.contains(#""embedding_time_ms":2.5"#))

    let vectorJSON = try engine.queryVectorJSON([1, 0, 0, 0, 0, 0, 0, 0], k: 1, filterJSON: nil)
    #expect(vectorJSON.contains(#""document_id":"doc-text""#))
    #expect(vectorJSON.contains(#""chunk_id":"doc-text:d9041255442c:0000""#))

    let lexicalPlan = try engine.prepareQueryTextJSON("E-1001", mode: "lexical")
    let lexicalHits = try engine.searchEmbeddedTextJSON(
        queryPlanJSON: lexicalPlan,
        queryEmbeddingJSON: nil,
        k: 1,
        filterJSON: #"{"metadata":{"topic":"ops"}}"#
    )
    #expect(lexicalHits.contains(#""document_id":"doc-text""#))
}

@Test func publicTextAddRoundTripsThroughNativeCore() throws {
    let db = try LodeDB(vectorDimension: 8)
    let embedder = FixedTestEmbedder(vector: [1, 0, 0, 0, 0, 0, 0, 0])
    try db.addText(
        "Alpha launch notes mention error code E-1001.",
        id: "doc-text",
        metadata: ["topic": "ops"],
        embedder: embedder
    )
    #expect(db.count == 1)

    let lexical = try db.search(text: "error code", k: 1, mode: .lexical)
    #expect(lexical.first?.id == "doc-text")
    #expect(lexical.first?.chunkID == "doc-text:d9041255442c:0000")

    let vector = try db.search(text: "anything", k: 1, mode: .vector, embedder: embedder)
    #expect(vector.first?.id == "doc-text")
}

private struct HashTestEmbedder: LodeEmbedder {
    let dimension: Int

    func embed(texts: [String]) throws -> [[Float]] {
        texts.map { text in
            var vector = Array(repeating: Float(0), count: dimension)
            let bucket = abs(text.hashValue) % dimension
            vector[bucket] = 1
            return vector
        }
    }
}

private struct FixedTestEmbedder: LodeEmbedder {
    let vector: [Float]

    var dimension: Int {
        vector.count
    }

    func embed(texts: [String]) throws -> [[Float]] {
        texts.map { _ in vector }
    }
}
