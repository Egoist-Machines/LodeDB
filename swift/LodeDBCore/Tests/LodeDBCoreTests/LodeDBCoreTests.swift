import Foundation
import Testing

@testable import LodeDBCore

@Test func vectorOnlySearchRanksAndFilters() throws {
    let db = try LodeDB(vectorDimension: 3)
    try db.addVector([1, 0, 0], id: "a", metadata: ["topic": "ops"])
    try db.addVector([0, 1, 0], id: "b", metadata: ["topic": "ml"])

    let hits = try db.search(vector: [0, 1, 0], k: 2)
    #expect(hits.map(\.id) == ["b", "a"])

    let filtered = try db.search(vector: [1, 0, 0], k: 2, filter: MetadataFilter(["topic": "ops"]))
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
    let db = try LodeDB(vectorDimension: 3)
    let embedder = HashTestEmbedder(dimension: 3)
    let text = "Alpha launch notes mention error code E-1001 and a blue widget."
    let plan = try db.prepareTextUpsert(text, id: "doc-alpha", metadata: ["topic": "ops"])
    #expect(plan.chunks.first?.chunkID == "doc-alpha:6ed29ed824c2:0000")
    let embeddings = try embedder.embed(texts: plan.chunks.map(\.text))
    try db.applyTextUpsert(plan, embeddings: embeddings)

    let vectorHits = try db.search(text: "blue widget", k: 1, mode: .vector, embedder: embedder)
    #expect(vectorHits.first?.id == "doc-alpha")
}

@Test func swiftTextLexicalAndHybridSearchUsePayloadTokens() throws {
    let db = try LodeDB(vectorDimension: 3)
    let embedder = HashTestEmbedder(dimension: 3)
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

@Test func rustFFIPrepareApplyTextProtocolMatchesSwiftBinding() throws {
    guard let dylib = ProcessInfo.processInfo.environment["LODEDB_FFI_DYLIB"] else {
        return
    }

    let library = try NativeCoreLibrary(path: dylib)
    #expect(library.abiVersion() == 1)

    let core = try NativeTextCore(library: library, vectorDimension: 2)
    let documentsJSON = """
    [{"document_id":"doc-text","text":"Alpha launch notes mention error code E-1001.","metadata":{"topic":"ops"}}]
    """
    let planJSON = try core.prepareTextUpsertJSON(
        documentsJSON,
        storeText: true,
        indexText: true,
        chunkCharacterLimit: 900
    )
    #expect(planJSON.contains(#""chunk_id":"doc-text:d9041255442c:0000""#))
    #expect(planJSON.contains(#""chunks_to_embed""#))

    let resultJSON = try core.applyTextUpsertJSON(
        planJSON: planJSON,
        embeddingsJSON: "[[1.0,0.0]]",
        embeddingTimeMS: 2.5
    )
    #expect(resultJSON.contains(#""embedded_chunks":1"#))
    #expect(resultJSON.contains(#""embedding_time_ms":2.5"#))

    let hits = try core.queryVector([1, 0], k: 1)
    #expect(hits.first == NativeSearchHit(id: "doc-text", chunkID: "doc-text:d9041255442c:0000", score: 1))
}

@Test func publicTextAddUsesNativePrepareApplyWhenConfigured() throws {
    guard ProcessInfo.processInfo.environment["LODEDB_FFI_DYLIB"] != nil else {
        return
    }

    let db = try LodeDB(vectorDimension: 2)
    #expect(db.nativeCoreEnabled)
    let embedder = FixedTestEmbedder(vector: [1, 0])
    try db.addText(
        "Alpha launch notes mention error code E-1001.",
        id: "doc-text",
        metadata: ["topic": "ops"],
        embedder: embedder
    )
    #expect(db.nativeVectorSearchReady)

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
