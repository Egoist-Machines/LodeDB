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

    let engine = try NativeEngine.inMemory(vectorDimension: 8)
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

@Test func durableStoreRoundTripsThroughNativeOpenPersistReopen() throws {
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("lodedb-swift-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: dir) }
    let embedder = FixedTestEmbedder(vector: [0, 1, 0, 0, 0, 0, 0, 0])

    do {
        let db = try LodeDB(path: dir, vectorDimension: 8)
        try db.addVector([0, 1, 0, 0, 0, 0, 0, 0], id: "vec-1", metadata: ["topic": "ops"])
        try db.addText("durable notes about a blue widget", id: "txt-1", metadata: ["topic": "docs"], embedder: embedder)
        #expect(db.count == 2)
        try db.persist()
        try db.close()
    }

    // Reopen the same on-disk store read-only and confirm it serves real native search.
    let reopened = try LodeDB.openReadOnly(path: dir)
    #expect(reopened.count == 2)
    let hits = try reopened.search(vector: [0, 1, 0, 0, 0, 0, 0, 0], k: 2)
    #expect(hits.contains { $0.id == "vec-1" })
    #expect(try reopened.get("txt-1") == "durable notes about a blue widget")
    #expect(try reopened.getDocument("vec-1")?.metadata["topic"] == "ops")
}

@Test func crudRemoveGetListUpdateRunNatively() throws {
    let db = try LodeDB(vectorDimension: 8)
    try db.addVector([1, 0, 0, 0, 0, 0, 0, 0], id: "a", metadata: ["k": "1"])
    try db.addVector([0, 1, 0, 0, 0, 0, 0, 0], id: "b", metadata: ["k": "2"])
    #expect(db.count == 2)
    #expect(try db.collections() == ["default"])

    #expect(Set(try db.listDocuments().map(\.id)) == ["a", "b"])
    #expect(try db.getDocument("a")?.metadata["k"] == "1")

    try db.updateDocument(id: "a", metadata: ["k": "updated"])
    #expect(try db.getDocument("a")?.metadata["k"] == "updated")

    #expect(try db.remove("b") == true)
    #expect(try db.remove("missing") == false)
    #expect(db.count == 1)
    #expect(try db.getDocument("b") == nil)
}

@Test func closedStoreRejectsFurtherOperations() throws {
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("lodedb-swift-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: dir) }

    let db = try LodeDB(path: dir, vectorDimension: 8)
    try db.addVector([1, 0, 0, 0, 0, 0, 0, 0], id: "a")
    try db.close()
    try db.close() // idempotent

    // After close, writes must be rejected rather than silently lost to an in-memory copy.
    #expect(throws: LodeDBError.self) {
        try db.addVector([0, 1, 0, 0, 0, 0, 0, 0], id: "b")
    }
    #expect(throws: LodeDBError.self) {
        _ = try db.search(vector: [1, 0, 0, 0, 0, 0, 0, 0], k: 1)
    }
    // The post-close write must not have reached disk.
    let reopened = try LodeDB.openReadOnly(path: dir)
    #expect(reopened.count == 1)
}

@Test func listDocumentsRejectsNegativeLimitWithoutCrashing() throws {
    let db = try LodeDB(vectorDimension: 8)
    try db.addVector([1, 0, 0, 0, 0, 0, 0, 0], id: "a")
    #expect(throws: LodeDBError.self) {
        _ = try db.listDocuments(limit: -1)
    }
}

@Test func textUpdateClearAndSetRoundTrip() throws {
    let db = try LodeDB(vectorDimension: 8)
    let embedder = FixedTestEmbedder(vector: [1, 0, 0, 0, 0, 0, 0, 0])
    try db.addText("original retained text", id: "doc", embedder: embedder)
    #expect(try db.get("doc") == "original retained text")

    try db.updateDocument(id: "doc", text: .set("replacement text"))
    #expect(try db.get("doc") == "replacement text")

    try db.updateDocument(id: "doc", text: .clear)
    #expect(try db.get("doc") == nil)
}

@Test func readOnlyOpenRejectsMutations() throws {
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("lodedb-swift-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: dir) }
    do {
        let writable = try LodeDB(path: dir, vectorDimension: 8)
        try writable.addVector([1, 0, 0, 0, 0, 0, 0, 0], id: "a")
        try writable.persist()
        try writable.close()
    }

    let readOnly = try LodeDB.openReadOnly(path: dir)
    #expect(throws: LodeDBError.self) {
        try readOnly.addVector([0, 1, 0, 0, 0, 0, 0, 0], id: "b")
    }
    #expect(throws: LodeDBError.self) {
        _ = try readOnly.remove("a")
    }
}

@Test func statsReportNativeMetrics() throws {
    let db = try LodeDB(vectorDimension: 8)
    try db.addVector([1, 0, 0, 0, 0, 0, 0, 0], id: "a")
    let stats = try db.stats()
    #expect(stats.documentCount == 1)
    #expect(stats.vectorDimension == 8)
    #expect(!stats.nativeCoreVersion.isEmpty)
}

@Test func mongoStyleFiltersResolveNatively() throws {
    let db = try LodeDB(vectorDimension: 8)
    try db.addVector(unitVector(0), id: "a", metadata: ["topic": "ml", "year": "2020"])
    try db.addVector(unitVector(1), id: "b", metadata: ["topic": "ops", "year": "2018"])
    try db.addVector(unitVector(2), id: "c", metadata: ["topic": "ml", "year": "2022"])

    func ids(_ predicate: FilterPredicate) throws -> Set<String> {
        Set(try db.listDocuments(filter: MetadataFilter(predicate: predicate)).map(\.id))
    }
    #expect(try ids(.greaterThan("year", "2019")) == ["a", "c"])
    #expect(try ids(.inSet("topic", ["ops"])) == ["b"])
    #expect(try ids(.or([.equals("topic", "ops"), .greaterThanOrEqual("year", "2022")])) == ["b", "c"])
    #expect(try ids(.not(.equals("topic", "ml"))) == ["b"])
    #expect(try ids(.and([.equals("topic", "ml"), .lessThan("year", "2021")])) == ["a"])
    #expect(try ids(.exists("year", true)) == ["a", "b", "c"])

    let allowlisted = try db.listDocuments(
        filter: MetadataFilter(documentIDs: ["a", "b"], predicate: .equals("topic", "ml")))
    #expect(Set(allowlisted.map(\.id)) == ["a"])
}

@Test func searchManyReturnsPerQueryResults() throws {
    let db = try LodeDB(vectorDimension: 8)
    try db.addVector(unitVector(0), id: "a")
    try db.addVector(unitVector(1), id: "b")
    let results = try db.searchMany(vectors: [unitVector(0), unitVector(1)], k: 1)
    #expect(results.count == 2)
    #expect(results[0].first?.id == "a")
    #expect(results[1].first?.id == "b")
}

@Test func searchManyTextsRunLexicalPerQuery() throws {
    let db = try LodeDB(vectorDimension: 8)
    let embedder = HashTestEmbedder(dimension: 8)
    try db.addText("Beta incident report for serial AX-42.", id: "doc-beta", embedder: embedder)
    try db.addText("Gamma handbook on local recovery.", id: "doc-gamma", embedder: embedder)
    let results = try db.searchMany(texts: ["serial AX-42", "local recovery"], k: 1, mode: .lexical)
    #expect(results.count == 2)
    #expect(results[0].first?.id == "doc-beta")
    #expect(results[1].first?.id == "doc-gamma")
}

@Test func lateInteractionMaxSimRanksByPatchSimilarity() throws {
    let index = try LodeLateInteractionIndex(vectorDimension: 8)
    try index.addDocument(id: "doc-x", patches: [unitVector(0), unitVector(1)], metadata: ["k": "x"])
    try index.addDocument(id: "doc-y", patches: [unitVector(2), unitVector(3)])

    let hits = try index.search(queryPatches: [unitVector(0), unitVector(1)], k: 2)
    #expect(hits.first?.id == "doc-x")

    let filtered = try index.search(
        queryPatches: [unitVector(0)], k: 2, filter: MetadataFilter(documentIDs: ["doc-y"]))
    #expect(filtered.map(\.id) == ["doc-y"])
}

@Test func lateInteractionReplaceWithSameAnchorUpdatesPatches() throws {
    let index = try LodeLateInteractionIndex(vectorDimension: 8)
    try index.addDocument(id: "doc", patches: [unitVector(0), unitVector(1)])
    let before = try index.search(queryPatches: [unitVector(0)], k: 1).first
    #expect(before?.id == "doc")

    // Same mean-pooled anchor (0.5, 0.5, 0, ...) but different patches: the engine
    // must not treat this as a no-op and keep the stale MaxSim payload.
    let half: [Float] = [0.5, 0.5, 0, 0, 0, 0, 0, 0]
    try index.addDocument(id: "doc", patches: [half, half])
    let after = try index.search(queryPatches: [unitVector(0)], k: 1).first
    #expect(after?.id == "doc")
    #expect((after?.score ?? 0) < (before?.score ?? 0))
}

private func unitVector(_ index: Int, dimension: Int = 8) -> [Float] {
    var vector = Array(repeating: Float(0), count: dimension)
    vector[index] = 1
    return vector
}

@Test func concurrentSearchAndAddDoNotRaceTheNativeCore() throws {
    let db = try LodeDB(vectorDimension: 8)
    for index in 0..<8 {
        var vector = Array(repeating: Float(0), count: 8)
        vector[index] = 1
        try db.addVector(vector, id: "doc-\(index)", metadata: ["bucket": "\(index)"])
    }
    // Warm the lazily built native index before hammering it concurrently.
    _ = try db.search(vector: [1, 0, 0, 0, 0, 0, 0, 0], k: 3)

    let failures = Atomic(0)
    DispatchQueue.concurrentPerform(iterations: 64) { iteration in
        do {
            if iteration % 4 == 0 {
                var vector = Array(repeating: Float(0), count: 8)
                vector[iteration % 8] = 1
                try db.addVector(vector, id: "extra-\(iteration)", metadata: [:])
            } else {
                let hits = try db.search(vector: [0, 1, 0, 0, 0, 0, 0, 0], k: 2)
                if hits.isEmpty { failures.mutate { $0 += 1 } }
            }
        } catch {
            failures.mutate { $0 += 1 }
        }
    }
    #expect(failures.value == 0)
    #expect(db.count == 8 + 16) // 16 of the 64 iterations (every 4th) add a unique doc.
}

private final class Atomic<Value>: @unchecked Sendable {
    private var value_: Value
    private let lock = NSLock()
    init(_ value: Value) { self.value_ = value }
    var value: Value { lock.lock(); defer { lock.unlock() }; return value_ }
    func mutate(_ body: (inout Value) -> Void) { lock.lock(); defer { lock.unlock() }; body(&value_) }
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
