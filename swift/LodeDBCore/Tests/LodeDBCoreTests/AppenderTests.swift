import Foundation
import Testing

@testable import LodeDBCore

@Test func appenderFoldsVectorsIntoNextWriter() throws {
    let store = FileManager.default.temporaryDirectory
        .appendingPathComponent("lodedb-appender-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: store, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: store) }

    // A writer creates the WAL store and its index, then closes (releasing the
    // writer lock) so the shared appender can open.
    let writer = try LodeDB(path: store, vectorDimension: 8)
    try writer.close()

    // The appender logs a single record, a batch record, and a delete, each
    // returning a strictly increasing LSN. Scoped so its shared lock releases
    // before the next writer opens.
    do {
        let appender = try LodeAppender.open(at: store)
        let first = try appender.append(
            id: "doc-a", vector: [1, 0, 0, 0, 0, 0, 0, 0], metadata: ["topic": "ops"])
        #expect(first > 0)
        let batch = try appender.append([
            LodeAppendDocument(id: "doc-b", vector: [0, 1, 0, 0, 0, 0, 0, 0])
        ])
        #expect(batch > first)
        let removed = try appender.delete(ids: ["doc-b"])
        #expect(removed > batch)
    }

    // The next writer folds every appended record in: doc-a survives (and is
    // queryable with its metadata), doc-b was appended then deleted.
    let reopened = try LodeDB(path: store, vectorDimension: 8)
    #expect(reopened.count == 1)
    let hits = try reopened.search(vector: [1, 0, 0, 0, 0, 0, 0, 0], k: 2)
    #expect(hits.map(\.id) == ["doc-a"])
    #expect(hits.first?.metadata["topic"] == "ops")
    try reopened.close()
}

@Test func readOnlyRefreshGivesReaderFreshnessAndReadYourWrites() throws {
    let store = FileManager.default.temporaryDirectory
        .appendingPathComponent("lodedb-refresh-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: store, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: store) }

    // A writer creates the WAL store + index, then closes so a reader/appender can open.
    try LodeDB(path: store, vectorDimension: 8).close()

    let reader = try LodeDB.openReadOnly(path: store)
    #expect(reader.count == 0)
    let baseLSN = try reader.appliedLSN()

    // A separate appender logs a vector-in record.
    var appendedLSN: UInt64 = 0
    do {
        let appender = try LodeAppender.open(at: store)
        appendedLSN = try appender.append(
            id: "fresh", vector: [1, 0, 0, 0, 0, 0, 0, 0], metadata: ["topic": "ops"])
    }

    // Still the snapshot: no refresh, no change.
    #expect(reader.count == 0)
    #expect(try reader.appliedLSN() == baseLSN)

    // Refresh overlays the WAL tail: the append is visible and applied LSN caught up.
    try reader.refresh()
    #expect(try reader.appliedLSN() >= appendedLSN)
    #expect(reader.count == 1)
    let hits = try reader.search(vector: [1, 0, 0, 0, 0, 0, 0, 0], k: 1)
    #expect(hits.map(\.id) == ["fresh"])
    try reader.close()
}

@Test func appenderOpenRequiresAnExistingIndex() throws {
    let empty = FileManager.default.temporaryDirectory
        .appendingPathComponent("lodedb-appender-empty-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: empty) }

    // No index at the path: the appender must fail closed rather than open.
    #expect(throws: (any Error).self) {
        _ = try LodeAppender.open(at: empty)
    }
}

@Test func appenderRetainsCaptionOnlyWithStoreText() throws {
    let store = FileManager.default.temporaryDirectory
        .appendingPathComponent("lodedb-appender-caption-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: store, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: store) }

    try LodeDB(path: store, vectorDimension: 8).close()

    // storeText on: the caption (e.g. an image's) survives the next writable open.
    do {
        let appender = try LodeAppender.open(at: store, storeText: true)
        _ = try appender.append(
            id: "img-1", vector: [1, 0, 0, 0, 0, 0, 0, 0], metadata: ["kind": "image"],
            text: "a red bicycle")
    }
    do {
        let reopened = try LodeDB(path: store, vectorDimension: 8)
        #expect(try reopened.get("img-1") == "a red bicycle")
        try reopened.close()
    }

    // Default (storeText off): the caption is not logged, so none is retained.
    do {
        let appender = try LodeAppender.open(at: store)
        _ = try appender.append(id: "vec-2", vector: [0, 1, 0, 0, 0, 0, 0, 0], text: "dropped")
    }
    do {
        let reopened = try LodeDB(path: store, vectorDimension: 8)
        #expect(try reopened.get("vec-2") == nil)
        try reopened.close()
    }
}

/// A tiny deterministic embedder: axis 1 for text containing "zebra", axis 0
/// otherwise. Enough to exercise the text-append path without a real model.
private struct KeywordEmbedder: LodeEmbedder {
    var dimension: Int { 8 }
    func embed(texts: [String]) throws -> [[Float]] {
        texts.map { text in
            var vector = [Float](repeating: 0, count: 8)
            vector[text.contains("zebra") ? 1 : 0] = 1
            return vector
        }
    }
}

@Test func appenderFoldsEmbeddedTextIntoNextWriter() throws {
    let store = FileManager.default.temporaryDirectory
        .appendingPathComponent("lodedb-appender-text-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: store, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: store) }

    let embedder = KeywordEmbedder()

    // A writer creates the text store (store_text/index_text on by default) with one
    // seed document, then closes so the shared appender can open.
    let writer = try LodeDB(path: store, vectorDimension: embedder.dimension)
    try writer.addText("a seed document about apples", id: "seed", embedder: embedder)
    try writer.close()

    // The appender chunks + embeds a text document and logs a post-embedding record.
    do {
        let appender = try LodeAppender.open(at: store, storeText: true, indexText: true)
        let lsn = try appender.append(
            text: "launch code report zebra", id: "doc-a", metadata: ["topic": "ops"],
            embedder: embedder)
        #expect(lsn > 0)
    }

    // The next writer folds the appended text in: both documents present, and the
    // appended raw text is retained (store_text on).
    let reopened = try LodeDB(path: store, vectorDimension: embedder.dimension)
    #expect(reopened.count == 2)
    #expect(try reopened.get("doc-a") == "launch code report zebra")
    try reopened.close()
}
