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
