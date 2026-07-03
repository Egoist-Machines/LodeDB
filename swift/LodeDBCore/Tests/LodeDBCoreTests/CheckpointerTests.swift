import Foundation
import Testing

@testable import LodeDBCore

@Test func checkpointerFoldsAppendsIntoTheCommittedGeneration() throws {
    let store = FileManager.default.temporaryDirectory
        .appendingPathComponent("lodedb-checkpointer-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: store, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: store) }

    // A writer creates the WAL store + index, then closes (releasing the writer lock)
    // so the checkpointer and appender can open.
    try LodeDB(path: store, vectorDimension: 8).close()

    // Open the checkpointer (folds nothing yet: the WAL is empty), then an appender
    // logs a record; checkpoint() folds it into a fresh committed generation while the
    // appender stays open, with no writable reopen. A second fold finds nothing new.
    do {
        let checkpointer = try LodeCheckpointer.open(at: store)
        let appender = try LodeAppender.open(at: store)
        let lsn = try appender.append(
            id: "fresh", vector: [1, 0, 0, 0, 0, 0, 0, 0], metadata: ["topic": "ops"])
        #expect(lsn > 0)
        #expect(try checkpointer.checkpoint() >= 1)
        #expect(try checkpointer.checkpoint() == 0)
    }

    // A read-only open reads only the committed base (it does not replay the WAL):
    // "fresh" is present because the checkpointer folded it, not because a writable
    // open did.
    let reader = try LodeDB.openReadOnly(path: store)
    #expect(reader.count == 1)
    let hits = try reader.search(vector: [1, 0, 0, 0, 0, 0, 0, 0], k: 1)
    #expect(hits.map(\.id) == ["fresh"])
    #expect(hits.first?.metadata["topic"] == "ops")
    try reader.close()
}
