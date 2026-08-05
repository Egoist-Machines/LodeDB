import Foundation
import Testing

@testable import LodeDBCore

@Test func managedPullStagesAndRestoresWithoutNetwork() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("lodedb-swift-replica-\(UUID().uuidString)")
    let source = root.appendingPathComponent("source")
    let destination = root.appendingPathComponent("destination")
    let staging = root.appendingPathComponent("staging")
    try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    do {
        let db = try LodeDB(path: source, vectorDimension: 8)
        try db.addVector([1, 0, 0, 0, 0, 0, 0, 0], id: "restored-doc", metadata: ["kind": "test"])
        try db.persist()
        try db.close()
    }

    let replica = try LodeReplica()
    let remoteID = "orecloud://swift-test/default#host=https://example.test"
    let sourcePlan = try replica.managedPlan(
        directory: source,
        indexKey: "default",
        remoteID: remoteID,
        includeText: true,
        includeLexical: true
    )
    #expect(sourcePlan.classification == .localAhead)
    guard let local = sourcePlan.local else {
        throw LodeDBError.internalError("managed plan omitted the local generation")
    }
    #expect(!local.artifacts.isEmpty)

    // This mirrors cloud-core's stage_generation helper: the Swift HTTP edge names
    // each downloaded file by its manifest digest before native materialization.
    for artifact in local.artifacts {
        try FileManager.default.copyItem(
            at: source.appendingPathComponent(artifact.name),
            to: staging.appendingPathComponent(artifact.sha256)
        )
    }

    let requirements = try replica.pullRequirements(
        directory: destination,
        indexKey: "default",
        body: local.bodyJSON
    )
    #expect(Set(requirements.map(\.sha256)) == Set(local.artifacts.map(\.sha256)))

    let outcome = try replica.materialize(
        directory: destination,
        indexKey: "default",
        remoteID: remoteID,
        body: local.bodyJSON,
        stagingDirectory: staging
    )
    #expect(outcome.transfer.pointerPublished)
    #expect(outcome.open.documentCount == 1)

    let restored = try LodeDB.openReadOnly(path: destination)
    let hits = try restored.search(vector: [1, 0, 0, 0, 0, 0, 0, 0], k: 1)
    #expect(hits.map(\.id) == ["restored-doc"])
    try restored.close()
}
