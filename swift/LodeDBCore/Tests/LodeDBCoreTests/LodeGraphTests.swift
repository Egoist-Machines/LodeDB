import Foundation
import XCTest

import LodeDBCore

/// A deterministic dim-8 embedder (bucket bytes into 8 bins, L2-normalize) so the
/// suite is offline — similar text yields similar vectors, which is all these
/// membership assertions need.
private struct HashEmbedder: LodeEmbedder {
    let dimension = 8
    func embed(texts: [String]) throws -> [[Float]] {
        texts.map { text in
            var v = [Float](repeating: 0, count: 8)
            for byte in Array(text.lowercased().utf8) { v[Int(byte) % 8] += 1 }
            let norm = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
            if norm > 0 { return v.map { $0 / norm } }
            var zero = [Float](repeating: 0, count: 8); zero[0] = 1; return zero
        }
    }
}

/// End-to-end tests for the on-device `LodeGraph`, exercising the bi-temporal
/// invariants through the native `lodedb-graph` core (vector-in, embedded on device).
final class LodeGraphTests: XCTestCase {
    private func graph() throws -> LodeGraph { try LodeGraph(embedder: HashEmbedder()) }

    func testInvalidationPreservesHistoryAndAsOf() throws {
        let g = try graph()
        _ = try g.upsertEntity(id: "alice", type: "Person", label: "Alice, engineer")
        _ = try g.upsertEntity(id: "acme", type: "Org", label: "Acme Corp")
        _ = try g.upsertEntity(id: "globex", type: "Org", label: "Globex Corp")

        let fAcme = try g.addFact(src: "alice", relation: "works_at", dst: "acme",
                                  fact: "Alice works at Acme", validAt: 1000)
        let fGlobex = try g.addFact(src: "alice", relation: "works_at", dst: "globex",
                                    fact: "Alice works at Globex", validAt: 2000,
                                    invalidates: [fAcme])

        XCTAssertEqual(try g.neighbors(id: "alice", relation: "works_at").map(\.dst), ["globex"])
        XCTAssertEqual(try g.neighbors(id: "alice", relation: "works_at", asOf: .at(1500)).map(\.dst), ["acme"])
        XCTAssertEqual(try g.neighbors(id: "alice", relation: "works_at", asOf: .at(2500)).map(\.dst), ["globex"])

        let hist = try g.history(entityID: "alice")
        XCTAssertEqual(hist.count, 2, "both assertions preserved")
        let acme = try XCTUnwrap(hist.first { $0.id == fAcme })
        XCTAssertEqual(acme.invalidAt, 2000)
        XCTAssertNotNil(acme.expiredAt)
        let globex = try XCTUnwrap(try g.getFact(fGlobex))
        XCTAssertEqual(
            try g.neighbors(
                id: "alice", relation: "works_at",
                asOf: .atKnown(validAt: 2500, knownAt: try XCTUnwrap(acme.expiredAt) - 1)
            ).map(\.id),
            [fAcme]
        )
        XCTAssertEqual(
            try g.neighbors(
                id: "alice", relation: "works_at",
                asOf: .atKnown(validAt: 2500, knownAt: globex.createdAt)
            ).map(\.id),
            [fGlobex]
        )
    }

    func testKHopTraversal() throws {
        let g = try graph()
        for (id, label) in [("a", "node a"), ("b", "node b"), ("c", "node c"), ("d", "node d")] {
            _ = try g.upsertEntity(id: id, type: "Thing", label: label)
        }
        _ = try g.addFact(src: "a", relation: "rel", dst: "b", fact: "a rel b", validAt: 1)
        _ = try g.addFact(src: "b", relation: "rel", dst: "c", fact: "b rel c", validAt: 1)
        _ = try g.addFact(src: "c", relation: "rel", dst: "d", fact: "c rel d", validAt: 1)

        let one = try g.kHop(seeds: ["a"], k: 1, direction: "out")
        let ids1 = Set(one.entities.map(\.id))
        XCTAssertTrue(ids1.isSuperset(of: ["a", "b"]))
        XCTAssertFalse(ids1.contains("c"))

        let two = try g.kHop(seeds: ["a"], k: 2, direction: "out")
        let ids2 = Set(two.entities.map(\.id))
        XCTAssertTrue(ids2.contains("c"))
        XCTAssertFalse(ids2.contains("d"))
    }

    func testEnumerateAndSearch() throws {
        let g = try graph()
        _ = try g.upsertEntity(id: "alice", type: "Person", label: "Alice builds robots")
        _ = try g.upsertEntity(id: "acme", type: "Org", label: "Acme robotics company")
        _ = try g.upsertEntity(id: "nyc", type: "Place", label: "New York City")

        XCTAssertEqual(try g.entities(type: "Person").map(\.id), ["alice"])
        XCTAssertEqual(try g.entities().count, 3)
        XCTAssertFalse(try g.semanticEntities("robots", k: 5).isEmpty)
        XCTAssertEqual(try g.stats().entities, 3)
    }

    func testEpisodeReferenceTime() throws {
        let g = try graph()
        _ = try g.upsertEntity(id: "p", type: "Person", label: "Pat")
        _ = try g.upsertEntity(id: "q", type: "Org", label: "QCo")
        let ep = try g.addEpisode(source: "note", body: "Pat joined QCo", occurredAt: 4242, mentions: ["p"])
        let fid = try g.addFact(src: "p", relation: "works_at", dst: "q", fact: "Pat works at QCo",
                                episodes: [ep], validAt: 4242)
        XCTAssertEqual(try g.getFact(fid)?.referenceTime, 4242)
    }

    func testStrictNowAndStableEpisodeRollback() throws {
        let g = try graph()
        _ = try g.upsertEntity(id: "a", type: "Thing", label: "future source")
        _ = try g.upsertEntity(id: "b", type: "Thing", label: "future target")
        let now = Int64(Date().timeIntervalSince1970 * 1000)
        _ = try g.addFact(src: "a", relation: "rel", dst: "b",
                          fact: "future relation", validAt: now + 86_400_000)
        XCTAssertEqual(try g.neighbors(id: "a").count, 1)
        XCTAssertTrue(try g.neighbors(id: "a", asOf: .nowValid).isEmpty)

        let episode = try g.addEpisode(source: "note", body: "stable", occurredAt: now,
                                       id: "stable-episode")
        XCTAssertEqual(
            try g.addEpisode(source: "note", body: "stable", occurredAt: now,
                             id: "stable-episode"),
            episode
        )
        let episodeFact = try g.addFact(
            src: "a", relation: "from_episode", dst: "b", fact: "stable derivation",
            episodes: [episode], id: "stable-fact"
        )
        XCTAssertEqual(
            try g.addFact(
                src: "a", relation: "from_episode", dst: "b", fact: "stable derivation",
                episodes: [episode], id: "stable-fact"
            ),
            episodeFact
        )
        XCTAssertTrue(try g.episodes().contains { $0.id == episode })
        XCTAssertEqual(try g.factsByEpisode(episode).map(\.id), [episodeFact])
        XCTAssertTrue(try g.removeEpisode(episode))
        XCTAssertNil(try g.getEpisode(episode))
        XCTAssertNil(try g.getFact(episodeFact))

        let source = try g.addEpisode(
            source: "event", body: "activated", occurredAt: now, id: "activation"
        )
        _ = try g.upsertEntity(
            id: "a", type: "Thing", label: "future source",
            properties: ["owner": "u1", "status": "new"]
        )
        _ = try g.upsertEntity(
            id: "a", type: "Thing", label: "future source",
            properties: ["owner": "u1", "status": "active"],
            propertySources: ["status": source]
        )
        let status = try g.entityPropertyHistory("a", key: "status")
        XCTAssertEqual(status.map(\.value), [.string("new"), .string("active")])
        XCTAssertEqual(status.last?.episodeID, source)
        XCTAssertEqual(
            try g.semanticEntities(
                "future source", predicate: .equals("owner", "u1")
            ).map(\.entity.id),
            ["a"]
        )
    }
}
