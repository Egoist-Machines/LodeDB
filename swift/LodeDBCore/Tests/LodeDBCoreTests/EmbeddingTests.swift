import Foundation
import Testing

@testable import LodeDBCore

@Test func embeddingPoolingMatchesContract() throws {
    let tokens: [[Float]] = [[1, 0], [0, 2]]

    // Mean pooling weights by the attention mask and divides by max(maskSum, 1).
    #expect(try EmbeddingMath.pool(tokens, attentionMask: [1, 1], pooling: .mean) == [0.5, 1.0])
    // A masked-out token is excluded.
    #expect(try EmbeddingMath.pool(tokens, attentionMask: [1, 0], pooling: .mean) == [1.0, 0.0])
    // All-masked: denominator floored at 1, so the zero sum stays a zero vector.
    #expect(try EmbeddingMath.pool(tokens, attentionMask: [0, 0], pooling: .mean) == [0.0, 0.0])
    // CLS pooling takes the first token.
    #expect(try EmbeddingMath.pool(tokens, attentionMask: [1, 1], pooling: .cls) == [1.0, 0.0])

    let normalized = EmbeddingMath.l2Normalize([3, 4])
    #expect(abs(normalized[0] - 0.6) < 1e-6)
    #expect(abs(normalized[1] - 0.8) < 1e-6)
    // A zero vector is preserved (no division by zero).
    #expect(EmbeddingMath.l2Normalize([0, 0]) == [0, 0])
}

@Test func onnxTextEmbedderAppliesContractAndPrefix() throws {
    let tokenizer = RecordingTokenizer()
    let session = FixedSession(tokenEmbeddings: [[1, 0, 0, 0], [0, 1, 0, 0]])
    let preset = EmbeddingPreset(name: "test", dimension: 4, queryPrefix: "Q: ", modelIdentity: "test/model")

    let documentEmbedder = ONNXTextEmbedder(preset: preset, tokenizer: tokenizer, session: session, role: .document)
    let vectors = try documentEmbedder.embed(texts: ["hello"])
    #expect(vectors.count == 1)
    #expect(vectors[0] == EmbeddingMath.l2Normalize([0.5, 0.5, 0, 0]))
    #expect(tokenizer.lastText == "hello") // no document prefix
    #expect(documentEmbedder.modelIdentity == "test/model")

    let queryEmbedder = ONNXTextEmbedder(preset: preset, tokenizer: tokenizer, session: session, role: .query)
    _ = try queryEmbedder.embed(texts: ["hello"])
    #expect(tokenizer.lastText == "Q: hello") // query prefix applied
    #expect(queryEmbedder.dimension == 4)
}

@Test func onnxTextEmbedderRejectsDimensionMismatch() throws {
    let session = FixedSession(tokenEmbeddings: [[1, 0, 0]]) // dim 3 != preset 4
    let preset = EmbeddingPreset(name: "test", dimension: 4, modelIdentity: "test/model")
    let embedder = ONNXTextEmbedder(preset: preset, tokenizer: RecordingTokenizer(), session: session)
    #expect(throws: LodeDBError.self) {
        _ = try embedder.embed(texts: ["x"])
    }
}

@Test func presetsCarryTheExpectedContract() {
    #expect(EmbeddingPreset.miniLM.dimension == 384)
    #expect(EmbeddingPreset.miniLM.queryPrefix.isEmpty)
    #expect(EmbeddingPreset.bge.dimension == 768)
    #expect(EmbeddingPreset.bge.queryPrefix == "Represent this sentence for searching relevant passages: ")
}

@Test func nlEmbedderRoundTripsThroughLodeDB() throws {
    // Skip where the OS has no on-device sentence embedding model available.
    guard let embedder = try? NLEmbedder() else { return }
    #expect(embedder.dimension > 0)

    let db = try LodeDB(vectorDimension: embedder.dimension)
    try db.addText("The cat sat on the warm windowsill in the afternoon sun.", id: "cat", embedder: embedder)
    try db.addText("Quarterly revenue rose on strong enterprise software sales.", id: "revenue", embedder: embedder)

    let hits = try db.search(text: "a feline resting in the sunlight", k: 2, mode: .vector, embedder: embedder)
    #expect(hits.first?.id == "cat")
}

private final class RecordingTokenizer: TextTokenizer, @unchecked Sendable {
    private(set) var lastText: String?

    func encode(_ text: String, maxLength: Int) throws -> TokenizedText {
        lastText = text
        return TokenizedText(inputIDs: [1, 2], attentionMask: [1, 1])
    }
}

private struct FixedSession: EmbeddingModelSession {
    let tokenEmbeddings: [[Float]]

    func run(_ input: TokenizedText) throws -> [[Float]] {
        tokenEmbeddings
    }
}
