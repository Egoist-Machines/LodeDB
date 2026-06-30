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

    let embedder = ONNXTextEmbedder(preset: preset, tokenizer: tokenizer, session: session)

    // The bare embed(texts:) uses the document role (no prefix); mean pool of the two
    // tokens = [0.5, 0.5, 0, 0], then L2-normalized.
    let vectors = try embedder.embed(texts: ["hello"])
    #expect(vectors.count == 1)
    #expect(vectors[0] == EmbeddingMath.l2Normalize([0.5, 0.5, 0, 0]))
    #expect(tokenizer.lastText == "hello") // no document prefix
    #expect(embedder.modelIdentity == "test/model")
    #expect(embedder.dimension == 4)

    _ = try embedder.embed(texts: ["hello"], role: .query)
    #expect(tokenizer.lastText == "Q: hello") // query prefix applied
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
    #expect(EmbeddingPreset.miniLM.pooling == .mean)
    #expect(EmbeddingPreset.bge.dimension == 768)
    #expect(EmbeddingPreset.bge.queryPrefix == "Represent this sentence for searching relevant passages: ")
    // BGE is trained for CLS pooling (matches the Python preset's pooling="cls").
    #expect(EmbeddingPreset.bge.pooling == .cls)
}

@Test func clsPoolingSelectsTheFirstTokenNotTheMean() throws {
    // token 0 ([1,0]) differs from the mean ([0.5,0.5]); a CLS preset must pick token 0.
    let session = FixedSession(tokenEmbeddings: [[1, 0], [0, 1]])
    let preset = EmbeddingPreset(name: "cls", dimension: 2, pooling: .cls, modelIdentity: "m")
    let embedder = ONNXTextEmbedder(preset: preset, tokenizer: RecordingTokenizer(), session: session)
    #expect(try embedder.embed(texts: ["x"])[0] == EmbeddingMath.l2Normalize([1, 0]))
}

@Test func roleAwarePrefixIsAppliedThroughLodeDB() throws {
    let tokenizer = RecordingTokenizer()
    let session = FixedSession(tokenEmbeddings: [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]])
    let preset = EmbeddingPreset(name: "bge-like", dimension: 8, pooling: .cls, queryPrefix: "Q: ", modelIdentity: "m")
    let embedder = ONNXTextEmbedder(preset: preset, tokenizer: tokenizer, session: session)
    let db = try LodeDB(vectorDimension: 8)

    try db.addText("document body", id: "d", embedder: embedder)
    #expect(tokenizer.lastText == "document body") // ingest uses the document role (no prefix)

    _ = try db.search(text: "the query", k: 1, mode: .vector, embedder: embedder)
    #expect(tokenizer.lastText == "Q: the query") // search uses the query role (prefix applied)
}

@Test func meanPoolingRejectsMaskLengthMismatch() throws {
    #expect(throws: LodeDBError.self) {
        _ = try EmbeddingMath.pool([[1, 0], [0, 1], [1, 1]], attentionMask: [1, 1], pooling: .mean)
    }
}

@Test func singleRowOutputIsReturnedAsIs() throws {
    // An already-pooled [1][dim] session output is used directly (any mask).
    #expect(try EmbeddingMath.pool([[5, 6, 7]], attentionMask: [1, 1, 1, 1], pooling: .mean) == [5, 6, 7])
    #expect(try EmbeddingMath.pool([[5, 6, 7]], attentionMask: [], pooling: .cls) == [5, 6, 7])
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

@Test func embedderModelIdentityIsOptionalWithDefaultNil() throws {
    let preset = EmbeddingPreset(name: "m", dimension: 4, modelIdentity: "vendor/model")
    let onnx = ONNXTextEmbedder(
        preset: preset, tokenizer: RecordingTokenizer(), session: FixedSession(tokenEmbeddings: [[1, 0, 0, 0]]))
    #expect(onnx.modelIdentity == "vendor/model")
    // An embedder that does not declare one gets the protocol default (nil).
    #expect(NoIdentityEmbedder().modelIdentity == nil)
}

private struct NoIdentityEmbedder: LodeEmbedder {
    var dimension: Int { 4 }
    func embed(texts: [String]) throws -> [[Float]] { texts.map { _ in [1, 0, 0, 0] } }
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
