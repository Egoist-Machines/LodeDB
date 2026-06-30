#if canImport(NaturalLanguage)
import Foundation
import NaturalLanguage

/// A batteries-included on-device `LodeEmbedder` backed by Apple's NaturalLanguage
/// sentence embeddings. It needs no bundled model, no ONNX Runtime, and no network:
/// the embedding model ships with the OS, so this is the zero-setup way to get a real
/// (non-test) embedder on Apple platforms.
///
/// It does NOT reproduce the MiniLM/BGE vectors the Python presets use, so an index
/// built with `NLEmbedder` must be queried with `NLEmbedder` (it is its own model).
/// For cross-runtime parity with the Python stack, use `ONNXTextEmbedder` instead.
public final class NLEmbedder: LodeEmbedder {
    private let embedding: NLEmbedding
    private let normalize: Bool

    public let dimension: Int

    /// A stable identity for this embedder, suitable as the persisted model name.
    public let modelIdentity: String?

    /// Creates a sentence embedder for `language`. Throws `.unsupported` if the OS has
    /// no sentence-embedding model for that language available.
    public init(language: NLLanguage = .english, normalize: Bool = true) throws {
        guard let embedding = NLEmbedding.sentenceEmbedding(for: language) else {
            throw LodeDBError.unsupported(
                "no on-device sentence embedding is available for language '\(language.rawValue)'")
        }
        self.embedding = embedding
        self.dimension = embedding.dimension
        self.normalize = normalize
        self.modelIdentity = "apple.NLEmbedding.sentence.\(language.rawValue)"
    }

    public func embed(texts: [String]) throws -> [[Float]] {
        try texts.map { text in
            guard let vector = embedding.vector(for: text) else {
                throw LodeDBError.invalidArgument("no on-device embedding for the given text")
            }
            let floats = vector.map { Float($0) }
            return normalize ? EmbeddingMath.l2Normalize(floats) : floats
        }
    }
}
#endif
