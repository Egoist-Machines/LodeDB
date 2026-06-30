import Foundation

/// How token embeddings are pooled into one sentence embedding.
public enum EmbeddingPooling: String, Sendable {
    /// Attention-mask mean pooling (the sentence-transformers / LodeDB default).
    case mean
    /// The first ([CLS]) token's embedding.
    case cls
}

/// Whether a text is embedded as a stored document or a search query. Only affects
/// which preset prefix is applied (BGE uses an asymmetric query prefix).
public enum EmbeddingRole: Sendable {
    case document
    case query
}

/// Tokenizer output for one text: token ids and the attention mask, as the ONNX
/// graph expects them.
public struct TokenizedText: Sendable, Equatable {
    public let inputIDs: [Int32]
    public let attentionMask: [Int32]

    public init(inputIDs: [Int32], attentionMask: [Int32]) {
        self.inputIDs = inputIDs
        self.attentionMask = attentionMask
    }
}

/// Tokenizes text for an embedding model. Conform with a Swift tokenizer (e.g.
/// swift-transformers reading the model's `tokenizer.json`) or a binding over the
/// Rust `tokenizers` crate; the tokenizer must match the one the model was trained
/// with or recall degrades.
public protocol TextTokenizer: Sendable {
    /// Encodes `text`, truncating to at most `maxLength` tokens (padding is the
    /// caller's concern; pooling ignores masked positions).
    func encode(_ text: String, maxLength: Int) throws -> TokenizedText
}

/// Runs an embedding model on tokenized input. Conform with an ONNX Runtime session
/// (CoreML execution provider on device, CPU fallback). The result is either the
/// per-token hidden states (`[seqLen][dim]`, pooled here) or an already-pooled
/// sentence row (`[1][dim]`).
public protocol EmbeddingModelSession: Sendable {
    func run(_ input: TokenizedText) throws -> [[Float]]
}

/// The embedding contract: pooling and L2 normalization, kept byte-faithful to the
/// Python pipeline (`embedding_backends._pool_onnx_output` / `_l2_normalize_rows`)
/// so an index built with one runtime stays compatible with the other.
public enum EmbeddingMath {
    /// Pools per-token embeddings into one vector. Mean pooling weights each token by
    /// its attention-mask value and divides by `max(maskSum, 1)`; CLS takes token 0.
    public static func pool(
        _ tokenEmbeddings: [[Float]],
        attentionMask: [Int32],
        pooling: EmbeddingPooling
    ) throws -> [Float] {
        guard let first = tokenEmbeddings.first else {
            throw LodeDBError.invalidArgument("token embeddings must not be empty")
        }
        // A single row is either an already-pooled sentence vector (the Python path
        // returns a 2D ONNX output as-is) or a one-token sequence; both pool to that
        // row under mean and equal token 0 under CLS, so return it directly.
        if tokenEmbeddings.count == 1 {
            return first
        }
        switch pooling {
        case .cls:
            return first
        case .mean:
            // Fail loudly on a tokenizer/session shape mismatch rather than silently
            // treating missing/extra mask entries as zero.
            guard attentionMask.count == tokenEmbeddings.count else {
                throw LodeDBError.invalidArgument(
                    "attention mask length \(attentionMask.count) does not match token count \(tokenEmbeddings.count)")
            }
            let dimension = first.count
            var sum = [Float](repeating: 0, count: dimension)
            var denominator: Float = 0
            for (index, token) in tokenEmbeddings.enumerated() {
                guard token.count == dimension else {
                    throw LodeDBError.invalidArgument("ragged token embedding rows")
                }
                let weight = Float(attentionMask[index])
                if weight == 0 { continue }
                for component in 0..<dimension {
                    sum[component] += token[component] * weight
                }
                denominator += weight
            }
            let safe = max(denominator, 1)
            return sum.map { $0 / safe }
        }
    }

    /// Row-wise L2 normalization, preserving an all-zero vector (matching the Python
    /// `safe_norms` guard).
    public static func l2Normalize(_ vector: [Float]) -> [Float] {
        let norm = vector.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
        let safe = norm == 0 ? 1 : norm
        return vector.map { $0 / safe }
    }
}

/// The contract for a bundled embedding model: dimension, sequence length, pooling,
/// asymmetric prefixes, and the persisted model identity (written into the store so
/// a mismatched model is rejected rather than silently degrading recall).
public struct EmbeddingPreset: Sendable, Equatable {
    public let name: String
    public let dimension: Int
    public let maxSequenceLength: Int
    public let pooling: EmbeddingPooling
    public let queryPrefix: String
    public let documentPrefix: String
    public let modelIdentity: String

    public init(
        name: String,
        dimension: Int,
        maxSequenceLength: Int = 256,
        pooling: EmbeddingPooling = .mean,
        queryPrefix: String = "",
        documentPrefix: String = "",
        modelIdentity: String
    ) {
        self.name = name
        self.dimension = dimension
        self.maxSequenceLength = maxSequenceLength
        self.pooling = pooling
        self.queryPrefix = queryPrefix
        self.documentPrefix = documentPrefix
        self.modelIdentity = modelIdentity
    }

    /// all-MiniLM-L6-v2: 384-dim, mean pooling, no prefixes. Fast default.
    public static let miniLM = EmbeddingPreset(
        name: "minilm",
        dimension: 384,
        modelIdentity: "sentence-transformers/all-MiniLM-L6-v2"
    )

    /// BAAI/bge-base-en-v1.5: 768-dim, CLS pooling, asymmetric query prefix. (BGE is
    /// trained for CLS pooling, matching the Python `bge` preset's `pooling="cls"`.)
    public static let bge = EmbeddingPreset(
        name: "bge",
        dimension: 768,
        pooling: .cls,
        queryPrefix: "Represent this sentence for searching relevant passages: ",
        modelIdentity: "BAAI/bge-base-en-v1.5"
    )
}

/// A `LodeEmbedder` that tokenizes, runs an ONNX model, mean-pools, and L2-normalizes
/// per `preset`, applying the role-appropriate prefix. The tokenizer and ONNX session
/// are supplied by the integrator (ONNX Runtime + the model's tokenizer); this type
/// owns only the parity-critical pooling/normalization/prefix/identity contract.
public final class ONNXTextEmbedder: LodeEmbedder {
    public let preset: EmbeddingPreset
    private let tokenizer: TextTokenizer
    private let session: EmbeddingModelSession

    public var dimension: Int { preset.dimension }

    /// The persisted model identity (`required_model_name`); used by LodeDB so a store
    /// records which model produced its vectors.
    public var modelIdentity: String? { preset.modelIdentity }

    public init(preset: EmbeddingPreset, tokenizer: TextTokenizer, session: EmbeddingModelSession) {
        self.preset = preset
        self.tokenizer = tokenizer
        self.session = session
    }

    public func embed(texts: [String]) throws -> [[Float]] {
        try embed(texts: texts, role: .document)
    }

    /// Embeds with the role-appropriate prefix (BGE applies its query prefix only to
    /// queries). `LodeDB` calls this with `.document` for ingest and `.query` for search.
    public func embed(texts: [String], role: EmbeddingRole) throws -> [[Float]] {
        let prefix = role == .query ? preset.queryPrefix : preset.documentPrefix
        return try texts.map { text in
            let tokenized = try tokenizer.encode(prefix + text, maxLength: preset.maxSequenceLength)
            let tokenEmbeddings = try session.run(tokenized)
            let pooled = try EmbeddingMath.pool(
                tokenEmbeddings, attentionMask: tokenized.attentionMask, pooling: preset.pooling)
            let normalized = EmbeddingMath.l2Normalize(pooled)
            guard normalized.count == preset.dimension else {
                throw LodeDBError.invalidArgument(
                    "embedding dimension \(normalized.count) does not match preset \(preset.dimension)")
            }
            return normalized
        }
    }
}
