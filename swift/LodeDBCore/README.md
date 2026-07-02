# LodeDBCore (Swift)

A Swift binding for LodeDB on macOS and iOS. It is a thin, ergonomic layer over the
vendored Rust `lodedb-core` engine (the same native core the Python package uses),
linked as a prebuilt static `LodeDBCoreFFI.xcframework`. All search, ranking,
chunking, tokenization, and durable storage run in the native core; Swift marshals
values across a small C ABI and serializes access.

It runs fully on device: no Python runtime, no network, no server. The on-disk
`.tvim` format is byte-compatible across platforms, so an index built on a server
loads on a phone.

- Platforms: macOS 13+, iOS 16+ (arm64; device and simulator).
- Storage: exact, quantized (4-bit TurboVec) brute-force scan with a NEON kernel.
  This is not an ANN index; it is exact and O(n), so it suits small to medium
  on-device corpora.

## Adding the package

Released versions are published as a standalone SwiftPM package at
[`Egoist-Machines/swift-lodedb`](https://github.com/Egoist-Machines/swift-lodedb), whose
binary target is the matching `LodeDBCoreFFI.xcframework`:

```swift
.package(url: "https://github.com/Egoist-Machines/swift-lodedb", from: "1.1.0")
```

See [docs/swift-publishing.md](../../docs/swift-publishing.md) for how that package is
produced on each release.

### Building from this repo (development)

The in-repo `Package.swift` resolves the XCFramework from a local `Artifacts/` directory,
so a checkout must build it once:

```sh
# from the repo root: builds the host slice and runs the package
swift/LodeDBCore/scripts/package_xcframework.sh
# all Apple slices (device + simulator):
LODEDB_XCFRAMEWORK_TARGETS="aarch64-apple-darwin aarch64-apple-ios aarch64-apple-ios-sim" \
  swift/LodeDBCore/scripts/package_xcframework.sh
```

The `LodeDBCore` target links `Accelerate`. To resolve a hosted XCFramework instead of a
local build, set `LODEDB_FFI_BINARY_URL` and `LODEDB_FFI_BINARY_CHECKSUM` (the value
`swift package compute-checksum` prints for the zip); the manifest then uses the remote
`binaryTarget`.

## Quick start

### Vector store

```swift
import LodeDBCore

let db = try LodeDB(vectorDimension: 384)
try db.addVector(embedding, id: "doc-1", metadata: ["topic": "ops"])
let hits = try db.search(vector: queryEmbedding, k: 5,
                         filter: MetadataFilter(["topic": "ops"]))
for hit in hits { print(hit.id, hit.score, hit.metadata) }
```

### Text + an embedder

```swift
let embedder = try NLEmbedder()                       // on-device, no model download
let db = try LodeDB(vectorDimension: embedder.dimension)
try db.addText("The deploy runbook for payments.", id: "doc-1",
               metadata: ["topic": "ops"], embedder: embedder)

let vector = try db.search(text: "payment deploy steps", k: 5, mode: .vector, embedder: embedder)
let lexical = try db.search(text: "error E-1001", k: 5, mode: .lexical)
let hybrid = try db.search(text: "rollback procedure", k: 5, mode: .hybrid, embedder: embedder)
```

### Durable store

```swift
let url = URL(fileURLWithPath: "/path/to/store")
let db = try LodeDB(path: url, vectorDimension: 384, modelIdentity: embedder.modelIdentity)
try db.addText("...", id: "doc-1", embedder: embedder)
try db.persist()
try db.close()

// Reopen read-only (lock-free snapshot); identity is checked when given.
let snapshot = try LodeDB.openReadOnly(path: url, modelIdentity: embedder.modelIdentity)
let hits = try snapshot.search(text: "...", k: 5, mode: .vector, embedder: embedder)
```

`LodeStoreOptions` controls durability (`.fsync` / `.buffered`), commit mode
(`.wal` / `.generation`), text retention (`storeText`, `indexText`), and the
single-writer lock.

### CRUD and enumeration

```swift
let record = try db.getDocument("doc-1")          // payload-free: id, metadata, chunkCount
let text = try db.get("doc-1")                    // retained text (nil if not stored)
let page = try db.listDocuments(filter: MetadataFilter(["topic": "ops"]), limit: 50)
try db.updateDocument(id: "doc-1", metadata: ["topic": "archived"])
try db.remove("doc-1")
let stats = try db.stats()                        // documentCount, vectorDimension, model, ...
let collections = try db.collections()
```

### Metadata filters

Flat exact match, or the full Mongo-style predicate grammar, plus a `document_ids`
allowlist:

```swift
MetadataFilter(["topic": "ops"])
MetadataFilter(predicate: .and([
    .equals("topic", "ops"),
    .greaterThanOrEqual("year", "2024"),
    .inSet("severity", ["high", "critical"]),
]))
MetadataFilter(documentIDs: ["doc-1", "doc-2"], predicate: .exists("owner", true))
```

Operators: `equals`, `notEquals`, `greaterThan`, `greaterThanOrEqual`, `lessThan`,
`lessThanOrEqual`, `inSet`, `notInSet`, `exists`, `and`, `or`, `not`. Ordered
comparisons are numeric when both sides parse as numbers, lexical otherwise.

### Batch search

```swift
let perVector = try db.searchMany(vectors: queryVectors, k: 5)        // [[SearchHit]]
let perQuery  = try db.searchMany(texts: queries, k: 5, mode: .lexical)
```

### Late interaction (multi-vector / MaxSim)

```swift
let index = try LodeLateInteractionIndex(vectorDimension: 128)
try index.addDocument(id: "doc-1", patches: tokenVectors, metadata: ["topic": "ops"])
let hits = try index.search(queryPatches: queryTokenVectors, k: 5)
```

### Agent memory

`LodeMemory` is a `save` / `recall` / `forget` facade for using LodeDB as an agent's
long-term memory:

```swift
let memory = try LodeMemory(embedder: try NLEmbedder())
let id = try memory.save("User prefers metric units.", metadata: ["kind": "preference"])
let recalled = try memory.recall("what units does the user like?", k: 3)   // hybrid by default
for hit in recalled { print(hit.id, hit.score, hit.text ?? "") }
try memory.forget(id)
```

See [docs/swift-agent-contract.md](../../docs/swift-agent-contract.md) for the
agent-facing contract, privacy guarantees, and guidance.

## Embeddings on device

`LodeDB` does not embed text itself; you supply a `LodeEmbedder`.

- `NLEmbedder` wraps Apple's NaturalLanguage sentence embeddings. It needs no bundled
  model, no ONNX Runtime, and no network, so it is the zero-setup option. It is its
  own model (not MiniLM/BGE-compatible): an index built with it must be queried with
  it.
- `ONNXTextEmbedder` is the cross-runtime parity path. It owns the embedding contract
  (attention-mask mean pooling or CLS, L2 normalization, the BGE query prefix, the
  output-dimension and model-identity guards) and matches the Python pipeline. You
  supply a `TextTokenizer` and an `EmbeddingModelSession` (ONNX Runtime with the model
  and its tokenizer); those artifacts are provisioned by your app, not vendored here.
  `EmbeddingPreset.miniLM` (384-d) and `EmbeddingPreset.bge` (768-d, CLS, query
  prefix) carry the contracts.
- Any type conforming to `LodeEmbedder` works. Use `embed(texts:role:)` when an
  embedder is prefix-asymmetric; `LodeDB` requests `.document` on ingest and `.query`
  on search.

## Scan backend

The default search path is the native NEON scan over the quantized index; on-device
benchmarks show it beats a GPU/MPS scan, and CUDA is irrelevant on Apple hardware.

`MetalVectorScanner` is an opt-in, exact (full-precision f32) brute-force top-k
dot-product scan with a Metal GPU path and a CPU fallback, selected via a capability
probe. It is for callers that hold raw f32 vectors (exact reranking, app-managed
vectors); it does not replace the core's quantized scan.

## Concurrency and errors

A `LodeDB` instance is safe to share across threads; access is serialized internally
(the native engine is single-writer). Operations throw `LodeDBError`, mapped from the
C ABI status codes (`invalidArgument`, `notFound`, `corruptStore`, `planStale`,
`unsupported`, `internalError`).

For concurrent ingest, `LodeAppender` lets many processes durably log vector-in
records to one store's WAL at once (a shared lock, distinct from the exclusive
writer's), folded into the index by the next *writable* `LodeDB` open. A read-only
snapshot (`openReadOnly`) ignores the WAL tail, so appended records are queryable
only after a writable open replays and checkpoints them. It requires WAL commit
mode. Each record is a precomputed vector plus metadata, and an optional caption
(e.g. an image's) retained only when opened with `storeText: true` (off by default,
so no raw text reaches the WAL):

```swift
let appender = try LodeAppender.open(at: url)  // storeText defaults off
let lsn = try appender.append(id: "doc-1", vector: embedding, metadata: ["topic": "ops"])
_ = try appender.append([LodeAppendDocument(id: "doc-2", vector: other)]) // batch
_ = try appender.delete(ids: ["doc-1"])

// Retain a caption (only for a store whose writer also uses store_text):
let captioned = try LodeAppender.open(at: url, storeText: true)
_ = try captioned.append(id: "img-1", vector: clipVector, text: "a red bicycle")
```

## Out of scope

The CLI, dev server, MCP server, model-download flow, framework adapters
(LangChain / LlamaIndex / mem0 / PrivateGPT), the CUDA path, and the
PyTorch/sentence-transformers stack stay in the Python package; iOS apps call this
Swift API directly.
