# LodeDB Swift: agent contract

This is the contract for using LodeDB as an on-device agent memory from Swift, for
LLM agents and the developers who wire them. It is the Swift-native equivalent of the
Python MCP server's tools: the same `save` / `recall` / `forget` verbs, no server and
no network. The API is `LodeMemory` (in `LodeDBCore`); the lower-level `LodeDB` is
available when you need full control.

## Operations

`LodeMemory` wraps a `LodeDB` collection and an embedder.

| Operation | Signature | Behavior |
| --- | --- | --- |
| save | `save(_ text:, id:, metadata:) -> String` | Stores a memory; returns its id (a fresh UUID when `id` is omitted). Re-saving the same id replaces it. |
| recall | `recall(_ query:, k:, mode:, filter:) -> [MemoryHit]` | Returns the top `k` memories. Default mode is `.hybrid` (vector + lexical). Each `MemoryHit` carries `id`, `score`, the stored `text`, and `metadata`. |
| forget | `forget(_ id:) -> Bool` | Removes a memory; returns whether it existed. |
| count | `count: Int` | Number of stored memories. |
| persist | `persist()` | Flushes a durable store to disk (no-op for in-memory). |

`MemoryHit`: `{ id: String, score: Float, text: String?, metadata: [String: String] }`.

Lower-level `LodeDB` primitives an agent host can expose as tools map cleanly:
`addText` / `addVector` (save), `search` / `searchMany` (recall), `remove` (forget),
`get` / `getDocument` / `listDocuments` (inspect), `stats` (metrics).

## Privacy guarantees

- On device only. No network calls, no server, no telemetry of content. Data lives in
  the store you open (in memory, or a file path you control).
- Search results are payload-free. `search` returns ids, scores, and metadata; it does
  not return document text. `LodeMemory.recall` fetches text explicitly per hit via
  `get`, so the only place raw text crosses back is a deliberate fetch.
- Raw text is retained only when `storeText` is on (the default for `LodeMemory` and
  `addText`). With `storeText` off, `get` returns nil and only vectors and metadata are
  kept.
- `stats()` is metrics-only (counts, dimension, model identity, generation). It never
  includes document text, vectors, or queries.

## Do / don't

- Do use one embedding model per collection. Bind it with `modelIdentity` (e.g.
  `LodeMemory(embedder:)` or `LodeDB(path:vectorDimension:modelIdentity:)`); reopening
  a store with a different same-dimension model is then rejected instead of silently
  degrading recall.
- Do use `.hybrid` (the default) for agent recall: it combines semantic (vector) and
  exact-token (lexical) matches, which matters for ids, codes, and serials a pure
  vector search would miss.
- Don't open a writable store at the same path from two processes. LodeDB is
  single-writer; a writable open takes a lock. Use `openReadOnly` for concurrent
  readers.
- Don't expect `.lexical` or `.hybrid` to work without indexed text. Text added via
  `addText` is indexed by default (`indexText`); vectors added via `addVector` are not.
- Don't store secrets in `metadata` or the model identity; they are persisted in the
  store's manifest.

## Capabilities and limits

- The scan is exact and O(n) over a 4-bit-quantized index (not an ANN index). It suits
  small to medium on-device corpora; it does not target billion-scale. Quantization
  trades a little recall (about 0.95) for size and speed.
- Filters run natively before top-k. Supported operators: `equals`, `notEquals`,
  `greaterThan(OrEqual)`, `lessThan(OrEqual)`, `inSet`, `notInSet`, `exists`, `and`,
  `or`, `not`, plus a `documentIDs` allowlist. Metadata values are strings; ordered
  comparisons are numeric when both sides parse as numbers.
- Durable stores are crash-atomic (WAL or generation commit). Reopen reflects the last
  committed generation.
- A `LodeDB` instance is thread-safe (access is serialized); a closed handle rejects
  further operations.

## Example: long-term memory for an in-app agent

```swift
import LodeDBCore

// One memory per user, persisted on device.
let memory = try LodeMemory(
    db: try LodeDB(path: storeURL, vectorDimension: embedder.dimension,
                   modelIdentity: embedder.modelIdentity),
    embedder: embedder
)

// On each turn: save salient facts, recall relevant ones.
try memory.save("User's project ships on Fridays.", metadata: ["kind": "fact"])
let context = try memory.recall("when does the project ship?", k: 5)
let grounding = context.compactMap(\.text).joined(separator: "\n")
// ... feed `grounding` into the prompt ...
try memory.persist()
```

## Example: exact-token recall for codes and ids

```swift
// Hybrid recall surfaces an exact code a pure vector search would rank lower.
let hits = try memory.recall("error E-1001 in the payments service", k: 5, mode: .hybrid)
```
