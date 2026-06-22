# Findings: make the graph's semantic index fully rebuildable

> Investigation of [`research-prompts/04-durable-vector-rebuildable-index.md`](../research-prompts/04-durable-vector-rebuildable-index.md).
> Design input — analysis and a recommendation, not a committed decision. All
> claims are grounded in code (`file:line`) or measured under `/tmp` against the
> **real** vendored TurboVec (`LODEDB_ALLOW_MOCK_TURBOVEC=0`).

## 1. TL;DR

**Recommend Option 1: persist the raw f32 node vector in the SQLite topology
store**, gated behind an opt-in flag (`retain_vectors=True`), so `reindex()` can
re-`add_vectors` from the source of truth. It is the only option that actually
satisfies the success criterion ("post-reindex retrieval identical to
pre-reindex") while keeping the rebuild a true *rebuild-from-source-of-truth*.

The two alternatives do not hold up under the code:

- **Option 2 (export/import encoded codes) cannot rebuild from scratch.** The
  TurboVec binding does expose `export_encoded`/`add_encoded` (real, not just the
  conftest mock — confirmed below), and the engine *already* uses them for
  O(changed) delta persistence. But encoded codes are portable **only** into an
  index whose `calibration_fingerprint()` matches the exporting index
  (`id_map.rs:298-306`, `turbovec_delta_store.py:249-270`, *fails closed*). A
  rebuilt-from-empty index refits its TQ+ calibration to whatever the new first
  batch happens to be (`lib.rs:770-789`, `encode.rs:45,149`), so the fingerprint
  differs and `add_encoded` is rejected. Option 2 therefore degenerates into
  "preserve the engine's own `.tvim`/`.tvd` artifacts" — i.e. snapshot/restore,
  not the "drop the index dir and rebuild from SQLite" the prompt asks for. It is
  worth adopting as a *fast-path*, but not as the correctness mechanism.
- **Option 3 (reindex callback) is correct but pushes the burden onto every
  caller** and silently regresses if a caller forgets — the opposite of a
  self-healing rebuildable index. Keep it as a complementary hook, not the
  primary answer.

The payload-free boundary (AGENTS.md) is **not violated** by Option 1: raw
vectors land in the *user's own* topology DB (`topology.sqlite3`), never in any
telemetry/redacted artifact. But because the f32 vector is now durable
user-content-adjacent data, the store must be **redaction-aware** — gated by the
same opt-in spirit as `store_text=False`. Details in §5.

---

## 2. Current behavior + a concrete demonstration that vector-in reindex is lossy

### 2.1 How a vector-in node is stored, and why the vector is gone after a rebuild

`KnowledgeGraph.add_node(..., embedding=[...])` writes the node to SQLite, then
indexes it via `_index_node`:

```python
# src/lodedb/graph/knowledge_graph.py:427-438
def _index_node(self, node, *, embedding=None):
    doc_id = _NODE_PREFIX + node.id
    metadata = {"kind": "node", "type": node.type, "node_id": node.id}
    if embedding is not None:
        self._db.add_vectors(embedding, id=doc_id, metadata=metadata)   # vector-in
    elif node.label.strip():
        self._db.add(node.label, id=doc_id, metadata=metadata)          # text path
    else:
        self._db.remove(doc_id)
```

`add_vectors` quantizes and discards the raw vector — only packed codes + a scale
survive in the engine (`db.py:300-341` → `index.upsert_vectors_batch`; the engine
keeps the f32 only transiently and drops the synced rows per AGENTS.md's
"O(changed)" rule). The topology store keeps **only** `label`/`properties`, never
the vector — by design and by its own docstring ("It is stdlib-only (`sqlite3`)
and holds no embeddings — vectors live in LodeDB.", `_store.py:11`; schema
`_store.py:23-41` has columns `id, type, label, properties` and nothing else).

`reindex()` rebuilds each node by calling `_index_node(node)` **with no
`embedding`** (`knowledge_graph.py:378-379`):

```python
# src/lodedb/graph/knowledge_graph.py:377-379
for node in self._store.iter_nodes():
    if node.label.strip():
        self._index_node(node)        # embedding=None -> re-embeds the LABEL
```

So a vector-in node is rebuilt by re-embedding its `label` text — a completely
different vector from the one the caller supplied. `reindex()`'s own docstring
admits this (`knowledge_graph.py:361-366`). Two failure modes follow directly
from the `_index_node` branch table:

| vector-in node has… | reindex branch taken | result |
|---|---|---|
| a (placeholder) `label` | `add(node.label)` | **wrong vector** (silent corruption) |
| an empty `label` | `remove(doc_id)` | **node vanishes** from the index (unsearchable) |

### 2.2 Measured demonstration (real TurboVec, `/tmp`, `minilm`/dim 384)

**Demo A — placeholder labels (silent corruption).** Build a KG with 5 vector-in
nodes (each a distinct random unit vector; label is unrelated placeholder text),
record retrieval, drop `index/`, reopen, `reindex()`, re-measure. Query each node
*by its own original vector*; the correct top-1 is the node itself.

```
=== BEFORE reindex (query each vec node by its own vector) ===
  vnode-0: top3=['vnode-0','vnode-2','vnode-1']  top1_score=0.9969  self_is_top1=True
  vnode-1: top3=['vnode-1','vnode-2','vnode-3']  top1_score=1.0011  self_is_top1=True
  vnode-2: top3=['vnode-2','vnode-0','vnode-1']  top1_score=0.9978  self_is_top1=True
  vnode-3: top3=['vnode-3','vnode-1','vnode-4']  top1_score=0.9985  self_is_top1=True
  vnode-4: top3=['vnode-4','vnode-3','vnode-0']  top1_score=1.0005  self_is_top1=True

=== AFTER drop-index + reindex() ===
  vnode-0: top3=['vnode-2','vnode-1','vnode-0']  top1_score=0.0679  self_is_top1=False
  vnode-1: top3=['vnode-2','vnode-0','vnode-4']  top1_score=0.0773  self_is_top1=False
  vnode-2: top3=['vnode-3','vnode-4','vnode-2']  top1_score=0.0685  self_is_top1=False
  vnode-3: top3=['vnode-0','vnode-4','vnode-2']  top1_score=0.0285  self_is_top1=False
  vnode-4: top3=['vnode-0','vnode-1','vnode-4']  top1_score=0.0349  self_is_top1=False

=== SUMMARY ===
index docs before/after: 8 / 8        <- count UNCHANGED (masks the corruption)
self-is-top1 before:     5/5
self-is-top1 after:      0/5          <- not one node retrieves itself
rankings identical:      0/5          <- every ranking changed
```

The doc count is preserved (8/8), so `stats()` looks healthy — but every
vector-in doc now carries the *wrong* vector (re-embedded label), and self-recall
collapses from ~1.0 to ~0.03–0.08. This is silent corruption, not an error.

**Demo B — empty labels (the canonical bring-your-own-vector case, strictly
worse).** 4 vector-in nodes with `label=""`:

```
docs before: 4   stats nodes=4 indexed_documents=4
reindex result: {'reindexed_nodes': 0, 'removed_orphans': 0}
docs after:  0   stats nodes=4 indexed_documents=0
vector-in docs still present in index after reindex: 0/4
```

`reindex()` reports `reindexed_nodes: 0` and the index empties to 0 docs: the 4
nodes still exist in the topology but are **gone** from the semantic index — they
silently drop out of `semantic_nodes`/`search_subgraph` entirely. Scripts:
`/tmp/kg_demo.py`, `/tmp/kg_demo2.py` (scratch; not committed).

> Caveat on test fidelity: these were run against the **real** vendored TurboVec.
> The conftest mock's `export_encoded`/`add_encoded` are *fakes* — `export_encoded`
> returns all-zero codes and `add_encoded` stores a zero vector
> (`tests/conftest.py:134-147`). Any Option-2 prototype validated against the mock
> would falsely "pass." The validation plan in §6 must force the real binding.

---

## 3. Prior art (cited)

The dominant production pattern is exactly LodeDB's stated framing — *separate the
durable source of truth from the derived, rebuildable index* — but every system
that makes the index droppable **retains the vectors** (or the raw content), not
merely the quantized codes:

- **LanceDB / Lance format.** Vectors live in the columnar Lance storage; the
  vector index (IVF/HNSW) is a *separate artifact* layered alongside the data, and
  the underlying vector data is independent of the index. LanceDB exposes
  `drop_index` and a `reindexing`/`optimize()` flow that rebuilds the index *from
  the retained columnar vectors* — the index is throwaway, the vectors are not.
  This is the precise shape Option 1 gives LodeDB's graph layer.
  ([Lance format](https://docs.lancedb.com/lance),
  [Reindexing](https://docs.lancedb.com/indexing/reindexing),
  [Vector Indexes](https://docs.lancedb.com/indexing/vector-index))
- **Cognee / Graphiti / Zep (the comparable agent-memory stacks).** Cognee stores
  embeddings in LanceDB keyed to graph nodes (SQLite for metadata, Kuzu for
  topology); Graphiti provides "clear the graph and rebuild indices." They keep
  the **embeddings** durably and rebuild the index from them — they do not try to
  reconstruct embeddings from labels.
  ([Neo4j: Graphiti](https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/),
  [Zep vs Cognee](https://vectorize.io/articles/zep-vs-cognee))
- **General reindexing guidance.** "A robust pipeline separates the Source of
  Truth (your raw data) from the Vector Representation (your searchable index).
  You must maintain a persistent, non-vectorized store of your raw content." The
  index is rebuilt *from retained data*, not re-derived from a lossy proxy.
  ([Reindexing pipeline](https://medium.com/@kandaanusha/vector-database-reindexing-pipeline-87efa1d1cd19),
  [Unstructured: indexing strategies](https://unstructured.io/insights/vector-indexing-strategies-for-high-performance-ai-search))
- **Memory–disk split (why keeping raw vectors is normal).** Large systems hold
  quantized codes in RAM and **full-precision vectors on disk** for a rerank;
  pgvector's recommended pattern is "store the full vector in the table and
  quantize to `halfvec` in the index." Persisting the raw vector beside a
  quantized index is the standard, not an anomaly.
  ([pgvector quantization](https://jkatz05.com/post/postgres/pgvector-scalar-binary-quantization/),
  [MongoDB vector quantization](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-quantization/))
- **Quantizers are data-dependent and not portable across a rebuild** (the
  theory behind §4's fingerprint constraint). "The quantizer is specific to the
  data it was trained on… changes to the dataset may necessitate retraining
  rather than simply reusing the existing codebook across different index
  versions." This is exactly why Option 2 cannot rebuild from scratch.
  ([Quantization under streaming updates (arXiv)](https://arxiv.org/pdf/2512.18335),
  [Weaviate PQ compression](https://docs.weaviate.io/weaviate/configuration/compression/pq-compression))
- **Snapshot/restore ≠ rebuild-from-source.** Qdrant snapshots archive the
  *encoded index state* for fast restore "if you don't want to go through the long
  process of indexing." That is the analogue of preserving LodeDB's `.tvim`/`.tvd`
  (Option 2), and it is complementary to — not a substitute for — a true rebuild
  from the source of truth.
  ([Qdrant snapshots / production](https://qdrant.tech/articles/vector-search-production/))

**Synthesis:** prior art unanimously says *retain the vectors in the durable
store and treat the index as derived* (Option 1). No comparable system rebuilds a
node's vector from a text proxy, and quantized-code reuse is a snapshot-restore
optimization bounded by quantizer portability — never the from-source rebuild
path. This independently corroborates the recommendation.

---

## 4. Recommended design (Option 1 in detail)

Persist the raw, normalized f32 vector for vector-in nodes/edges in the SQLite
topology store, so `reindex()` re-supplies it via `add_vectors`. The topology DB
is already the declared source of truth (`docs/graph.md:11-13`), so storing the
vector there *closes the loop* rather than introducing a second source.

### 4.1 Schema (`src/lodedb/graph/_store.py`)

Add a dedicated sidecar table (not a column on `nodes`) so text nodes pay nothing
and edge vectors share the mechanism:

```sql
CREATE TABLE IF NOT EXISTS entity_vectors (
    kind    TEXT NOT NULL,             -- 'node' | 'edge'
    id      TEXT NOT NULL,             -- node id / edge id
    dim     INTEGER NOT NULL,          -- guards dim drift / model swaps
    vector  BLOB NOT NULL,             -- np.float32 little-endian, len == dim*4
    PRIMARY KEY (kind, id)
);
```

Store the **post-normalization** vector (what `add_vectors` actually indexed, see
`db.py:331` `_prepare_vector(..., normalize=normalize)`), so the rebuild is
byte-faithful without re-applying normalization. New `TopologyStore` methods:
`put_entity_vector(kind, id, vector)`, `get_entity_vector`, `delete_entity_vector`,
`iter_entity_vectors(kind)`. Delete the row inside the existing `remove_node` /
`remove_edge` transactions so vectors never outlive their entity.

### 4.2 API (`src/lodedb/graph/knowledge_graph.py`)

- Constructor gains `retain_vectors: bool = False` (opt-in; see §5 payload-free).
- `_index_node` / `_index_edge`: when `embedding is not None` **and**
  `retain_vectors`, persist the prepared vector via the store alongside the
  `add_vectors` call. To store *exactly* the indexed bytes, normalize once in the
  graph layer and pass `normalize=False` to `add_vectors`, persisting the same
  array (avoids a normalize/re-normalize mismatch).
- `reindex()`: replace the single text-only loop with a two-source rebuild —

  ```python
  for node in self._store.iter_nodes():
      vec = self._store.get_entity_vector("node", node.id) if self.retain_vectors else None
      if vec is not None:
          self._index_node(node, embedding=vec)     # faithful vector-in rebuild
      elif node.label.strip():
          self._index_node(node)                     # text rebuild (unchanged)
  ```

  This makes the success criterion hold for *all* nodes. No engine change is
  required for correctness.

### 4.3 Engine changes

**None required.** Option 1 lives entirely in the graph layer and reuses the
public `add_vectors` path. (Optionally, a future fast-path can call the engine's
existing `export_encoded`/`add_encoded` — see §5 — but that is an optimization,
not part of the correctness design.)

### 4.4 Storage cost estimate

Raw f32 is ~**8×** the persisted 4-bit code (TurboVec default `bit_width=4`,
`presets.py:52-55`). Per vector-in entity:

| dim (preset) | raw f32 (Option 1) | 4-bit codes + scale (already on disk) | overhead ratio |
|---|---|---|---|
| 384 (`minilm`) | 1 536 B | ~196 B | ~7.8× |
| 768 (`bge`) | 3 072 B | ~388 B | ~7.9× |
| 1536 | 6 144 B | ~772 B | ~8.0× |

Absolute cost is modest for graph-memory scales: **1 M** vector-in nodes at dim
384 ≈ **1.5 GB** of raw vectors in `topology.sqlite3` (plus SQLite per-row +
btree overhead, call it ~1.7–1.9 GB). For agent-memory graphs (typically 10²–10⁵
nodes) this is single-digit MB. Because it is opt-in, text-labelled graphs and
callers who accept lossy vector-in reindex pay **zero**.

---

## 5. Tradeoffs / risks / invariant + payload-free analysis

### 5.1 Payload-free boundary (the question the prompt foregrounds)

AGENTS.md's hard rule scopes "no raw embeddings" to **telemetry and the redacted
artifacts** (`.json` snapshot, `.jsd` journal, `.tvim`/`.tvd` sidecars, audit
log) — *"counts / bytes / latency / ids / timestamps"* only (`AGENTS.md:22-27`).
Option 1 writes the vector to `topology.sqlite3`, the **user's own** store, which
is *not* one of those artifacts and which no telemetry/redacted path reads. So the
letter of the invariant is **not** violated.

But the spirit matters: the f32 vector is sensitive, user-content-adjacent data,
and AGENTS.md already establishes the precedent that even the user's own durable
content is **opt-out** (the raw-text store is retained by default but
`store_text=False` keeps no text on disk, `AGENTS.md:22-27`). The raw vector
should be **redaction-aware** under the same model:

- **Opt-in, not default** (`retain_vectors=False`). A caller using vector-in often
  does so precisely because they own the embedder and can re-supply (Option 3) —
  don't force a 8× durable copy on them.
- **Symmetry with `store_text`.** Document that with `retain_vectors=False`,
  vector-in nodes are **not** faithfully rebuildable (today's behavior, now
  explicit), and `reindex()` should *report* how many vector-in entities it could
  not rebuild (e.g. `unrebuildable_vector_nodes` in its return dict) instead of
  silently corrupting/dropping them.
- **Never widen the redacted boundary.** The vector must never leak into the
  engine's `.tvim`/`.tvd`/journal/telemetry beyond what `add_vectors` already does
  (codes only). Option 1 touches only the SQLite sidecar, so this holds.

### 5.2 Risks

- **Model/dim drift.** If a caller reopens with a different `model`, the persisted
  dim won't match; the `dim` column lets `reindex()` skip/raise rather than feed a
  mismatched vector to `add_vectors`. Mirrors the "only mix vectors from the same
  model" rule (`docs/graph.md:78-79`).
- **Normalization double-apply.** Storing post-normalization bytes + passing
  `normalize=False` on rebuild avoids re-normalizing an already-unit vector.
  Covered in §4.2; must be tested (§6).
- **Write amplification.** Each vector-in upsert now also writes a BLOB to SQLite
  (WAL-journaled). For bursty ingest this is one extra small row write per node —
  negligible next to the embed/quantize cost, but should appear in the bench
  (§6).
- **Quantization is still lossy** — but that is *identical* to the original
  ingest. Option 1 reproduces the original quantized state because it replays the
  same f32 through the same `add_vectors`; it does not promise the index is
  lossless, only that rebuild == original.

### 5.3 Why not Option 2 as the correctness mechanism (the decisive code)

The TurboVec binding genuinely exposes the encoded-row API (real, not just the
mock):

- `export_encoded(ids) -> (codes, scales)` and `add_encoded(ids, codes, scales)`
  in the **core** (`turbovec/src/id_map.rs:227`, `:309`) and the **PyO3 binding**
  (`turbovec-python/src/lib.rs:311`, `:344`), with dedicated round-trip tests
  (`turbovec/tests/encoded_rows.rs:30` "export_then_add_encoded_reproduces_search_results").
- The engine **already** uses them for O(changed) delta persistence:
  `codes, scales = index.export_encoded(upsert_stable_ids)`
  (`turbovec_delta_store.py:141`) on commit, replayed via `index.add_encoded(...)`
  on load (`turbovec_delta_store.py:282`).

But codes are portable **only** between indexes with an equal calibration
fingerprint, and the engine *enforces* this, failing closed:

```python
# src/lodedb/engine/turbovec_delta_store.py:249-255   (base segment)
recorded_fingerprint = int(base_entry.get("calibration_fingerprint", 0))
if recorded_fingerprint and hasattr(index, "calibration_fingerprint"):
    actual = int(index.calibration_fingerprint())
    if actual != recorded_fingerprint:
        raise RuntimeError("TurboVec delta replay rejected: base calibration fingerprint mismatch")
# :266-270   (each delta segment) — same check, also raises on mismatch
```

And the fingerprint is **data-dependent** — it hashes the TQ+ calibration that is
fit from the first batch:

```rust
// third_party/turbovec/turbovec/src/lib.rs:770-789
pub fn calibration_fingerprint(&self) -> u64 {
    ...
    mix(&(self.dim.unwrap_or(0) as u64).to_le_bytes());
    mix(&(self.bit_width as u64).to_le_bytes());
    for value in &self.tqplus_shift { mix(&value.to_bits().to_le_bytes()); }   // fitted from data
    for value in &self.tqplus_scale { mix(&value.to_bits().to_le_bytes()); }   // fitted from data
    hash
}
```

`tqplus_shift`/`tqplus_scale` are identity below `TQPLUS_MIN_SAMPLES = 1000`
vectors and otherwise fit to the **first batch's empirical quantiles**
(`encode.rs:45,149`), then frozen. `build_turbovec_serving_index` rebuilds by
re-`add_with_ids` over all current chunks (a fresh calibration fit), so a
rebuilt-from-empty index's fingerprint will generally **differ** from the
original's whenever the corpus crossed the 1000-vector threshold or the first
batch differs.

**Consequence:** "drop the index dir, re-import the exported codes" hits the
fingerprint check and is **rejected**. To make Option 2 work you must persist *and
restore the calibration state*, i.e. keep the engine's `.tvim`/`.tvd` artifacts —
at which point you are doing snapshot/restore, not rebuilding from the topology
source of truth. Option 2 is a legitimate *fast-path* (skip re-quantization when
the engine artifacts are intact and only the graph-layer wiring was lost), but it
cannot be the correctness guarantee the prompt requires. It also couples the
graph layer to engine internals (fingerprint, code layout) that AGENTS.md keeps
behind the commit-manifest boundary.

### 5.4 Why not Option 3 as the primary

Correct and lowest-storage, but it makes the rebuildable-index guarantee
**conditional on caller discipline**: forget to wire the callback and `reindex()`
silently regresses to today's corruption (§2). It is the right *complement* to
Option 1 (a `reindex(supply_embedding=fn)` hook for callers who decline
`retain_vectors`), but a self-healing index should not depend on the caller
re-running their model.

---

## 6. Prototype / validation plan

### 6.1 Correctness test (the success criterion)

Add to the graph test suite (must run against the **real** TurboVec — the conftest
mock fakes `export_encoded`/`add_encoded`, §2.2, so an Option-2-style test would
falsely pass; an Option-1 test is fine on the mock for ranking but should also run
real to validate quantization parity):

```python
def test_reindex_faithful_for_vector_in_nodes(tmp_path):
    kg = KnowledgeGraph(tmp_path / "kg", retain_vectors=True,
                        _embedding_backend=HashEmbeddingBackend(native_dim=384))
    rng = np.random.default_rng(0)
    vecs = {f"v{i}": rng.standard_normal(384).astype(np.float32) for i in range(8)}
    for vid, v in vecs.items():
        kg.add_node(id=vid, type="Vec", label="", embedding=v.tolist())  # empty label!
    kg.persist()
    before = {vid: kg.semantic_nodes(embedding=v.tolist(), k=5, node_type="Vec")
              for vid, v in vecs.items()}

    shutil.rmtree(tmp_path / "kg" / "index")          # drop the derived index
    kg.close()
    kg = KnowledgeGraph(tmp_path / "kg", retain_vectors=True,
                        _embedding_backend=HashEmbeddingBackend(native_dim=384))
    kg.reindex(); kg.persist()

    for vid, v in vecs.items():
        after = kg.semantic_nodes(embedding=v.tolist(), k=5, node_type="Vec")
        assert [n.id for _s, n in after] == [n.id for _s, n in before[vid]]   # identical ranking
        assert after[0][1] == pytest.approx(before[vid][0][1], abs=1e-6)      # identical score
        assert after[0][1] == [n.id for _s, n in after][0:1] and after[0][1] > 0.9  # self-recall
```

Add a negative control (`retain_vectors=False`) asserting `reindex()` now *reports*
the unrebuildable count rather than silently corrupting, and asserting the §2
lossy numbers (self-is-top1 == 0) so the regression is locked in. Also test
`search_subgraph` parity end-to-end, and the empty-label drop-out case (Demo B).

### 6.2 Throughput benchmark (`benchmarks/graph_memory/`)

Extend `graph_memory_bench.py` (the existing harness already builds a synthetic KG
and reports metrics-only, `graph_memory_bench.py:1-22`) with a `reindex`
sub-benchmark:

- Build a vector-in KG of `--graph-nodes` N (sweep 1e3 / 1e4 / 1e5), `persist()`.
- Time `reindex()` after dropping `index/`; report **nodes/s**, p50/p95 of the
  rebuild, and `indexed_documents` before/after (parity check).
- Compare three configs: `retain_vectors` (Option 1, re-`add_vectors`),
  text-label rebuild (today, as the lossy baseline), and — if prototyped — an
  Option-2 fast-path (artifacts-intact `add_encoded`). Surface the storage
  delta (`topology.sqlite3` size with/without `retain_vectors`) using the §4.4
  table as the expected ~8× ground truth.
- Keep it metrics-only (counts/bytes/latency) per the repo's benchmark provenance
  rule; no raw vectors in output.

---

## 7. Open questions

1. **Default posture.** `store_text` defaults to *on* (text retained); should
   `retain_vectors` match that for symmetry, or stay *off* given the 8× cost and
   that vector-in callers usually own their embedder? (Leaning off, with a loud
   `reindex()` report — but this is a product call.)
2. **Edges.** Index-edges (`index_edges=True`) with `embedding=` have the identical
   gap (`_index_edge`, `knowledge_graph.py:457-458`). The `entity_vectors(kind)`
   schema covers them; confirm edge facts want the same opt-in semantics.
3. **Option-2 fast-path scope.** Is it worth a graph-layer fast-path that, when the
   engine `.tvim`/`.tvd` are intact (fingerprint matches), skips re-quantization?
   That is really "don't drop the engine artifacts on a graph-only rebuild" — may
   be better solved by *not* dropping `index/` wholesale and instead reconciling
   via enumeration (which `reindex()` already does for orphans).
4. **Cross-store atomicity of the vector write.** The vector BLOB write to SQLite
   and the `add_vectors` engine commit are two writes; a crash between them is
   already handled by `reindex()` (it re-derives the index from SQLite, now
   including the vector) — but confirm the SQLite write is the *first* of the two
   so the source of truth is never behind the index.
5. **Quantization determinism across versions.** Option 1 promises rebuild ==
   original *for a fixed TurboVec build*. A TurboVec upgrade that changes encoding
   could shift quantized scores on rebuild even from identical f32. Acceptable
   (same as any re-quantization), but worth a documented note alongside the
   `calibration_fingerprint` portability rule.
