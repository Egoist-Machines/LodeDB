# Late-interaction (multi-vector / MaxSim) retrieval

Late interaction (ColBERT, and its visual-document descendants ColPali / ColQwen)
represents each document, or each page rendered as an image, as a *set* of
token/patch vectors instead of one pooled vector, and ranks with **MaxSim**:

```
score(query, doc) = sum over query tokens of  max over doc patches of  <q, d>
```

This is the leading approach for visual-document RAG, where a page is encoded as
roughly a thousand patch vectors and retrieval runs over the patches directly,
without OCR or layout parsing.

`LodeLateInteractionIndex` runs this on top of a bring-your-own-vectors LodeDB
index, with no engine change. Each document is **one row**: its id is the document
id, its vector is the mean-pooled patch vector, and its full patch matrix is kept
in the per-row text sidecar (at the `storage` precision). An unfiltered query is
answered by a resident exact scan that holds every patch in one in-memory matrix
and scores the whole corpus in a single GEMM plus a segmented max -- returning the
true top-k at a few milliseconds on thousands of pages, with no candidate-recall
loss. See [Search paths](#search-paths) and [Storage precision](#storage-precision).

## Encoder is bring-your-own

ColPali / ColQwen weights are multi-GB, so the page/token encoder is not bundled.
Pass precomputed patch matrices, or an `encoder` exposing `encode_documents` /
`encode_queries`. The patch dimension must be a multiple of 8 (the TurboVec store
requirement).

```python
from lodedb import LodeLateInteractionIndex

idx = LodeLateInteractionIndex("./pages", dim=128)

# page_patches: a (num_patches, 128) matrix from your ColPali encoder
idx.add_document("report-p1", page_patches, metadata={"file": "report.pdf"})
idx.persist()

# query_tokens: a (num_query_tokens, 128) matrix for the query
for score, doc_id, meta in idx.search(query_tokens, k=5):
    print(score, doc_id, meta)
```

With an encoder, the text/content convenience path is available:

```python
idx = LodeLateInteractionIndex("./pages", dim=128, encoder=my_colpali)
idx.add_texts([{"id": "p1", "content": page_image, "metadata": {"file": "r.pdf"}}])
hits = idx.search_text(query_string, k=5)
```

Metadata filters narrow retrieval by your document metadata, with the same grammar
as `LodeDB.search`:

```python
idx.search(query_tokens, k=5, filter={"file": "report.pdf"})
```

## Search paths

`search` picks one of three exact paths automatically; all return the true top-k.

- **Resident** (default, unfiltered, within the memory budget): every patch is held
  in one in-memory matrix and the whole corpus is scored in a single GEMM plus a
  segmented max -- a few milliseconds on thousands of pages. Built on the first
  query (`resident_build_seconds` in the benchmark) and rebuilt after any write.
  Control with `resident=True|False|"auto"` (default `"auto"`) and
  `resident_max_bytes` (default 512 MB).
- **Filtered**: a query with a `filter` resolves the matching documents
  engine-side and scores that subset exhaustively -- so a metadata filter both
  narrows and stays exact.
- **Streaming**: a corpus over the resident budget (or `resident=False`) is scored
  by reading documents back from disk in bounded chunks -- slower (disk-bound) but
  exact and constant-memory, so the exact path is never capped by RAM.

The original prototype instead ran a two-stage quantized-candidate-then-rescore
query: ~110 ms/query at recall 0.73 (profile: candidate generation ~61%, patch
read-back ~38%, scoring ~1%). The resident scan removes both dominant costs.

## Storage precision

Each document's patch matrix is stored at a precision chosen with `storage=`.

| storage | size vs float32 | recall | notes |
|---|---|---|---|
| `"float32"` (default) | 1x | exact | bit-exact; fastest query |
| `"float16"` | 0.5x | ~exact (1.0 in the benchmark) | half the disk and RAM |
| `"int8"` | 0.25x | ~0.98 | per-vector-scaled; smallest |

The default is `float32` for its query speed and bit-exactness; choose `float16`
or `int8` when footprint matters more. The precision is **persisted with the
index** (in a small `lodedb_late_interaction.meta` sidecar), so you set it once at
creation and reopen without re-passing it:

```python
LodeLateInteractionIndex("./pages", dim=128, storage="int8")  # create as int8
...
LodeLateInteractionIndex("./pages", dim=128)                  # reopens as int8
```

Leave `storage=None` (the default) to adopt the index's stored precision, or
float32 for a brand-new index. Passing a value that disagrees with the stored one
raises `ValueError`, so an index keeps a single precision; every document also
records its own precision, so decoding is always correct.

Combined with the one-row-per-document layout, float16 is ~7x smaller on disk than
the original one-row-per-patch float32 prototype and keeps 2x more pages resident
in RAM; int8 is ~14x smaller. Scoring upcasts to float32 in bounded chunks, so a
compact resident matrix never forces a full-precision copy of the whole corpus.

## Scoring backend

The exact-MaxSim computation has two backends, selected with `scoring=`; both
return identical scores:

- `scoring="numpy"` (default): a `query @ patches.T` GEMM via numpy. On a build
  with an optimized BLAS (Apple Accelerate, OpenBLAS) this is the fastest path and
  needs no compiled kernel.
- `scoring="native"`: the TurboVec `maxsim_scores` Rust kernel (per-document faer
  GEMM, parallel across documents, GIL released). Provided for builds without a
  fast BLAS; falls back to numpy if the compiled kernel is absent.

See [issue #25](https://github.com/Egoist-Machines/LodeDB/issues/25) for the
broader late-interaction track.
