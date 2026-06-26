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
index, with no engine change: each document's patches are stored as ordinary rows
keyed `<doc_id>#NNNNN` with a `parent_id` in metadata, and the full-precision
patch vectors are retained so scoring is **exact** (no quantization noise). By
default an unfiltered query is answered by a resident exact scan that holds every
patch in one in-memory matrix and scores the whole corpus in a single GEMM plus a
segmented max -- returning the true top-k at ~1-2 ms on a few thousand pages, with
no candidate-recall loss. See [Search paths](#search-paths) for when the indexed
(filtered / very large corpus) path is used instead.

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
as `LodeDB.search` (a filtered query uses the indexed path; see below):

```python
idx.search(query_tokens, k=5, filter={"file": "report.pdf"})
```

## Search paths

`search` picks one of two paths automatically; both return identical, exact
MaxSim scores.

- **Resident** (default for an unfiltered query within the memory budget): every
  patch is held in one in-memory float32 matrix and the whole corpus is scored in
  a single GEMM plus a segmented max. This returns the true top-k with no
  candidate-recall loss, at ~1-2 ms on a few thousand pages. The matrix is built
  on the first query (a one-time cost, reported as `resident_build_seconds` in the
  benchmark) and rebuilt after any write. Control it with `resident=True|False|"auto"`
  (default `"auto"`) and `resident_max_bytes` (default 512 MB); above the budget,
  `"auto"` falls back to the indexed path.
- **Indexed** (used for a filtered query, a corpus over the resident budget, or
  `resident=False`): the two-stage path -- a batched any-patch scan to depth
  `candidate_depth` gathers candidate documents (the filter is pushed engine-side),
  then exact MaxSim rescores them.

This is a change from the original two-stage-only prototype, which ran ~110 ms/query
at recall 0.73 (a profile showed candidate generation ~61% and patch loading ~38%
of that, with scoring ~1%). The resident path removes both costs: ~1.5 ms/query and
recall 1.0.

## Scoring backend

The exact-MaxSim computation has two backends, selected with `scoring=` on the
constructor; both return identical scores and apply to either search path:

- `scoring="numpy"` (default): a `query @ patches.T` GEMM via numpy. On a build
  with an optimized BLAS (Apple Accelerate, OpenBLAS) this is the fastest path and
  needs no compiled kernel.
- `scoring="native"`: the TurboVec `maxsim_scores` Rust kernel (per-document faer
  GEMM, parallel across documents, GIL released). Provided for builds without a
  fast BLAS; falls back to numpy if the compiled kernel is absent.

## Footprint and what's next (issue #25)

A document contributes one stored row per patch, so a page with ~1000 patches is
~1000 rows, plus the retained float32 patch sidecar the exact scan reads back, and
the resident matrix is full-precision in memory. That footprint -- ~400x a
single-vector index -- is now the main open item: native quantized multi-vector
**storage** in the TurboVec core (plus patch pooling on the encoder side) is what
would shrink it and extend the fast exact path to corpora past the resident budget.
The MaxSim scoring kernel is already native (`scoring="native"`). Tracked in
[issue #25](https://github.com/Egoist-Machines/LodeDB/issues/25).

## Tuning

`candidate_depth` (default 16) applies to the **indexed** path only (the resident
path is exhaustive, so it is always exact). It is the per-query-token any-patch
search depth used to gather candidates; higher values raise recall on the indexed
path at the cost of more rescoring work. Pass `candidate_depth=` to the constructor
or per call to `search`.
