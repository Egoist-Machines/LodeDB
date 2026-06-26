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
index, with no engine change: each document's patches are stored as ordinary
rows keyed `<doc_id>#NNNNN` with a `parent_id` in metadata, retrieval gathers
candidate documents by any-patch similarity over the existing TurboVec scan, and
the **exact** MaxSim score is recomputed over the candidate set. The compact
quantized index is used only to surface candidates; the full-precision patch
vectors are retained so the final ranking does not inherit quantization noise.

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

Metadata filters narrow the candidate scan by your document metadata, with the
same grammar as `LodeDB.search`:

```python
idx.search(query_tokens, k=5, filter={"file": "report.pdf"})
```

## Footprint

A document contributes one stored row per patch, so a page with ~1000 patches is
~1000 rows, plus the retained float32 patch sidecar the exact rescore reads back.
That is the known cost of late interaction, and the reason native multi-vector
storage and a MaxSim kernel in the TurboVec core are a separate, benchmark-gated
track ([issue #25](https://github.com/Egoist-Machines/LodeDB/issues/25)). Use
this prototype to validate retrieval quality and the API shape first; patch
pooling on the encoder side keeps the row count down.

## Tuning

`candidate_depth` (default 16) is the per-query-token any-patch search depth used
to gather rescoring candidates. Higher values raise recall at the cost of more
rescoring work; pass `candidate_depth=` to the constructor or per call to
`search`.
