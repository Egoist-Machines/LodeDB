# Multimodal image-vector storage benchmark

**Question:** as the store behind image/multimodal retrieval, how does LodeDB compare
to Chroma and Qdrant on ingest, on-disk footprint, query latency, and recall, holding
the embeddings fixed?

Every store is fed the **same** precomputed CLIP-dimension vectors (512-d, the
`clip-ViT-B-32` size), so this isolates storage and scan from the encoder. Recall is
measured for every store against the exact brute-force top-k of the identical vectors,
so the metric is comparable across backends. All artifacts are metrics-only (counts,
bytes, latency, recall, backend labels).

The real end-to-end image path (CLIP encode, `add_image`, cross-modal text search) is
shown in [`examples/multimodal_clip.py`](../../examples/multimodal_clip.py); this
benchmark fixes the vectors on purpose so the comparison is apples-to-apples.

## Run

```bash
# LodeDB only (no extra installs):
uv run python benchmarks/multimodal_image/run.py

# include competitors when installed:
uv pip install chromadb qdrant-client
uv run python benchmarks/multimodal_image/run.py --n 5000 --queries 200
```

Flags: `--n` corpus size, `--queries` query count, `--k` top-k, `--dim` embedding
dimension, `--seed`.

## Results

`measured` on Apple Silicon (local), `--n 1500 --queries 100 --k 10 --dim 512`, on
synthetic unit-Gaussian vectors:

| Backend | Footprint (MB) | Mean query (ms) | p95 query (ms) | Recall@10 | Ingest (s) |
|---|---:|---:|---:|---:|---:|
| LodeDB | 0.81 | 0.17 | 0.20 | 0.876 | 0.64 |
| Chroma | 9.88 | 1.20 | 1.66 | 0.929 | 0.36 |
| Qdrant | 7.43 | 1.23 | 2.21 | 1.000 | 1.13 |

Reading it honestly:

- **Footprint is the clear win:** LodeDB's compact 4-bit codes are roughly 9-12x
  smaller on disk than Chroma or Qdrant at the same vector count and dimension.
- **Query latency is lowest** here (exact SIMD scan over a small corpus), about 7x
  faster than either competitor on this run.
- **Recall is lower**, which is the expected cost of 4-bit quantization: Qdrant (HNSW
  over fp32) returns the exact neighbors, and Chroma is higher than LodeDB. Note this
  is a worst case for quantization recall: unit-Gaussian vectors are spread evenly with
  no cluster structure, whereas real CLIP embeddings cluster and quantize with higher
  recall. Raise `bit_width` for more recall at a larger footprint.
- **Ingest is competitive, not best:** Chroma ingests faster on this run; LodeDB sits
  between Chroma and Qdrant.

Numbers vary with machine, corpus size, and (for the ANN stores) index parameters;
rerun locally for your own. LodeDB's exact scan is O(N) per query, so its latency
advantage narrows as the corpus grows into the range where ANN indexes are designed to
win.
