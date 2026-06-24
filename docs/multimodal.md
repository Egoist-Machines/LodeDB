# Multimodal and bring-your-own vectors

LodeDB's storage and scan are modality-agnostic. The compact TurboVec index stores
any normalized float32 vector, so an image, audio, or video embedding is just a
vector of some dimension, indexed and scanned exactly like a text embedding. That
gives two ways to do multimodal search.

## Bring your own vectors (any model, any modality)

Open a vector-only index at your embedding dimension and hand LodeDB the vectors
you already computed. No embedding model is bundled or loaded on this path, so you
can use any encoder: CLIP, SigLIP, ImageBind, a code or biomedical text model, an
audio or video encoder, or a hosted embedding API.

```python
from lodedb import LodeDB

db = LodeDB.open_vector_store("./media", vector_dim=512)
db.add_vectors(image_vector, id="img-001", metadata={"path": "photos/img-001.jpg"})
hits = db.search_by_vector(text_or_image_query_vector, k=10)
```

This is the path mem0 and other vector-owning systems use. It gets the full
benefit of compact storage, the exact scan, delta persistence, metadata filters,
and (on CUDA hosts) the GPU-resident batch scan, with no embedding dependency.

## The CLIP preset (image and text, built in)

For the common image case, the `clip` preset embeds both images and text into one
shared space, so a text query can retrieve images and an image query can retrieve
images and text. It runs on the base sentence-transformers stack and needs only
Pillow for decoding image files:

```
uv pip install -e '.[image]'     # or: pip install 'lodedb[image]'
```

```python
from lodedb import LodeDB

db = LodeDB("./gallery", model="clip")           # downloads clip-ViT-B-32 on first use
db.add_image("photos/beach.jpg", metadata={"path": "photos/beach.jpg"})

db.search("a beach at sunset", k=5)              # text query, cross-modal
db.search_by_image("photos/beach.jpg", k=5)      # image query
```

`add_image` accepts a path, raw `bytes`, or a PIL `Image`. It embeds the image and
stores a single vector through the same atomic-commit path as `add_vectors`. The
raw image bytes are never stored: keep the file on disk (or object storage) and put
its path or URI in `metadata` so a hit can be resolved back to the image. A caption
can go in the optional `text=` argument.

## One embedding model per index

Similarity scores are only meaningful between vectors from the same model. Do not
mix encoders (or different versions of one encoder) in a single index. LodeDB pins
the index's model identity in the snapshot header and re-enforces it on reopen, so
an accidental dimension or model mismatch is caught rather than silently returning
meaningless scores.

To keep several encoders side by side, use a collection of named spaces.

## Named vector spaces

`LodeCollection` groups independent indexes ("spaces") under one directory, each
free to use a different model or dimension. Spaces are searched independently;
there is no cross-space scoring.

```python
from lodedb import LodeCollection

col = LodeCollection("./memory")
notes = col.space("text", model="minilm")
gallery = col.space("image", model="clip")

notes.add("design review notes for the new layout")
gallery.add_image("screens/layout.png", metadata={"path": "screens/layout.png"})
col.close()
```

The collection records each space's configuration in `collection.json` and reopens
it the same way.

## Custom embedder

To drive a text-capable index with your own model at any dimension, pass
`embedder=` an object implementing the embedding protocol (`embed_documents`,
`embed_query`, `native_dim`). This covers domain-specific text models, hosted
embedding APIs, and multimodal encoders. If the backend also exposes an
`embed_images` method, `add_image` and `search_by_image` work too.

```python
db = LodeDB("./store", embedder=my_backend)      # model= is ignored; shape from the backend
```

## Notes

- Cross-modal calibration: CLIP maps text and images into one space, but the two
  modalities do not occupy it identically. Cross-modal cosine scores are useful for
  ranking but are not directly comparable to text-to-text scores, so tune `k` and
  any score thresholds per query type.
- Encoder versions: a stored index is tied to its embedding model. Re-embed the
  corpus into a new index when you change models or model versions, rather than
  mixing old and new vectors.
- Audio and video: there is no bundled audio or video encoder. Embed with your own
  model (for example an audio CLAP model or a video encoder) and use the
  bring-your-own-vectors path.
- Late interaction: ColPali and ColQwen style multi-vector retrieval is a separate,
  larger effort tracked in the issue tracker; it does not fit the current
  one-vector-per-id index.
