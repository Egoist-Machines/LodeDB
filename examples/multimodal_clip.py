"""Multimodal LodeDB: image and text search two ways.

1. The built-in CLIP preset (``model="clip"``): index images and query them with
   text or with another image, over one shared embedding space. Needs the image
   extra:

       uv pip install -e '.[image]'   # adds Pillow; CLIP rides the base torch stack

2. Bring-your-own vectors (``open_vector_store``): LodeDB stores any normalized
   float32 vectors, so you can embed with any model you like (CLIP, SigLIP,
   ImageBind, an audio or video encoder) and hand LodeDB the vectors. No embedding
   model is bundled for this path.

Run inside the project environment:

    uv run python examples/multimodal_clip.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from lodedb import LodeDB


def clip_demo(workdir: Path) -> None:
    """Indexes a few images with the CLIP preset and queries them by text and image."""

    try:
        from PIL import Image
    except ImportError:
        print("clip_demo skipped: install the image extra with `uv pip install -e '.[image]'`")
        return

    # Stand-in images: solid color tiles. Real use indexes your own image files.
    swatches = {"red": (220, 20, 20), "green": (20, 200, 20), "blue": (20, 20, 220)}
    image_dir = workdir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, rgb in swatches.items():
        path = image_dir / f"{name}.png"
        Image.new("RGB", (96, 96), rgb).save(path)
        paths[name] = path

    # The first run downloads sentence-transformers/clip-ViT-B-32 from Hugging Face.
    db = LodeDB(path=workdir / "clip_store", model="clip")
    for name, path in paths.items():
        # Store only the vector; keep the file path in metadata to resolve a hit.
        db.add_image(str(path), id=name, metadata={"path": str(path), "color": name})
    print(f"indexed {db.count()} images with model='clip'")

    # Cross-modal: a text query searches the same space the images live in.
    print("search('a red square'):")
    for score, doc_id, meta in db.search("a red square", k=3):
        print(f"  {score:.3f}  {doc_id}  {meta['path']}")

    # Image-to-image: find the images most similar to a query image.
    print("search_by_image(red.png):")
    for hit in db.search_by_image(str(paths["red"]), k=2):
        print(f"  {hit.score:.3f}  {hit.id}")

    db.persist()


def bring_your_own_vectors_demo(workdir: Path) -> None:
    """Stores precomputed embeddings directly, with no bundled embedding model.

    The vectors here are illustrative one-hot rows; in practice they come from
    whatever model you choose. Keep one embedding model per index: scores are only
    comparable within a single shared space.
    """

    dim = 8
    db = LodeDB.open_vector_store(workdir / "byov_store", vector_dim=dim)

    def one_hot(i: int) -> list[float]:
        vector = [0.0] * dim
        vector[i] = 1.0
        return vector

    db.add_vectors(one_hot(0), id="cat", metadata={"caption": "a cat"})
    db.add_vectors(one_hot(1), id="dog", metadata={"caption": "a dog"})
    db.add_vectors(one_hot(2), id="car", metadata={"caption": "a car"})
    print(f"\nindexed {db.count()} bring-your-own vectors (dim={dim})")

    hit = db.search_by_vector(one_hot(1), k=1)[0]
    print(f"nearest to the 'dog' vector: {hit.id} ({hit.metadata['caption']})")
    db.persist()


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        clip_demo(workdir)
        bring_your_own_vectors_demo(workdir)


if __name__ == "__main__":
    main()
