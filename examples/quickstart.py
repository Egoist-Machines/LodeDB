"""LodeDB quickstart — add, search, batch-search, fetch text, persist.

The first run downloads the embedding model from Hugging Face (then caches it locally).
Run inside the project environment:

    uv run python examples/quickstart.py
"""

from lodedb import LodeDB


def main() -> None:
    # "minilm" (all-MiniLM-L6-v2, fast) | "bge" (BAAI/bge-base-en-v1.5, higher quality)
    db = LodeDB(path="./data", model="minilm")

    fox = db.add("the quick brown fox jumps", metadata={"topic": "animals"})
    db.add("a lazy dog sleeps all day", metadata={"topic": "animals"})

    print("search('fox'):")
    for score, doc_id, meta in db.search("fox", k=5):
        print(f"  {score:.3f}  {doc_id}  {meta}")

    print("search_many(['fox', 'dog'])  # batched path; CUDA can serve it:")
    for query, hits in zip(["fox", "dog"], db.search_many(["fox", "dog"], k=5), strict=True):
        print(f"  {query!r}: {[(round(h.score, 3), h.id) for h in hits]}")

    # Raw text is retained by default; pass store_text=False to keep none on disk.
    print("get(fox) ->", db.get(fox))

    db.persist()  # durable .tvim/.tvd/.jsd (+ .tvtext) snapshot; replays on reopen
    print(f"persisted {db.count()} documents to ./data")


if __name__ == "__main__":
    main()
