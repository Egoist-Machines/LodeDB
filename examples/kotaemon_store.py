"""Use LodeDB as a kotaemon vector store.

The adapter is dependency-free (it duck-types kotaemon's objects), so this
script needs no extra install and runs in the base project environment:

    uv run python examples/kotaemon_store.py

Inside a kotaemon deployment the integration is a settings change only — no
kotaemon fork, no registration call. Point ``KH_VECTORSTORE`` at the adapter in
``flowsettings.py``:

    KH_VECTORSTORE = {
        "__type__": "lodedb.local.integrations.kotaemon.LodeDBVectorStore",
        "path": str(KH_USER_DATA_DIR / "vectorstore"),
    }

kotaemon's ``get_vectorstore`` then builds one LodeDB collection per index
(``<path>/<collection_name>``). kotaemon owns the embeddings, so this script
drives the same interface directly with deterministic one-hot vectors.
"""

from lodedb.local.integrations.kotaemon import LodeDBVectorStore

DIM = 8


def _onehot(index: int) -> list[float]:
    vector = [0.0] * DIM
    vector[index] = 1.0
    return vector


def main() -> None:
    # kotaemon injects collection_name per index; path comes from KH_VECTORSTORE.
    store = LodeDBVectorStore(path="./data_kotaemon", collection_name="index_1")

    # kotaemon's ingestion pipeline: add(embeddings=..., ids=[chunk ids]) with the
    # chunk metadata (file_id backs the app's selected-files filter).
    store.add(
        embeddings=[_onehot(0), _onehot(1), _onehot(2)],
        metadatas=[{"file_id": "f1"}, {"file_id": "f1"}, {"file_id": "f2"}],
        ids=["chunk-a", "chunk-b", "chunk-c"],
    )
    print("stored:", store.count())

    # kotaemon's retrieval: query(embedding, top_k, doc_ids=<chunk scope>).
    _, scores, ids = store.query(embedding=_onehot(1), top_k=2)
    rounded = [round(score, 3) for score in scores]
    print("query top hits:", list(zip(ids, rounded, strict=True)))

    _, _, scoped = store.query(embedding=_onehot(1), top_k=2, doc_ids=["chunk-c"])
    print("scoped to chunk-c:", scoped)

    store.delete(["chunk-a"])
    print("after delete:", store.count())

    # drop() removes the collection from disk (kotaemon calls it when an index
    # is deleted from the UI).
    store.drop()
    print("after drop:", store.count())


if __name__ == "__main__":
    main()
