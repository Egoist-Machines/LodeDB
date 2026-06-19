"""Use LodeDB as a LangChain ``VectorStore``.

Needs the langchain extra:  uv sync --extra langchain
Run inside the project environment:

    uv run python examples/langchain_store.py

LodeDB embeds internally, so no LangChain ``Embeddings`` object is required. Because the
store retains text by default, ``page_content`` is populated both in-session and after a
reopen (it falls back to the ``.tvtext`` sidecar).
"""

from lodedb.local.integrations.langchain import LodeDBVectorStore


def main() -> None:
    store = LodeDBVectorStore.from_texts(
        ["the quick brown fox jumps", "a lazy dog sleeps all day"],
        path="./data_langchain",
        metadatas=[{"topic": "animals"}, {"topic": "animals"}],
    )
    for doc in store.similarity_search("fox", k=2):
        print(f"  {doc.metadata.get('id')}  {doc.page_content!r}  {doc.metadata}")


if __name__ == "__main__":
    main()
