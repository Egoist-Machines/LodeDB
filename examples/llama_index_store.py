"""Use LodeDB as a LlamaIndex ``VectorStore``.

Needs the llama-index extra:  uv sync --extra llama-index
Run inside the project environment:

    uv run python examples/llama_index_store.py

This adapter is *text-path*: LodeDB embeds text internally (with the model set at
``LodeDB(model=...)``), so LlamaIndex's own ``embed_model`` is not used. The example below
talks to the store directly (add nodes, run a ``VectorStoreQuery``), which is the clearest
way to see that. To build through ``VectorStoreIndex`` instead, set a cheap
``Settings.embed_model = MockEmbedding(embed_dim=...)`` so LlamaIndex does not reach for a
remote embedding model — the vectors it computes are discarded; LodeDB re-embeds the text.

It also shows the two capabilities that map onto LodeDB's richer SDK: hybrid search
(``VectorStoreQueryMode.HYBRID`` -> BM25 + RRF, which recovers exact tokens the embedding
misses) and metadata filtering (``MetadataFilters`` -> LodeDB's predicate grammar).
"""

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)

from lodedb.local.integrations.llama_index import LodeDBVectorStore


def show(label: str, result) -> None:
    """Prints the ids/scores/text of one query result block."""

    print(label)
    for node, score in zip(result.nodes, result.similarities, strict=True):
        print(f"  {node.node_id}  score={score:.4f}  {node.get_content()!r}  {node.metadata}")


def main() -> None:
    store = LodeDBVectorStore.from_path("./data_llama_index")

    fox = TextNode(id_="fox", text="the quick brown fox jumps", metadata={"topic": "animals"})
    fox.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id="fables")
    dog = TextNode(id_="dog", text="a lazy dog sleeps all day", metadata={"topic": "animals"})
    dog.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id="fables")
    err = TextNode(id_="err", text="request failed with error E1234", metadata={"topic": "logs"})
    store.add([fox, dog, err])

    # Vector search (the default mode): semantic similarity.
    show("vector 'fox':", store.query(VectorStoreQuery(query_str="fox", similarity_top_k=2)))

    # Hybrid search recovers the exact token 'E1234' that a pure embedding can miss.
    show(
        "hybrid 'E1234':",
        store.query(
            VectorStoreQuery(
                query_str="E1234", similarity_top_k=2, mode=VectorStoreQueryMode.HYBRID
            )
        ),
    )

    # Metadata filter: only the 'animals' topic.
    animals = MetadataFilters(
        filters=[MetadataFilter(key="topic", value="animals", operator=FilterOperator.EQ)]
    )
    show(
        "filtered topic=animals:",
        store.query(VectorStoreQuery(query_str="sleepy", similarity_top_k=5, filters=animals)),
    )


if __name__ == "__main__":
    main()
