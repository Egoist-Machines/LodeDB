"""Use LodeDB as a LlamaIndex ``PropertyGraphStore``.

Needs the llama-index extra:  uv sync --extra llama-index
Run inside the project environment:

    uv run python examples/llama_index_graph_store.py

This wraps LodeDB's hybrid knowledge-graph layer (SQLite topology + LodeDB semantic index) as
a LlamaIndex ``PropertyGraphStore``. Like the vector-store adapter it is text-path: LodeDB
embeds node text (an entity's name, a chunk's text) with its own model, so LlamaIndex's
``embed_model`` is not used for storage. The example talks to the store directly (upsert nodes
and relations, traverse, run a semantic query), which is the clearest way to see the mapping.
"""

from llama_index.core.graph_stores.types import ChunkNode, EntityNode, Relation
from llama_index.core.vector_stores.types import VectorStoreQuery

from lodedb.local.integrations.llama_index_graph import LodeDBPropertyGraphStore


def main() -> None:
    store = LodeDBPropertyGraphStore.from_path("./data_llama_index_graph")

    store.upsert_nodes(
        [
            EntityNode(name="alice", label="PERSON", properties={"role": "engineer"}),
            EntityNode(name="acme", label="ORG"),
            ChunkNode(text="Alice works at Acme on the search team.", id_="c1"),
        ]
    )
    store.upsert_relations(
        [
            Relation(label="WORKS_AT", source_id="alice", target_id="acme"),
            Relation(label="MENTIONS", source_id="c1", target_id="alice"),
        ]
    )

    print("triplets around alice:")
    for src, rel, dst in store.get_triplets(entity_names=["alice"]):
        print(f"  ({src.id}) -[{rel.label}]-> ({dst.id})")

    print("rel map (depth 2) from alice:")
    for src, rel, dst in store.get_rel_map([EntityNode(name="alice", label="PERSON")], depth=2):
        print(f"  ({src.id}) -[{rel.label}]-> ({dst.id})")

    print("semantic node search for 'engineer':")
    nodes, scores = store.vector_query(
        VectorStoreQuery(query_str="who is the engineer?", similarity_top_k=3)
    )
    for node, score in zip(nodes, scores, strict=True):
        print(f"  {node.id}  score={score:.4f}  {node.properties}")


if __name__ == "__main__":
    main()
