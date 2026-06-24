"""Use LodeDB as PrivateGPT's vector store.

Needs the llama-index extra:  uv sync --extra llama-index
Run inside the project environment:

    uv run python examples/privategpt_provider.py

PrivateGPT (zylon-ai/private-gpt) has no vector-store interface of its own: its store layer is
LlamaIndex's ``BasePydanticVectorStore``, selected by ``vectorstore.database`` in
``settings.yaml``. LodeDB already ships that interface (the ``lodedb[llama-index]`` adapter), so
the integration is a one-line registration plus one settings key — not a new adapter.

This script shows the registration call and prints the exact wiring you add inside a PrivateGPT
checkout. Because PrivateGPT is an application (not a pip library), it is not installed here, so
the script also drives the underlying ``LodeDBVectorStore`` directly to prove the store works.

To actually run it in PrivateGPT (inside a PrivateGPT clone whose environment also has
``pip install 'lodedb[llama-index]'``):

1. Trigger registration before PrivateGPT builds its ``VectorStoreComponent``. The smallest way
   is a tiny launcher that imports and registers, then starts PrivateGPT:

       # run_privategpt_lodedb.py (in the PrivateGPT repo root)
       from lodedb.local.integrations.privategpt import register_lodedb_provider
       register_lodedb_provider()
       from private_gpt.__main__ import main  # or: uvicorn private_gpt.main:app
       main()

   (Equivalently, import the module and call ``register_lodedb_provider()`` near the top of
   ``private_gpt/__main__.py``.)

2. Point PrivateGPT at LodeDB in ``settings.yaml`` (or via the ``PGPT_VECTORSTORE`` env var):

       vectorstore:
         database: lodedb
       # optional LodeDB block (defaults shown):
       lodedb:
         path: local_data/lodedb
         model: minilm        # "minilm" (fast) or "bge" (quality)
         device: auto         # auto | cpu | mps | cuda
         store_text: true     # keep on for hybrid/lexical retrieval
         index_text: false

   LodeDB embeds text itself (text-path), so keep PrivateGPT on a cheap/mock embedding to avoid
   a redundant embedding call; ``vectorstore.embed_dim`` is informational for this provider.
"""

from lodedb.local.integrations.llama_index import LodeDBVectorStore


def show_direct_store() -> None:
    """Drives the LodeDB-backed LlamaIndex store directly (what PrivateGPT ends up calling)."""

    from llama_index.core.schema import TextNode
    from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode

    store = LodeDBVectorStore.from_path("./data_privategpt")
    store.add(
        [
            TextNode(
                id_="d1", text="LodeDB keeps your data on your machine.", metadata={"src": "a"}
            ),
            TextNode(
                id_="d2", text="PrivateGPT runs a fully local RAG pipeline.", metadata={"src": "b"}
            ),
            TextNode(id_="d3", text="request failed with error E1234", metadata={"src": "c"}),
        ]
    )

    res = store.query(VectorStoreQuery(query_str="where does my data live?", similarity_top_k=2))
    print("vector query:")
    for node, score in zip(res.nodes, res.similarities, strict=True):
        print(f"  {node.node_id}  score={score:.4f}  {node.get_content()!r}")

    # Hybrid recovers the exact token an embedding can miss (PrivateGPT can request this mode).
    hybrid = store.query(
        VectorStoreQuery(query_str="E1234", similarity_top_k=2, mode=VectorStoreQueryMode.HYBRID)
    )
    print("hybrid query 'E1234':", [n.node_id for n in hybrid.nodes])


def main() -> None:
    # The real integration point: register the provider, then select it from settings.yaml.
    try:
        from lodedb.local.integrations.privategpt import register_lodedb_provider

        register_lodedb_provider()
        print("registered LodeDB as PrivateGPT vector store 'lodedb'")
    except ImportError as exc:
        # Expected when running outside a PrivateGPT checkout (PrivateGPT is an app, not a dep).
        print("PrivateGPT not importable here, so registration is a no-op in this demo:")
        print(f"  {exc}")
        print("Run this from a PrivateGPT clone to register the provider for real.")

    # Either way, show the underlying store working (this is what PrivateGPT drives).
    show_direct_store()


if __name__ == "__main__":
    main()
