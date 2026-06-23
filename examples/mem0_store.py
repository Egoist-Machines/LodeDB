"""Use LodeDB as a mem0 ``VectorStoreBase`` backend.

Needs the mem0 extra:  uv sync --extra mem0
Run inside the project environment:

    uv run python examples/mem0_store.py

The intended integration point is ``register_mem0_provider()`` plus
``Memory.from_config(...)`` (shown at the bottom); that path also needs an LLM and
embedder configured (e.g. an ``OPENAI_API_KEY``), so this script first drives the
adapter directly with deterministic one-hot vectors so it runs with no API key.
"""

import os

from lodedb.local.integrations.mem0 import LodeDBVectorStore, register_mem0_provider

DIM = 8


def _onehot(index: int) -> list[float]:
    vector = [0.0] * DIM
    vector[index] = 1.0
    return vector


def main() -> None:
    # Registers the "lodedb" provider with mem0's vector-store factory so it can be
    # selected from a Memory config (see the from_config snippet below).
    register_mem0_provider()

    store = LodeDBVectorStore(
        path="./data_mem0",
        collection_name="memories",
        embedding_model_dims=DIM,
    )
    store.insert(
        vectors=[_onehot(0), _onehot(1)],
        ids=["alice", "project"],
        payloads=[
            {
                "data": "Alice likes espresso",
                "user_id": "u1",
                "text_lemmatized": "alice likes espresso",
            },
            {
                "data": "The project uses LodeDB",
                "user_id": "u1",
                "text_lemmatized": "project uses lodedb",
            },
        ],
    )

    vector_hits = [
        (hit.id, round(hit.score or 0.0, 3), hit.payload)
        for hit in store.search("alice", _onehot(0))
    ]
    keyword_hits = [(hit.id, round(hit.score or 0.0, 3)) for hit in store.keyword_search("lodedb")]
    print("vector :", vector_hits)
    print("keyword:", keyword_hits)
    store.close()

    # The real integration: wire LodeDB into mem0 via config. This needs an LLM +
    # embedder (mem0 extracts and embeds memories), so it only runs with a key set.
    if os.environ.get("OPENAI_API_KEY"):
        from mem0 import Memory

        memory = Memory.from_config(
            {
                "vector_store": {
                    "provider": "lodedb",
                    "config": {
                        "path": "./data_mem0_memory",
                        "collection_name": "memories",
                        "embedding_model_dims": 1536,
                    },
                }
            }
        )
        memory.add("Alice prefers espresso over tea", user_id="u1")
        print("mem0   :", memory.search("What does Alice drink?", user_id="u1"))
    else:
        print("(set OPENAI_API_KEY to run the full Memory.from_config flow)")


if __name__ == "__main__":
    main()
