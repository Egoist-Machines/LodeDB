"""Use LodeDB as a cognee ``vector_db_provider``.

Needs the cognee extra:  uv sync --extra cognee
Run inside the project environment:

    uv run python examples/cognee_provider.py

The intended integration point is ``register_cognee_adapter()`` plus
``cognee.config.set_vector_db_config(...)`` (shown at the bottom); the full
``cognee.add`` / ``cognee.cognify`` pipeline also needs an LLM + embedder configured
(e.g. an ``OPENAI_API_KEY``), so this script first drives the adapter directly with a
deterministic one-hot embedding engine so it runs with no API key.
"""

import asyncio

from cognee.infrastructure.engine import DataPoint

from lodedb.local.integrations.cognee import CogneeLodeDBAdapter, register_cognee_adapter

DIM = 8


class OneHotEmbeddingEngine:
    """Deterministic text -> one-hot embedding, so the demo needs no model/key."""

    def __init__(self) -> None:
        self._index: dict[str, int] = {}

    async def embed_text(self, text: list[str]) -> list[list[float]]:
        vectors = []
        for value in text:
            self._index.setdefault(value, len(self._index))
            vector = [0.0] * DIM
            vector[self._index[value]] = 1.0
            vectors.append(vector)
        return vectors

    def get_vector_size(self) -> int:
        return DIM

    def get_batch_size(self) -> int:
        return 32


class Note(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"]}


async def main() -> None:
    # Registers the "lodedb" provider with cognee's vector-adapter registry so it can
    # be selected from cognee config (see the set_vector_db_config snippet below).
    register_cognee_adapter()

    adapter = CogneeLodeDBAdapter(url="./data_cognee", embedding_engine=OneHotEmbeddingEngine())
    notes = [
        Note(text="Alice likes espresso", belongs_to_set=["people"]),
        Note(text="The project uses LodeDB", belongs_to_set=["project"]),
    ]
    await adapter.create_data_points("Note_text", notes)

    hits = await adapter.search(
        "Note_text", query_text="Alice likes espresso", include_payload=True
    )
    print("search :", [(str(hit.id), round(hit.score, 3), hit.payload["text"]) for hit in hits])

    scoped = await adapter.search(
        "Note_text", query_text="Alice likes espresso", node_name=["project"]
    )
    print("scoped :", [str(hit.id) for hit in scoped])  # only the 'project' note
    adapter.close()

    # The real integration: point cognee at LodeDB via config. The add/cognify
    # pipeline needs an LLM + embedder, so this snippet is illustrative:
    #
    #   import cognee
    #   register_cognee_adapter()
    #   cognee.config.set_vector_db_config(
    #       {"vector_db_provider": "lodedb", "vector_db_url": "./data_cognee"}
    #   )
    #   await cognee.add("Alice likes espresso")
    #   await cognee.cognify()
    #   print(await cognee.search("What does Alice like?"))


if __name__ == "__main__":
    asyncio.run(main())
