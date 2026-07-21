"""Optional framework adapters for the local LodeDB layer.

Each adapter is import-guarded behind its framework's optional extra, so this
package imports cleanly without those heavy deps installed:

- ``langchain``: ``langchain.LodeDBVectorStore`` (``pip install 'lodedb[langchain]'``).
- ``llama-index``: ``llama_index.LodeDBVectorStore`` and
  ``llama_index_graph.LodeDBPropertyGraphStore`` (``pip install 'lodedb[llama-index]'``).
- ``mem0``: ``mem0.LodeDBVectorStore`` (``pip install 'lodedb[mem0]'``).
- ``cognee``: ``cognee.CogneeLodeDBAdapter`` + ``cognee.register_cognee_adapter``
  (``pip install 'lodedb[cognee]'``).
- ``privategpt``: ``privategpt.register_lodedb_provider`` registers the LlamaIndex adapter as a
  PrivateGPT vector-store provider (needs the ``llama-index`` extra inside a PrivateGPT
  environment). It is a provider shim, not a new adapter: PrivateGPT's store layer *is*
  LlamaIndex's ``BasePydanticVectorStore``, which the LlamaIndex adapter already implements.
- ``kotaemon``: ``kotaemon.LodeDBVectorStore`` implements kotaemon's ``BaseVectorStore``
  contract by duck typing (no extra needed; the module imports neither kotaemon nor
  llama-index). Selected from kotaemon via ``KH_VECTORSTORE`` in ``flowsettings.py``.

All wrap a LodeDB handle. The LangChain and LlamaIndex adapters are text-path. LodeDB embeds
text internally (``is_embedding_query=False``) and the framework's own embedding model is not
used. The mem0 adapter is vector-in: mem0 owns the embeddings, LodeDB stores and searches them,
and the full mem0 payload JSON is retained in LodeDB's raw-text sidecar (never in redacted
metadata). The cognee adapter (``CogneeLodeDBAdapter``) is vector-in the same way (cognee owns
the embeddings via its ``EmbeddingEngine``), with one LodeDB index per cognee collection, the
serialized DataPoint payload in the raw-text sidecar, and ``belongs_to_set`` membership stored as
scalar presence keys so cognee's ``node_name`` filtering pushes into the metadata planner. The
``LodeDBPropertyGraphStore`` wraps :class:`lodedb.graph.KnowledgeGraph` instead
of the flat :class:`LodeDB` SDK, exposing the graph layer to LlamaIndex's ``PropertyGraphIndex``.
The PrivateGPT provider reuses the text-path LlamaIndex adapter and creates one local LodeDB
index per PrivateGPT collection. The kotaemon adapter is vector-in (kotaemon owns the
embeddings) with one lazily-shaped LodeDB collection per kotaemon index under the configured
``path``; chunk text stays in kotaemon's docstore.
"""
