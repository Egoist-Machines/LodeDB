"""Optional framework adapters for the local LodeDB layer.

Each adapter is import-guarded behind its framework's optional extra, so this
package imports cleanly without those heavy deps installed:

- ``langchain`` — ``langchain.LodeDBVectorStore`` (``pip install 'lodedb[langchain]'``).
- ``llama-index`` — ``llama_index.LodeDBVectorStore`` and
  ``llama_index_graph.LodeDBPropertyGraphStore`` (``pip install 'lodedb[llama-index]'``).
- ``mem0`` — ``mem0.LodeDBVectorStore`` (``pip install 'lodedb[mem0]'``).

All wrap a LodeDB handle. The LangChain and LlamaIndex adapters are text-path — LodeDB embeds
text internally (``is_embedding_query=False``) and the framework's own embedding model is not
used. The mem0 adapter is vector-in: mem0 owns the embeddings, LodeDB stores and searches them,
and the full mem0 payload JSON is retained in LodeDB's raw-text sidecar (never in redacted
metadata). The ``LodeDBPropertyGraphStore`` wraps :class:`lodedb.graph.KnowledgeGraph` instead
of the flat :class:`LodeDB` SDK, exposing the graph layer to LlamaIndex's ``PropertyGraphIndex``.
"""
