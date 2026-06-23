"""Optional framework adapters for the local LodeDB layer.

Each adapter is import-guarded behind its framework's optional extra, so this
package imports cleanly without those heavy deps installed:

- ``langchain`` — ``langchain.LodeDBVectorStore`` (``pip install 'lodedb[langchain]'``).
- ``llama-index`` — ``llama_index.LodeDBVectorStore`` and
  ``llama_index_graph.LodeDBPropertyGraphStore`` (``pip install 'lodedb[llama-index]'``).

All wrap a LodeDB handle, so LodeDB embeds text internally and the framework's own embedding
model is not used (the LlamaIndex adapters are text-path, ``is_embedding_query=False``). The
``LodeDBPropertyGraphStore`` wraps :class:`lodedb.graph.KnowledgeGraph` instead of the flat
:class:`LodeDB` SDK, exposing the graph layer to LlamaIndex's ``PropertyGraphIndex``.
"""
