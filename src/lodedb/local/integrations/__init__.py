"""Optional framework adapters for the local LodeDB layer.

Each adapter is import-guarded behind its framework's optional extra, so this
package imports cleanly without those heavy deps installed:

- ``langchain`` — ``langchain.LodeDBVectorStore`` (``pip install 'lodedb[langchain]'``).
- ``llama-index`` — ``llama_index.LodeDBVectorStore`` (``pip install 'lodedb[llama-index]'``).

Both wrap the :class:`LodeDB` SDK, so LodeDB embeds text internally and the framework's
own embedding model is not used (the LlamaIndex adapter is text-path,
``is_embedding_query=False``).
"""
