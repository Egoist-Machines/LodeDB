"""Optional framework adapters for the local LodeDB layer.

Each adapter is import-guarded behind its framework's optional extra, so this
package imports cleanly without those heavy deps installed:

- ``langchain`` — :class:`LodeDBVectorStore` (``pip install 'lodedb[langchain]'``).

LlamaIndex follows the same shape (wrap the :class:`LodeDB` SDK) and is a
straightforward follow-up.
"""
