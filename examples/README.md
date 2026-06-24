# Examples

Runnable examples for LodeDB. They assume you've installed from source (see the repo
[README](../README.md#install)) and run them inside the project environment with `uv run`.
The first run downloads the embedding model from Hugging Face and caches it locally.

| File | What it shows | Run |
|---|---|---|
| [`quickstart.py`](quickstart.py) | Core SDK: add, `search`, batched `search_many`, `get`, `persist` | `uv run python examples/quickstart.py` |
| [`langchain_store.py`](langchain_store.py) | LodeDB as a LangChain `VectorStore` (needs `--extra langchain`) | `uv run python examples/langchain_store.py` |
| [`llama_index_store.py`](llama_index_store.py) | LodeDB as a LlamaIndex `VectorStore` (needs `--extra llama-index`) | `uv run python examples/llama_index_store.py` |
| [`llama_index_graph_store.py`](llama_index_graph_store.py) | LodeDB as a LlamaIndex `PropertyGraphStore` (needs `--extra llama-index`) | `uv run python examples/llama_index_graph_store.py` |
| [`mem0_store.py`](mem0_store.py) | LodeDB as a mem0 `VectorStoreBase` backend (needs `--extra mem0`) | `uv run python examples/mem0_store.py` |
| [`privategpt_provider.py`](privategpt_provider.py) | Register LodeDB as PrivateGPT's vector store (needs `--extra llama-index`) | `uv run python examples/privategpt_provider.py` |
| [`mcp_config.json`](mcp_config.json) | Register LodeDB as an MCP-capable agent's local memory (needs `--extra mcp`) | drop into your agent's MCP config |

Each script writes its index under a local `./data*` folder, which is git-ignored.

## MCP config note

`pip install lodedb` puts the `lodedb` CLI on your `PATH`, so the normal MCP entry is just:

```json
{ "mcpServers": { "lodedb": { "command": "lodedb", "args": ["mcp", "--path", "./data"] } } }
```

If you installed into a virtual environment (including a `uv` project) where `lodedb` isn't on
`PATH`, use the `uv run --project` form in [`mcp_config.json`](mcp_config.json) instead and edit
the two absolute paths. The [main README](../README.md#use-as-an-mcp-server) also covers `lodedb
mcp install`, which writes the entry for you.
