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
| [`mcp_config.json`](mcp_config.json) | Register LodeDB as an MCP-capable agent's local memory (needs `--extra mcp`) | drop into your agent's MCP config |

Each script writes its index under a local `./data*` folder, which is git-ignored.

## MCP config note

`mcp_config.json` runs the `lodedb` CLI through `uv` against your cloned repo, since LodeDB
isn't on PyPI yet. Edit the two absolute paths. Once a packaged release exists, the entry
simplifies to:

```json
{ "mcpServers": { "lodedb": { "command": "lodedb", "args": ["mcp", "--path", "./data"] } } }
```
