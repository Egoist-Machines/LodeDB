"""The migration plan: a payload-free description of what a run will do.

A plan is produced by ``lodedb migrate plan`` (from a detection or explicit flags)
and consumed by ``lodedb migrate run``. It is the review surface: it states the
detected framework / provider, the replay mode and why, the source fingerprint, the
target path, the document-count estimate, the embedding dimension when relevant, the
required install extra, the validation thresholds, the rollback instructions, and
the exact code/config switch snippet. It carries no raw documents, vectors, queries,
payloads, or credentials.

The plan serializes to both JSON (``--plan ...json``, the machine input to ``run``)
and Markdown (``--out ...md``, the human review artifact). The two are written
side by side so ``run`` always has a JSON next to the Markdown a reviewer reads.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from lodedb.local.migrate.report import assert_payload_free
from lodedb.local.migrate.sources.base import MODE_TEXT_REPLAY, MODE_VECTOR_PRESERVE

PLAN_VERSION = 1

# Default validation thresholds. ``query_overlap`` is the fraction of a query's
# top-k that must survive the migration for representative queries (text-replay
# re-embeds, so exact rank parity is explicitly not promised — see the issue
# non-goals). ``sample_size`` bounds the id/metadata/text parity check.
DEFAULT_THRESHOLDS = {
    "count_parity": True,
    "query_overlap": 0.6,
    "query_sample": 5,
    "sample_size": 25,
}


@dataclass
class MigrationPlan:
    """A reviewable, payload-free migration plan."""

    route: str  # "framework" | "provider"
    mode: str  # text-replay | vector-preserve
    framework: str | None = None
    provider: str | None = None
    source_location_fingerprint: str = ""
    source_kind: str = ""  # in-memory | simple | qdrant | pgvector
    target_path: str = ""
    collection: str | None = None
    table: str | None = None
    document_count_estimate: int | None = None
    embedding_dim: int | None = None
    model: str | None = None
    device: str = "auto"
    package_manager: str = "pip"
    install_command: str | None = None
    install_extra: str | None = None
    store_text: bool = True
    unsupported_fields: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    thresholds: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    switch_snippet: str = ""
    rollback: list[str] = field(default_factory=list)
    source_options: dict[str, Any] = field(default_factory=dict)
    plan_version: int = PLAN_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Renders the plan as a payload-free JSON-serializable dict."""

        data = asdict(self)
        assert_payload_free(data, where="migration plan")
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MigrationPlan:
        """Rebuilds a plan from its JSON dict (consumed by ``run``)."""

        known = {f for f in cls.__dataclass_fields__}  # noqa: C416 - explicit set comprehension
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_markdown(self) -> str:
        """Renders the human-review Markdown plan (payload-free)."""

        target = self.framework or self.provider or "unknown"
        lines = [
            "# LodeDB migration plan",
            "",
            f"- Route: `{self.route}`"
            + (f" (framework: `{self.framework}`)" if self.framework else "")
            + (f" (provider: `{self.provider}`)" if self.provider else ""),
            f"- Mode: `{self.mode}` ({_mode_reason(self.mode)})",
            f"- Source: `{self.source_kind}` (fingerprint `{self.source_location_fingerprint}`)",
            f"- Target path: `{self.target_path}`",
        ]
        if self.collection:
            lines.append(f"- Collection: `{self.collection}`")
        if self.table:
            lines.append(f"- Table: `{self.table}`")
        if self.document_count_estimate is not None:
            lines.append(f"- Document count estimate: {self.document_count_estimate}")
        if self.embedding_dim is not None:
            lines.append(f"- Embedding dimension: {self.embedding_dim}")
        if self.model:
            lines.append(f"- LodeDB embedding model: `{self.model}` (device `{self.device}`)")
        lines += [
            f"- Package manager: `{self.package_manager}`",
            f"- Install: `{self.install_command}`",
            "",
            "## Risks and unsupported fields",
            "",
        ]
        if self.unsupported_fields:
            lines += [f"- Unsupported: {field}" for field in self.unsupported_fields]
        else:
            lines.append("- None recorded.")
        if self.warnings:
            lines += [""] + [f"- Warning: {w}" for w in self.warnings]
        lines += [
            "",
            "## Validation thresholds",
            "",
            f"- Count parity required: {self.thresholds.get('count_parity')}",
            f"- Query overlap threshold: {self.thresholds.get('query_overlap')}",
            f"- Sample size: {self.thresholds.get('sample_size')}",
            "",
            "## Switch the application",
            "",
            f"Apply this change to read from LodeDB ({target}):",
            "",
            "```python",
            self.switch_snippet.strip(),
            "```",
            "",
            "## Rollback",
            "",
        ]
        rollback = self.rollback or [
            "Keep the source store unchanged; it remains the rollback path.",
            "Revert the application config/code change to read from the source provider again.",
        ]
        lines += [f"- {step}" for step in rollback]
        lines += [
            "",
            "## Run",
            "",
            "```bash",
            f"lodedb migrate run --plan <plan>.json --target {self.target_path} --dry-run",
            f"lodedb migrate run --plan <plan>.json --target {self.target_path} --write",
            f"lodedb migrate validate --manifest {self.target_path}/migration.json",
            "```",
            "",
        ]
        return "\n".join(lines)


def _mode_reason(mode: str) -> str:
    """Returns the one-line reason a mode was chosen, for the plan body."""

    if mode == MODE_TEXT_REPLAY:
        return "LodeDB re-embeds the source text with its own preset"
    if mode == MODE_VECTOR_PRESERVE:
        return "the source owns embeddings; LodeDB stores the vectors verbatim"
    return mode


def default_rollback(framework: str | None, provider: str | None) -> list[str]:
    """Returns rollback steps appropriate to the route."""

    base = [
        "Do not delete the source store; it stays intact as the rollback path.",
    ]
    if framework == "mem0":
        base.append(
            "Revert the mem0 vector_store config to the previous provider and stop calling "
            "register_mem0_provider()."
        )
    elif framework in ("langchain", "llama-index"):
        base.append(
            "Revert the application to construct the previous vector store instead of the LodeDB "
            "adapter."
        )
    else:
        base.append(
            f"Revert the application to read from {provider or 'the source provider'} instead of "
            "LodeDB."
        )
    base.append("Re-run the application's retrieval tests plus one restart/reopen test.")
    return base


def build_switch_snippet(
    *,
    framework: str | None,
    provider: str | None,
    mode: str,
    target_path: str,
    model: str | None,
    embedding_dim: int | None,
    collection: str | None,
) -> str:
    """Builds the exact code/config switch snippet for the route.

    The snippet is the narrowest change that points retrieval at LodeDB: the
    shipped adapter for a framework route, or a direct ``LodeDB`` SDK handle for a
    direct provider (vector-preserve via ``open_vector_store`` when the app owns
    embeddings, text-owned via ``LodeDB(model=...)`` otherwise).
    """

    if framework == "langchain":
        return (
            "from lodedb import LodeDB\n"
            "from lodedb.local.integrations.langchain import LodeDBVectorStore\n\n"
            f"db = LodeDB({target_path!r}, model={model or 'minilm'!r})\n"
            "store = LodeDBVectorStore(db)\n"
            "# use `store` as your retriever's vector store\n"
            "retriever = store.as_retriever(search_kwargs={'k': 10})"
        )
    if framework == "llama-index":
        return (
            "from llama_index.core import StorageContext, VectorStoreIndex\n"
            "from llama_index.core.embeddings import MockEmbedding\n"
            "from lodedb.local.integrations.llama_index import LodeDBVectorStore\n\n"
            "vector_store = LodeDBVectorStore.from_path(\n"
            f"    {target_path!r}, model={model or 'minilm'!r}\n"
            ")\n"
            "storage_context = StorageContext.from_defaults(vector_store=vector_store)\n"
            "# LodeDB embeds text internally, so give VectorStoreIndex a cheap MockEmbedding:\n"
            "index = VectorStoreIndex.from_vector_store(\n"
            "    vector_store, embed_model=MockEmbedding(embed_dim=1)\n"
            ")"
        )
    if framework == "mem0":
        coll = collection or "memories"
        dim = embedding_dim or 1536
        return (
            "from mem0 import Memory\n"
            "from lodedb.local.integrations.mem0 import register_mem0_provider\n\n"
            "register_mem0_provider()  # call before Memory.from_config(...)\n"
            "config = {\n"
            '    "vector_store": {\n'
            '        "provider": "lodedb",\n'
            '        "config": {\n'
            f'            "path": {target_path!r},\n'
            f'            "collection_name": {coll!r},\n'
            f'            "embedding_model_dims": {dim},\n'
            "        },\n"
            "    },\n"
            "    # keep your existing embedder/llm config\n"
            "}\n"
            "memory = Memory.from_config(config)"
        )
    # Direct provider.
    if mode == MODE_VECTOR_PRESERVE:
        dim = embedding_dim or 1536
        return (
            "from lodedb import LodeDB\n\n"
            f"db = LodeDB.open_vector_store({target_path!r}, vector_dim={dim})\n"
            "hits = db.search_by_vector(\n"
            "    query_embedding, k=10, filter={'metadata': {'tenant_id': tenant_id}}\n"
            ")"
        )
    return (
        "from lodedb import LodeDB\n\n"
        f"db = LodeDB({target_path!r}, model={model or 'bge'!r})\n"
        "hits = db.search(\n"
        "    query_text, k=10, filter={'metadata': {'tenant_id': tenant_id}}\n"
        ")"
    )
