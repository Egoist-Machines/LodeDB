"""Tests for the ``lodedb migrate`` toolkit (issues #34 and #35).

Covers detection/routing (framework wins over a direct provider beneath it,
ambiguity stops), the four source exporters (LangChain in-memory, LlamaIndex
SimpleVectorStore, mem0 Qdrant via a fake client, direct pgvector via a fake
DB-API connection), the inspect -> plan -> dry-run -> run -> validate spine, and the
safety/validation invariants: payload-free reports, target collisions, dimension
validation, missing text, and resume/overwrite.

Real frameworks are exercised where they are installed (gated with
``importorskip``); the network providers (Qdrant, Postgres) are driven through
injected in-process fakes so the suite needs no server.
"""

from __future__ import annotations

import json

import pytest

from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.local.db import LodeDB
from lodedb.local.migrate import (
    MigrationError,
    build_plan,
    inspect_project,
    run_migration,
    target_has_store,
)
from lodedb.local.migrate.detect import Detection
from lodedb.local.migrate.plan import MigrationPlan, build_switch_snippet
from lodedb.local.migrate.report import (
    assert_payload_free,
    connection_host,
    is_local_source,
    redact_connection_string,
)
from lodedb.local.migrate.sources.base import (
    MODE_TEXT_REPLAY,
    MODE_VECTOR_PRESERVE,
    ExportedRow,
    SourceExport,
)

DIM = 8


def _backend() -> HashEmbeddingBackend:
    """A deterministic embedding backend so text-replay needs no model download."""

    return HashEmbeddingBackend(native_dim=384)


def _onehot(i: int) -> list[float]:
    vector = [0.0] * DIM
    vector[i] = 1.0
    return vector


class _FixtureExport(SourceExport):
    """An in-memory source export for driving the runner without a real source."""

    def __init__(self, rows: list[ExportedRow], **kwargs) -> None:
        self._rows = rows
        kwargs.setdefault("count", len(rows))
        super().__init__(**kwargs)

    def iter_rows(self):
        yield from self._rows


# --------------------------------------------------------------------------------------
# Report / redaction helpers.
# --------------------------------------------------------------------------------------


def test_redact_connection_string_drops_credentials_and_host():
    """A DSN is reduced to its scheme; a bare path/name is left unchanged."""

    assert redact_connection_string("postgresql://user:pw@db.example.com:5432/app") == (
        "postgresql://<redacted>"
    )
    assert redact_connection_string("./data/store") == "./data/store"
    assert connection_host("postgresql://u:pw@db.example.com:5432/app") == "db.example.com"
    assert connection_host("./data/store") is None


def test_is_local_source_gates_remote_hosts():
    """Loopback hosts and bare paths are local; a remote host is not."""

    assert is_local_source("postgresql://localhost:5432/app") is True
    assert is_local_source("postgresql://127.0.0.1/app") is True
    assert is_local_source("./data") is True
    assert is_local_source("postgresql://prod.db.example.com/app") is False


def test_assert_payload_free_flags_banned_keys():
    """The payload-free guard catches a leaked raw-text/credential key."""

    assert_payload_free({"counts": 3, "ids": ["a"]})  # fine
    with pytest.raises(ValueError):
        assert_payload_free({"target": {"text": "secret doc"}})
    with pytest.raises(ValueError):
        assert_payload_free({"password": "x"})


# --------------------------------------------------------------------------------------
# Detection and routing.
# --------------------------------------------------------------------------------------


def test_inspect_detects_mem0_framework_and_install_extra(tmp_path):
    """A mem0 project routes to the framework path with the right install extra."""

    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "app"\ndependencies = ["mem0ai>=2.0.0"]\n', encoding="utf-8"
    )
    (tmp_path / "app.py").write_text(
        "from mem0 import Memory\nm = Memory.from_config({})\n", encoding="utf-8"
    )
    det = inspect_project(tmp_path)
    assert det.route == "framework"
    assert det.framework == "mem0"
    assert det.install_extra == "lodedb[mem0]"
    assert "lodedb[mem0]" in det.install_command


def test_inspect_detects_langchain_install_extra(tmp_path):
    """A LangChain project prints the langchain extra."""

    (tmp_path / "rag.py").write_text(
        "from langchain_core.vectorstores import InMemoryVectorStore\n", encoding="utf-8"
    )
    det = inspect_project(tmp_path)
    assert det.route == "framework"
    assert det.framework == "langchain"
    assert det.install_extra == "lodedb[langchain]"


def test_inspect_detects_direct_pgvector_when_no_framework(tmp_path):
    """A direct pgvector project (no framework) routes to the provider path."""

    (tmp_path / "requirements.txt").write_text("psycopg[binary]\npgvector\n", encoding="utf-8")
    (tmp_path / "db.py").write_text(
        "import psycopg\n# CREATE EXTENSION vector;\n# embedding vector(1536)\n", encoding="utf-8"
    )
    det = inspect_project(tmp_path)
    assert det.route == "provider"
    assert det.provider == "pgvector"
    assert det.install_command.endswith("lodedb")


def test_framework_detection_wins_over_pgvector_beneath_it(tmp_path):
    """pgvector used *through* LangChain routes to the framework path, not direct pgvector."""

    (tmp_path / "app.py").write_text(
        "from langchain_postgres import PGVector\nimport psycopg\n", encoding="utf-8"
    )
    det = inspect_project(tmp_path)
    assert det.route == "framework"
    assert det.framework == "langchain"
    assert "pgvector" in det.providers_seen  # detected, but not the chosen route


def test_inspect_ambiguous_two_frameworks_stops(tmp_path):
    """Two frameworks with no clear owner is ambiguous and asks for --framework."""

    (tmp_path / "a.py").write_text("import langchain_core\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("import llama_index\n", encoding="utf-8")
    det = inspect_project(tmp_path)
    assert det.route == "ambiguous"
    assert set(det.frameworks_seen) == {"langchain", "llama-index"}
    assert det.next_step and "--framework" in det.next_step


def test_inspect_framework_override_pins_route(tmp_path):
    """An explicit --framework pins the route even on an empty project."""

    det = inspect_project(tmp_path, framework="llama-index")
    assert det.route == "framework"
    assert det.framework == "llama-index"


def test_inspect_unknown_framework_raises(tmp_path):
    """An unknown --framework value is rejected."""

    with pytest.raises(ValueError):
        inspect_project(tmp_path, framework="pinecone")


# --------------------------------------------------------------------------------------
# Plan: snippets, payload-free, round-trip.
# --------------------------------------------------------------------------------------


def test_plan_is_payload_free_and_round_trips():
    """A plan serializes payload-free and reloads from its own dict."""

    det = Detection(
        route="provider", provider="pgvector", install_command="pip install lodedb"
    )
    plan = build_plan(
        det,
        target="./data/lodedb",
        embedding_dim=1536,
        table="documents",
        source="postgresql://u:pw@localhost/app",
    )
    data = plan.to_dict()
    assert_payload_free(data, where="plan")  # raises if a banned key leaked
    again = MigrationPlan.from_dict(data)
    assert again.mode == MODE_VECTOR_PRESERVE
    assert again.embedding_dim == 1536
    # The Markdown plan must not carry the raw DSN.
    md = plan.to_markdown()
    assert "u:pw@localhost" not in md
    assert plan.source_location_fingerprint and plan.source_location_fingerprint in md


def test_switch_snippets_cover_both_sdk_shapes():
    """Direct-provider snippets cover vector-preserve and text-owned LodeDB usage."""

    vec = build_switch_snippet(
        framework=None,
        provider="pgvector",
        mode=MODE_VECTOR_PRESERVE,
        target_path="./data/lodedb",
        model=None,
        embedding_dim=1536,
        collection=None,
    )
    assert "open_vector_store" in vec
    assert "search_by_vector" in vec

    text = build_switch_snippet(
        framework=None,
        provider="pgvector",
        mode=MODE_TEXT_REPLAY,
        target_path="./data/lodedb",
        model="bge",
        embedding_dim=None,
        collection=None,
    )
    assert 'LodeDB(' in text
    assert "db.search(" in text

    mem0 = build_switch_snippet(
        framework="mem0",
        provider=None,
        mode=MODE_VECTOR_PRESERVE,
        target_path="./data/mem0-lodedb",
        model=None,
        embedding_dim=1536,
        collection="memories",
    )
    assert "register_mem0_provider()" in mem0
    assert '"provider": "lodedb"' in mem0


# --------------------------------------------------------------------------------------
# Runner: text-replay (LangChain/LlamaIndex), vector-preserve (mem0/pgvector), via fakes.
# --------------------------------------------------------------------------------------


def test_run_text_replay_into_fresh_target_validates_and_is_payload_free(tmp_path):
    """A text-replay run writes a fresh store, validates, and emits a payload-free manifest."""

    rows = [
        ExportedRow(id="a", text="alpha document", metadata={"k": "a"}),
        ExportedRow(id="b", text="beta document", metadata={"k": "b"}),
        ExportedRow(id="c", text="   ", metadata={"k": "c"}),  # blank -> skipped
    ]
    export = _FixtureExport(
        rows,
        framework="langchain",
        provider="in-memory",
        mode=MODE_TEXT_REPLAY,
        location="dump.json",
    )
    det = Detection(route="framework", framework="langchain")
    plan = build_plan(det, target=tmp_path / "lc", model="minilm")
    result = run_migration(plan, dry_run=False, source=export, embedding_backend=_backend())

    assert result.status == "migrated"
    assert result.written_count == 2
    assert [s["reason"] for s in result.skipped] == ["missing-text"]
    assert result.validation["count_parity"] is True
    assert result.validation["passed"] is True
    assert result.validation["audit"]["status"] == "passed"
    assert result.validation["audit"]["raw_document_text_present"] is False

    # The store reopens and the text round-trips.
    db = LodeDB(
        path=tmp_path / "lc", model="minilm", read_only=True, _embedding_backend=_backend()
    )
    try:
        assert db.count() == 2
        assert db.get("a") == "alpha document"
    finally:
        db.close()

    # The manifest is payload-free and contains no document text.
    manifest = json.loads((tmp_path / "lc" / "migration.json").read_text(encoding="utf-8"))
    assert_payload_free(manifest, where="migration.json")
    assert "alpha document" not in json.dumps(manifest)
    assert manifest["skipped"][0]["id_hash"] != "c"  # ids are hashed, not raw


def test_run_dry_run_writes_nothing(tmp_path):
    """A dry run probes the source and never creates the target."""

    export = _FixtureExport(
        [ExportedRow(id="a", text="x")],
        framework="langchain",
        provider="in-memory",
        mode=MODE_TEXT_REPLAY,
        location="dump.json",
    )
    det = Detection(route="framework", framework="langchain")
    plan = build_plan(det, target=tmp_path / "lc")
    result = run_migration(plan, dry_run=True, source=export)
    assert result.status == "dry-run"
    assert not (tmp_path / "lc").exists()
    assert not (tmp_path / "lc.tmp").exists()


def test_run_refuses_existing_target_then_overwrites(tmp_path):
    """A run refuses a non-empty target unless --overwrite-target is set."""

    det = Detection(route="framework", framework="langchain")
    plan = build_plan(det, target=tmp_path / "lc", model="minilm")

    def fresh_export():
        return _FixtureExport(
            [ExportedRow(id="a", text="alpha")],
            framework="langchain",
            provider="in-memory",
            mode=MODE_TEXT_REPLAY,
            location="dump.json",
        )

    run_migration(plan, dry_run=False, source=fresh_export(), embedding_backend=_backend())
    assert target_has_store(tmp_path / "lc")

    with pytest.raises(MigrationError):
        run_migration(plan, dry_run=False, source=fresh_export(), embedding_backend=_backend())

    # Overwrite succeeds and leaves a valid store.
    result = run_migration(
        plan,
        dry_run=False,
        overwrite_target=True,
        source=fresh_export(),
        embedding_backend=_backend(),
    )
    assert result.status == "migrated"
    assert result.written_count == 1


def test_run_vector_preserve_validates_dimension_multiple_of_8(tmp_path):
    """Vector-preserve refuses a dimension that is not a positive multiple of 8."""

    export = _FixtureExport(
        [ExportedRow(id="m1", vector=[0.0] * 10)],
        framework=None,
        provider="pgvector",
        mode=MODE_VECTOR_PRESERVE,
        location="postgresql://localhost/app",
        vector_dim=10,
    )
    det = Detection(route="provider", provider="pgvector")
    plan = build_plan(det, target=tmp_path / "pg", embedding_dim=10)
    with pytest.raises(MigrationError):
        run_migration(plan, dry_run=True, source=export)


def test_run_vector_preserve_skips_dimension_mismatch_rows(tmp_path):
    """A row whose vector width disagrees with the index dim is skipped, not fatal."""

    rows = [
        ExportedRow(id="m1", vector=_onehot(0), metadata={"tenant_id": "t1"}),
        ExportedRow(id="bad", vector=[0.0] * (DIM + 1)),  # wrong width -> skipped
    ]
    export = _FixtureExport(
        rows,
        framework=None,
        provider="pgvector",
        mode=MODE_VECTOR_PRESERVE,
        location="postgresql://localhost/app",
        vector_dim=DIM,
    )
    det = Detection(route="provider", provider="pgvector")
    plan = build_plan(det, target=tmp_path / "pg", embedding_dim=DIM)
    result = run_migration(plan, dry_run=False, source=export)
    assert result.written_count == 1
    assert [s["reason"] for s in result.skipped] == ["dimension-mismatch"]

    db = LodeDB.open_vector_store(tmp_path / "pg", vector_dim=DIM, read_only=True)
    try:
        hits = db.search_by_vector(_onehot(0), k=2)
        assert hits[0].id == "m1"
        assert hits[0].metadata == {"tenant_id": "t1"}
    finally:
        db.close()


def test_run_mem0_vector_preserve_preserves_payload_and_scalar_metadata(tmp_path):
    """A mem0 run keeps the full payload in the sidecar and scalar keys in metadata."""

    pytest.importorskip("mem0")
    rows = [
        ExportedRow(
            id="m1",
            vector=_onehot(0),
            raw_payload={"data": "Alice", "user_id": "u1", "linked_memory_ids": ["root"]},
        ),
        ExportedRow(id="m2", vector=_onehot(1), raw_payload={"data": "Bob", "user_id": "u2"}),
    ]
    export = _FixtureExport(
        rows,
        framework="mem0",
        provider="qdrant",
        mode=MODE_VECTOR_PRESERVE,
        location="qdrant-path",
        vector_dim=DIM,
    )
    det = Detection(route="framework", framework="mem0")
    plan = build_plan(det, target=tmp_path / "mem0", embedding_dim=DIM, collection="memories")
    result = run_migration(plan, dry_run=False, source=export)
    assert result.status == "migrated"
    assert result.written_count == 2
    assert result.store_subdir == "memories"
    assert result.validation["passed"] is True

    from lodedb.local.integrations.mem0 import LodeDBVectorStore

    store = LodeDBVectorStore(
        path=str(tmp_path / "mem0"), collection_name="memories", embedding_model_dims=DIM
    )
    try:
        got = store.get("m1")
        assert got.payload["data"] == "Alice"
        assert got.payload["linked_memory_ids"] == ["root"]
        assert store.client.get_document("m1")["metadata"] == {"user_id": "u1"}
    finally:
        store.close()


# --------------------------------------------------------------------------------------
# Real framework adapters (gated on the optional deps).
# --------------------------------------------------------------------------------------


def test_langchain_inmemory_export_roundtrip(tmp_path):
    """A real LangChain InMemoryVectorStore dump exports and migrates via the adapter."""

    pytest.importorskip("langchain_core")
    from langchain_core.documents import Document
    from langchain_core.vectorstores import InMemoryVectorStore

    from lodedb.local.migrate.sources.langchain_inmemory import LangChainInMemoryExport

    class _Emb:
        def embed_documents(self, texts):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

        def embed_query(self, text):
            return [0.1, 0.2, 0.3, 0.4]

    vs = InMemoryVectorStore(_Emb())
    vs.add_documents(
        [
            Document(id="x", page_content="hello world", metadata={"src": "a"}),
            Document(id="y", page_content="goodbye world", metadata={"src": "b"}),
        ]
    )
    dump = tmp_path / "dump.json"
    vs.dump(str(dump))

    export = LangChainInMemoryExport(str(dump))
    assert export.framework == "langchain"
    assert export.count == 2
    rows = {r.id: r for r in export.iter_rows()}
    assert rows["x"].text == "hello world"
    assert rows["x"].metadata == {"src": "a"}

    det = Detection(route="framework", framework="langchain")
    plan = build_plan(det, target=tmp_path / "lc", model="minilm", source=str(dump))
    result = run_migration(
        plan, dry_run=False, source=LangChainInMemoryExport(str(dump)), embedding_backend=_backend()
    )
    assert result.written_count == 2
    assert result.validation["passed"] is True


def test_llama_index_simple_export_preserves_source_relationship(tmp_path):
    """A persisted LlamaIndex docstore exports nodes with text/metadata/ref_doc_id."""

    pytest.importorskip("llama_index.core")
    from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
    from llama_index.core.storage.docstore import SimpleDocumentStore

    from lodedb.local.integrations.llama_index import LodeDBVectorStore
    from lodedb.local.migrate.sources.llama_index_simple import LlamaIndexSimpleExport

    n1 = TextNode(id_="n1", text="the quick brown fox", metadata={"topic": "animals"})
    n1.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id="docA")
    n2 = TextNode(id_="n2", text="quantum field theory", metadata={"topic": "physics"})
    src = tmp_path / "li_src"
    docstore = SimpleDocumentStore()
    docstore.add_documents([n1, n2])
    docstore.persist(str(src / "docstore.json"))

    export = LlamaIndexSimpleExport(str(src))
    rows = {r.id: r for r in export.iter_rows()}
    assert rows["n1"].text == "the quick brown fox"
    assert rows["n1"].ref_doc_id == "docA"

    det = Detection(route="framework", framework="llama-index")
    plan = build_plan(det, target=tmp_path / "li", model="minilm", source=str(src))
    result = run_migration(
        plan, dry_run=False, source=LlamaIndexSimpleExport(str(src)), embedding_backend=_backend()
    )
    assert result.written_count == 2
    assert result.validation["passed"] is True

    db = LodeDB(
        path=tmp_path / "li", model="minilm", read_only=True, _embedding_backend=_backend()
    )
    try:
        store = LodeDBVectorStore(db)
        node = store.get_nodes(node_ids=["n1"])[0]
        assert node.get_content() == "the quick brown fox"
        assert node.ref_doc_id == "docA"
        assert node.metadata["topic"] == "animals"
    finally:
        db.close()


def test_missing_text_in_text_replay_is_skipped_with_reason(tmp_path):
    """A node with no text exports as a skipped row, not a hard failure."""

    export = _FixtureExport(
        [
            ExportedRow(id="ok", text="content here"),
            ExportedRow(id="empty", text=None),
        ],
        framework="llama-index",
        provider="simple",
        mode=MODE_TEXT_REPLAY,
        location="li_src",
    )
    det = Detection(route="framework", framework="llama-index")
    plan = build_plan(det, target=tmp_path / "li", model="minilm")
    result = run_migration(plan, dry_run=False, source=export, embedding_backend=_backend())
    assert result.written_count == 1
    assert any(s["reason"] == "missing-text" for s in result.skipped)


# --------------------------------------------------------------------------------------
# Publish safety, dimension discovery, and batched writes.
# --------------------------------------------------------------------------------------


def test_failed_validation_does_not_publish_target(tmp_path):
    """A run whose validation fails must raise and leave no target on disk.

    Two source rows share an id, so the target holds one document while the run wrote
    two: count parity fails. The run must not publish a failed migration.
    """

    export = _FixtureExport(
        [ExportedRow(id="dup", text="first"), ExportedRow(id="dup", text="second")],
        framework="langchain",
        provider="in-memory",
        mode=MODE_TEXT_REPLAY,
        location="dump.json",
    )
    det = Detection(route="framework", framework="langchain")
    plan = build_plan(det, target=tmp_path / "lc", model="minilm")
    with pytest.raises(MigrationError):
        run_migration(plan, dry_run=False, source=export, embedding_backend=_backend())
    assert not (tmp_path / "lc").exists()


def test_failed_overwrite_leaves_existing_target_intact(tmp_path):
    """A failed `--overwrite-target` run must not replace a previously valid target."""

    det = Detection(route="framework", framework="langchain")
    plan = build_plan(det, target=tmp_path / "lc", model="minilm")

    good = _FixtureExport(
        [ExportedRow(id="a", text="alpha")],
        framework="langchain",
        provider="in-memory",
        mode=MODE_TEXT_REPLAY,
        location="dump.json",
    )
    run_migration(plan, dry_run=False, source=good, embedding_backend=_backend())
    assert target_has_store(tmp_path / "lc")

    bad = _FixtureExport(
        [ExportedRow(id="x", text="one"), ExportedRow(id="x", text="two")],
        framework="langchain",
        provider="in-memory",
        mode=MODE_TEXT_REPLAY,
        location="dump.json",
    )
    with pytest.raises(MigrationError):
        run_migration(
            plan, dry_run=False, overwrite_target=True, source=bad, embedding_backend=_backend()
        )

    db = LodeDB(path=tmp_path / "lc", model="minilm", read_only=True, _embedding_backend=_backend())
    try:
        assert db.count() == 1
        assert db.get("a") == "alpha"
    finally:
        db.close()


def test_vector_preserve_validates_with_discovered_dimension(tmp_path):
    """A provider plan with no --embedding-dim validates against the discovered dimension.

    The source discovers a 16-dim vector; the target is written 16-dim and validation
    must reopen it 16-dim (not the 8 fallback), so the run succeeds and the manifest
    records the discovered dimension.
    """

    rows = [
        ExportedRow(
            id=str(i),
            vector=[1.0 if j == i % 16 else 0.0 for j in range(16)],
            metadata={"t": "a"},
        )
        for i in range(3)
    ]
    export = _FixtureExport(
        rows,
        framework=None,
        provider="pgvector",
        mode=MODE_VECTOR_PRESERVE,
        location="postgresql://localhost/app",
        vector_dim=16,
    )
    det = Detection(route="provider", provider="pgvector")
    plan = build_plan(det, target=tmp_path / "pg", embedding_dim=None, table="docs")
    assert plan.embedding_dim is None

    result = run_migration(plan, dry_run=False, source=export)
    assert result.status == "migrated"
    assert result.embedding_dim == 16
    assert result.validation["passed"] is True

    manifest = json.loads((tmp_path / "pg" / "migration.json").read_text(encoding="utf-8"))
    assert manifest["target"]["embedding_dim"] == 16


def test_write_path_batches_rows_not_one_call_per_row(tmp_path, monkeypatch):
    """The writer flushes rows through the batch API a bounded number of times.

    1,000 rows must produce ceil(1000 / _WRITE_BATCH) batch calls, not 1,000, so large
    migrations keep batched commits and batched embedding.
    """

    from lodedb.local.migrate import runner as runner_mod

    n = 1000
    rows = [ExportedRow(id=str(i), text=f"doc {i}", metadata={"k": "v"}) for i in range(n)]
    export = _FixtureExport(
        rows, framework=None, provider="pgvector", mode=MODE_TEXT_REPLAY, location="src"
    )
    det = Detection(route="provider", provider="pgvector")
    plan = build_plan(det, target=tmp_path / "t", model="minilm", mode=MODE_TEXT_REPLAY)

    calls = {"batches": 0, "rows": 0}
    original = runner_mod._PlainTextTargetWriter.write_batch

    def counting(self, batch):
        calls["batches"] += 1
        calls["rows"] += len(batch)
        return original(self, batch)

    monkeypatch.setattr(runner_mod._PlainTextTargetWriter, "write_batch", counting)
    result = run_migration(plan, dry_run=False, source=export, embedding_backend=_backend())

    assert result.written_count == n
    assert calls["rows"] == n
    expected_batches = (n + runner_mod._WRITE_BATCH - 1) // runner_mod._WRITE_BATCH
    assert calls["batches"] == expected_batches
    assert calls["batches"] < n
