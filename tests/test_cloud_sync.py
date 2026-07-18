"""Sync tests through the real engine: the `lodedb cloud sync` CLI verb and the
`lodedb.cloud.sync`/`lodedb.cloud.status` lineage surface, driven by actual LodeDB
commits (so epochs, generation numbers, and the epoch GC behave exactly as a
user's database does)."""

import json

import pytest
from typer.testing import CliRunner

from lodedb import cloud
from lodedb.cloud.cli import app
from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.local.db import LodeDB

runner = CliRunner()


def add_documents(path, documents) -> None:
    """One further committed generation in `path` through the real engine."""
    db = LodeDB(
        path=path,
        model="minilm",
        commit_mode="generation",
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )
    try:
        db.add_many(documents)
    finally:
        db.close()


def docs(tag: str, count: int = 2):
    return [{"text": f"{tag} document {i}", "id": f"{tag}-{i}"} for i in range(count)]


def test_sync_lifecycle_two_clones(committed_store, tmp_path):
    source, key = committed_store
    remote = tmp_path / "remote"
    clone = tmp_path / "clone"
    clone.mkdir()

    # Fresh push.
    result = runner.invoke(app, ["sync", str(source), str(remote), key])
    assert result.exit_code == 0
    outcome = json.loads(result.output)
    assert outcome["classification"] == "local_ahead"
    assert outcome["action"] == "push"
    assert (source / f"{key}.orecloud").exists()

    # Converged: a second sync moves nothing.
    result = runner.invoke(app, ["sync", str(source), str(remote), key])
    assert result.exit_code == 0
    assert "action" in result.output and "none" in result.output

    # A new commit fast-forwards.
    add_documents(source, docs("second"))
    report = cloud.status(str(source), str(remote), key)
    assert report["sidecar_present"]
    assert report["classification"] == "local_ahead"
    result = runner.invoke(app, ["sync", str(source), str(remote), key])
    assert result.exit_code == 0
    assert "push" in result.output

    # A second clone pulls (and the pull verify-opens: counts are reported).
    result = runner.invoke(app, ["sync", str(clone), str(remote), key])
    assert result.exit_code == 0
    outcome = json.loads(result.output)
    assert outcome["classification"] == "remote_ahead"
    assert "document_count" in outcome


def test_diverged_requires_force_and_force_resolves(committed_store, tmp_path):
    source, key = committed_store
    remote = tmp_path / "remote"
    clone = tmp_path / "clone"
    clone.mkdir()

    runner.invoke(app, ["sync", str(source), str(remote), key])
    runner.invoke(app, ["sync", str(clone), str(remote), key])

    # The clone advances once and publishes; the source advances twice on its
    # own lineage. (Different commit counts keep the two lineages on different
    # base epochs — same-epoch artifact-name collisions across divergent
    # lineages are the fork-collision case the Phase-2 content-addressed
    # layout absorbs; a dumb remote fails them closed at the
    # immutable-artifact check.)
    add_documents(clone, docs("clone-side"))
    result = runner.invoke(app, ["sync", str(clone), str(remote), key])
    assert result.exit_code == 0
    add_documents(source, docs("source-side-a"))
    add_documents(source, docs("source-side-b"))

    # Diverged: sync refuses, names the classification and both force flags.
    result = runner.invoke(app, ["sync", str(source), str(remote), key])
    assert result.exit_code == 1
    assert "diverged" in result.output
    assert "--force-push" in result.output and "--force-pull" in result.output

    # The API surface raises the same refusal.
    with pytest.raises(RuntimeError, match="diverged"):
        cloud.sync(str(source), str(remote), key)

    # Both force flags at once is a usage error, not a transfer.
    result = runner.invoke(
        app, ["sync", str(source), str(remote), key, "--force-push", "--force-pull"]
    )
    assert result.exit_code == 2

    # Force-push keeps the source's lineage; the clone then fast-forwards.
    result = runner.invoke(app, ["sync", str(source), str(remote), key, "--force-push"])
    assert result.exit_code == 0
    assert json.loads(result.output)["forced"] is True
    result = runner.invoke(app, ["sync", str(clone), str(remote), key])
    assert result.exit_code == 0
    assert json.loads(result.output)["classification"] == "remote_ahead"


def test_sidecar_survives_engine_epoch_gc(committed_store, tmp_path):
    """The engine's local epoch GC must never eat the sync base: after enough
    commits to cycle the retained-epoch window, the sidecar still classifies
    the directory as trusted local_ahead (an eaten/corrupt sidecar would read
    as unknown)."""
    source, key = committed_store
    remote = tmp_path / "remote"
    runner.invoke(app, ["sync", str(source), str(remote), key])

    for round_ in range(4):
        add_documents(source, docs(f"round-{round_}"))

    assert (source / f"{key}.orecloud").exists()
    report = cloud.status(str(source), str(remote), key)
    assert report["sidecar_present"]
    assert report["classification"] == "local_ahead"

    # And the recorded base still fast-forwards.
    outcome = cloud.sync(str(source), str(remote), key)
    assert outcome["action"] == "push"
    assert outcome["classification"] == "local_ahead"


def test_sidecar_is_invisible_to_the_engine(committed_store, tmp_path):
    """The engine's load path treats unknown top-level `*.json` files as legacy
    index snapshots, so the sidecar must not look like one: after a sync, the
    directory must still open (and commit) as a plain LodeDB."""
    source, key = committed_store
    remote = tmp_path / "remote"
    result = runner.invoke(app, ["sync", str(source), str(remote), key])
    assert result.exit_code == 0
    assert (source / f"{key}.orecloud").exists()

    # Reopen through the engine and commit — this is exactly what broke when
    # the sidecar was named `<key>.orecloud.json`.
    add_documents(source, docs("after-sync"))
    (found_key,) = cloud.keys(str(source))
    assert found_key == key


def test_status_lineage_fields_without_a_sidecar(committed_store, tmp_path):
    """A never-synced directory reports its lineage as untrusted, not a guess."""
    source, key = committed_store
    remote = tmp_path / "remote"

    report = cloud.status(str(source), str(remote), key)
    assert not report["sidecar_present"]
    assert report["base_generation"] is None
    # Local exists, remote absent: direction is unambiguous even without a base.
    assert report["classification"] == "local_ahead"

    # Push via the plain verb also records the base (push and sync share it).
    cloud.push(str(source), str(remote), key)
    report = cloud.status(str(source), str(remote), key)
    assert report["sidecar_present"]
    assert report["classification"] == "in_sync"
