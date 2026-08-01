"""End-to-end tests of the Python API (`lodedb.cloud.*` over the native core):
push/status/verify/pull round trips against real engine-written generations,
the redacted-by-default privacy posture, and error mapping onto stdlib
exceptions.
"""

import pytest
from conftest import DOCUMENTS, read_pointer_body

from lodedb import cloud


def test_push_status_pull_round_trip(committed_store, tmp_path):
    """The full backup/restore cycle through local directories."""
    source, key = committed_store
    remote = tmp_path / "remote"
    restored = tmp_path / "restored"

    before = cloud.status(str(source), str(remote), key)
    assert not before["in_sync"]
    assert before["remote_generation"] is None

    pushed = cloud.push(str(source), str(remote), key)
    assert pushed["pointer_published"]
    assert pushed["artifacts_written"] > 0

    after = cloud.status(str(source), str(remote), key)
    assert after["in_sync"]
    assert after["artifacts_to_upload"] == 0

    report = cloud.verify(str(remote), key)
    assert report["artifacts_verified"] > 0

    # Pull restores AND proves the copy opens through the engine.
    outcome = cloud.pull(str(remote), str(restored), key)
    assert outcome["pointer_published"]
    assert outcome["document_count"] == len(DOCUMENTS)

    # The restored copy verifies clean too.
    cloud.verify(str(restored), key)


def test_repeated_push_is_idempotent(committed_store, tmp_path):
    source, key = committed_store
    remote = tmp_path / "remote"
    cloud.push(str(source), str(remote), key)
    again = cloud.push(str(source), str(remote), key)
    assert again["artifacts_written"] == 0
    assert again["bytes_written"] == 0
    assert not again["pointer_published"]


def test_push_is_redacted_by_default(committed_store, tmp_path):
    """Without the opt-in flags, the published remote body carries no text store."""
    source, key = committed_store
    remote = tmp_path / "remote"
    cloud.push(str(source), str(remote), key)

    body = read_pointer_body(remote, key)
    assert body["tvtext"] is None
    assert body["tvlex"] is None
    assert body["json"] is not None
    # The source itself still has its text store; redaction is per-transfer.
    assert read_pointer_body(source, key)["tvtext"] is not None


def test_opt_in_flags_ship_the_text_store(committed_store, tmp_path):
    source, key = committed_store
    remote = tmp_path / "remote"
    cloud.push(str(source), str(remote), key, include_text=True)
    assert read_pointer_body(remote, key)["tvtext"] is not None


def test_missing_generation_raises_file_not_found(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        cloud.push(str(empty), str(tmp_path / "remote"), "no-such-key")


def test_bad_target_scheme_raises_runtime_error(committed_store, tmp_path):
    source, key = committed_store
    with pytest.raises(RuntimeError, match="scheme"):
        cloud.push(str(source), "ftp://nope/x", key)


# --------------------------------------------------- LodeGraph sidecar bodies
#
# The native inventory behind pull/status/verify learned the two LodeGraph
# topology sidecar keys (gtopo: content-free episode anchors; gtopotext: a
# topology embedding user text). These mirror the Rust suite's cases through
# the Python binding: the recorded file name (topology.sqlite3) is validated
# as a safe single path component (never matched against a g<epoch>
# derivation; the layout addresses the artifact by content), unsafe names and
# both-keys bodies are refused, and a genuinely unknown key still fails
# closed.

HEX64_A = "a" * 64
HEX64_B = "b" * 64


def _graph_body(**stores) -> str:
    import json as _json

    body = {
        "index_key": "idx",
        "generation": 3,
        "base_epoch": 2,
        "document_count": 0,
        "chunk_count": 0,
        "json": {
            "base": {"file_name": "g2.json", "sha256": HEX64_A, "file_bytes": 0},
            "deltas": [],
        },
    }
    body.update(stores)
    return _json.dumps(body)


def _pull_requirements(tmp_path, body_json: str):
    from lodedb._turbovec import cloud as _core

    return _core.managed_pull_requirements(str(tmp_path), "idx", body_json)


def _topology_sub(file_name: str | None = "topology.sqlite3") -> dict:
    base = {"sha256": HEX64_B, "file_bytes": 12}
    if file_name is not None:
        base["file_name"] = file_name
    return {"base": base, "deltas": []}


@pytest.mark.parametrize("sidecar", ["gtopo", "gtopotext"])
@pytest.mark.parametrize("file_name", ["topology.sqlite3", None])
def test_graph_sidecar_bodies_walk_content_addressed(tmp_path, sidecar, file_name):
    # The native file name is accepted (and, like the server, optional); no
    # g<epoch> derivation is required (base_epoch=2 here). The layout name is
    # content-addressed, so successive topologies never collide with the
    # immutable-artifact rule.
    needed = _pull_requirements(tmp_path, _graph_body(**{sidecar: _topology_sub(file_name)}))
    names = {artifact["name"]: artifact["kind"] for artifact in needed}
    assert names[f"idx.gen/{HEX64_B}.{sidecar}"] == sidecar


def test_graph_sidecar_refuses_an_unsafe_file_name(tmp_path):
    for bad in ("", "../topology.sqlite3", "a/b.sqlite3", ".."):
        with pytest.raises(RuntimeError, match="file name"):
            _pull_requirements(tmp_path, _graph_body(gtopo=_topology_sub(bad)))


def test_managed_push_refuses_unredactable_graph_text():
    """A push that has not opted into text refuses a gtopotext generation
    outright (its user text lives inside the one SQLite artifact, so there is
    no redacted form to publish); the opt-in pushes it verbatim. Refusal
    happens before any control-plane call."""
    from lodedb.cloud.transfer import CloudError, ManagedRemote, _push_with_plan

    plan = {
        "local": {
            "snapshot_id": "s1",
            "logical_id": "l1",
            "generation": 1,
            "body_json": "{}",
            "pointer_document": "{}",
            "artifacts": [
                {"name": f"idx.gen/{HEX64_B}.gtopotext", "sha256": HEX64_B,
                 "size_bytes": 12, "kind": "gtopotext"},
            ],
        },
        "base": None,
    }
    remote = ManagedRemote("acme", "prod", "user-42")
    with pytest.raises(CloudError, match="include_text"):
        _push_with_plan(
            object(), "unused-dir", "idx", remote, "https://cloud.test",
            {"head": None}, plan, include_text=False,
        )


def test_graph_sidecars_are_mutually_exclusive(tmp_path):
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        _pull_requirements(
            tmp_path, _graph_body(gtopo=_topology_sub(), gtopotext=_topology_sub())
        )


def test_unknown_sidecar_keys_still_fail_closed(tmp_path):
    with pytest.raises(RuntimeError, match="gfuture"):
        _pull_requirements(tmp_path, _graph_body(gfuture=_topology_sub()))
