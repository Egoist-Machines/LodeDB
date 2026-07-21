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
