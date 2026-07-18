"""`lodedb.cloud._env_file`: dotenv writing that never clobbers, plus the
gitignore guard — the pieces that keep `tokens mint --env-file` and
`init --agents` from leaking or destroying anything."""

import stat
import sys

from lodedb.cloud import _env_file


def _assert_owner_only(path):
    """POSIX-only exact-mode check: Windows has no 0600 to assert (owner-only
    there is an ACL property, which mkstemp's descriptor-level creation
    already provides)."""
    if sys.platform != "win32":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_creates_missing_file_owner_only(tmp_path):
    target = tmp_path / ".env"
    _env_file.write_env_values(target, {"ORECLOUD_TOKEN": "ore_sk_abc"})
    assert target.read_text() == "ORECLOUD_TOKEN=ore_sk_abc\n"
    _assert_owner_only(target)


def test_replaces_managed_keys_and_preserves_everything_else(tmp_path):
    target = tmp_path / ".env"
    target.write_text("# app config\nDATABASE_URL=postgres://x\nORECLOUD_TOKEN=old\n\nDEBUG=1\n")
    _env_file.write_env_values(target, {"ORECLOUD_TOKEN": "new", "ORECLOUD_HOST": "https://c.example"})
    assert target.read_text() == (
        "# app config\nDATABASE_URL=postgres://x\nORECLOUD_TOKEN=new\n\nDEBUG=1\n"
        "ORECLOUD_HOST=https://c.example\n"
    )


def test_existing_permissive_file_ends_up_owner_only(tmp_path):
    """A pre-existing 0644 `.env` (the common default) must not keep the
    minted token group/world-readable: the write always lands 0600."""
    import os

    target = tmp_path / ".env"
    target.write_text("DEBUG=1\n")
    os.chmod(target, 0o644)
    _env_file.write_env_values(target, {"ORECLOUD_TOKEN": "ore_sk_abc"})
    _assert_owner_only(target)
    assert target.read_text() == "DEBUG=1\nORECLOUD_TOKEN=ore_sk_abc\n"


def test_read_env_value_last_assignment_wins(tmp_path):
    target = tmp_path / ".env"
    target.write_text("ORECLOUD_TOKEN=first\nORECLOUD_TOKEN=second\n")
    assert _env_file.read_env_value(target, "ORECLOUD_TOKEN") == "second"
    assert _env_file.read_env_value(target, "MISSING") is None
    assert _env_file.read_env_value(tmp_path / "absent", "ORECLOUD_TOKEN") is None


def test_gitignore_added_at_repo_root(tmp_path):
    (tmp_path / ".git").mkdir()
    note = _env_file.ensure_gitignored(tmp_path / ".env")
    assert "added .env" in note
    assert ".env\n" in (tmp_path / ".gitignore").read_text()
    # Second call sees the fresh entry and does nothing.
    assert _env_file.ensure_gitignored(tmp_path / ".env") is None


def test_gitignore_respects_existing_coverage_up_the_tree(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text("node_modules/\n.env\n")
    sub = tmp_path / "service"
    sub.mkdir()
    assert _env_file.ensure_gitignored(sub / ".env") is None
    assert not (sub / ".gitignore").exists()


def test_gitignore_appends_without_eating_the_last_line(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text("dist/")  # no trailing newline
    _env_file.ensure_gitignored(tmp_path / ".env")
    assert (tmp_path / ".gitignore").read_text() == "dist/\n.env\n"


def test_outside_a_git_repo_notes_instead_of_writing(tmp_path):
    note = _env_file.ensure_gitignored(tmp_path / ".env")
    assert "not inside a git repository" in note
    assert not (tmp_path / ".gitignore").exists()


def test_scratch_write_ignores_a_planted_symlink(tmp_path):
    """The scratch file is unique and O_EXCL: a pre-planted predictable
    `.<name>.tmp` symlink must never receive the secret (its target stays
    untouched) and the final file is still owner-only."""
    import os

    target = tmp_path / ".env"
    sentinel = tmp_path / "sentinel"
    sentinel.write_text("untouched\n")
    os.symlink(sentinel, tmp_path / "..env.tmp")

    _env_file.write_env_values(target, {"ORECLOUD_TOKEN": "ore_sk_abc"})

    assert sentinel.read_text() == "untouched\n"
    assert target.read_text() == "ORECLOUD_TOKEN=ore_sk_abc\n"
    _assert_owner_only(target)
