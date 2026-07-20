"""Credential resolution for the managed cloud: the baked-in hosted default
host, ORECLOUD_TOKEN-only environments (CI), and the fail-closed host-only
override. The `_config` tests need no third-party deps; the client/CLI tests
skip without the [cloud] extra's dependencies.
"""

import pytest

from lodedb.cloud import _config


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """An empty credentials dir and a clean credential environment, so tests
    never see the developer's real login or env vars."""
    monkeypatch.setenv("ORECLOUD_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("ORECLOUD_TOKEN", raising=False)
    monkeypatch.delenv("ORECLOUD_HOST", raising=False)


def test_env_token_alone_targets_the_hosted_default(isolated_config, monkeypatch):
    monkeypatch.setenv("ORECLOUD_TOKEN", "ore_pat_ci")
    creds = _config.load_credentials()
    assert creds == _config.Credentials(
        host=_config.DEFAULT_HOST, token="ore_pat_ci", source="env"
    )


def test_env_host_overrides_the_default(isolated_config, monkeypatch):
    monkeypatch.setenv("ORECLOUD_TOKEN", "ore_pat_ci")
    monkeypatch.setenv("ORECLOUD_HOST", "https://staging.example/")
    creds = _config.load_credentials()
    assert creds.host == "https://staging.example"
    assert creds.source == "env"


def test_env_token_never_borrows_the_file_host(isolated_config, monkeypatch):
    """A CI token must not aim at whatever control plane a developer's stored
    login points to — without ORECLOUD_HOST it targets the hosted default."""
    _config.save_credentials("https://other.example", "ore_pat_stored")
    monkeypatch.setenv("ORECLOUD_TOKEN", "ore_pat_ci")
    creds = _config.load_credentials()
    assert creds.host == _config.DEFAULT_HOST
    assert creds.token == "ore_pat_ci"


def test_env_host_alone_fails_closed(isolated_config, monkeypatch):
    """ORECLOUD_HOST without a token must not re-aim the stored credential at
    a control plane the login never targeted."""
    _config.save_credentials("https://other.example", "ore_pat_stored")
    monkeypatch.setenv("ORECLOUD_HOST", "https://staging.example")
    with pytest.raises(_config.CredentialsError, match="ORECLOUD_TOKEN"):
        _config.load_credentials()


def test_file_credentials_keep_their_own_host(isolated_config):
    """A stored self-hosted login is never silently re-aimed at the default."""
    _config.save_credentials("https://selfhost.example/", "ore_pat_stored")
    creds = _config.load_credentials()
    assert creds == _config.Credentials(
        host="https://selfhost.example", token="ore_pat_stored", source="file"
    )


def test_resolve_credentials_defaults_the_host(isolated_config):
    """`Client(token=...)` needs no URL: an explicit token with nothing else
    configured talks to the hosted control plane."""
    client = pytest.importorskip(
        "lodedb.cloud.client", reason="needs the [cloud] extra's dependencies"
    )
    assert client.resolve_credentials("ore_sk_x", None) == (_config.DEFAULT_HOST, "ore_sk_x")


def test_resolve_credentials_still_requires_a_token(isolated_config):
    client = pytest.importorskip(
        "lodedb.cloud.client", reason="needs the [cloud] extra's dependencies"
    )
    with pytest.raises(client.CloudError, match="no credential configured"):
        client.resolve_credentials(None, None)


def test_login_defaults_to_the_hosted_control_plane(isolated_config, monkeypatch):
    """`lodedb cloud login` with no --host lands on the hosted default (the
    control-plane call is faked; what's under test is the host choice)."""
    pytest.importorskip("httpx", reason="needs the [cloud] extra's dependencies")
    pytest.importorskip("nacl", reason="needs the [cloud] extra's dependencies")
    from typer.testing import CliRunner

    import lodedb.cloud.cli as cli

    class FakeClient:
        """Stands in for CloudClient: records the host, answers me()."""

        def __init__(self, host, token=None):
            self.host = host

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def me(self):
            return {"user": {"email": "ci@example.com"}}

    monkeypatch.setattr(cli, "CloudClient", FakeClient)
    result = CliRunner().invoke(cli.app, ["login", "--token", "ore_pat_ci"])
    assert result.exit_code == 0, result.output
    stored = _config.load_credentials()
    assert stored.host == _config.DEFAULT_HOST
    assert stored.source == "file"


def test_default_host_is_public_on_the_cloud_module():
    """`from lodedb.cloud import DEFAULT_HOST` works without the extra's
    dependencies — servers rendering docs conditionally rely on it."""
    from lodedb import cloud

    assert cloud.DEFAULT_HOST == _config.DEFAULT_HOST
    assert "DEFAULT_HOST" in cloud.__all__
