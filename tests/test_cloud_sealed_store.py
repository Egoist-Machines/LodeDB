"""Sealed-store client composition, HPKE contexts, and refusal handling."""

from __future__ import annotations

import base64
import json

import pytest

pytest.importorskip("httpx", reason="needs the [cloud] extra's dependencies")
pytest.importorskip("nacl", reason="needs the [cloud] extra's dependencies")

import httpx  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from lodedb.cloud import cli  # noqa: E402
from lodedb.cloud.client import Client  # noqa: E402
from lodedb.cloud.serving import CloudStore  # noqa: E402
from lodedb.cloud.transfer import CloudClient, CloudError  # noqa: E402


@pytest.fixture
def hpke_suite():
    """Provide the deployed HPKE suite and raw-X25519 serialization helpers."""

    pytest.importorskip("cryptography", reason="needs the [cloud-sealed] extra")
    from cryptography.hazmat.primitives import hpke, serialization
    from cryptography.hazmat.primitives.asymmetric import x25519

    suite = hpke.Suite(hpke.KEM.X25519, hpke.KDF.HKDF_SHA256, hpke.AEAD.AES_256_GCM)
    return suite, serialization, x25519


def _recipient_public_key(private_key, serialization) -> str:
    """Encode one test X25519 recipient public key in the server wire form."""

    raw = private_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    return base64.b64encode(raw).decode()


def test_encrypted_create_fetches_a_recipient_and_seals_bound_material(hpke_suite):
    """Encrypted creation encrypts the exact material under its create context."""

    suite, serialization, x25519 = hpke_suite
    private_key = x25519.X25519PrivateKey.generate()
    recipient_public_key = _recipient_public_key(private_key, serialization)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Serve the creation recipient and capture the create request."""

        requests.append(request)
        if request.method == "GET":
            assert request.url.path.endswith("/stores/create-challenge")
            return httpx.Response(200, json={"recipient_public_key": recipient_public_key})
        assert request.method == "POST"
        return httpx.Response(201, json={"store": "user-42", "encrypted": True})

    material = b"m" * 32
    with Client(
        token="ore_sk_test",
        host="http://testserver",
        org="acme",
        environment="prod",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = client.create_store(
            "user-42",
            encrypted=True,
            key_material=material,
            preset="minilm",
        )

    assert result["encrypted"] is True
    assert [request.method for request in requests] == ["GET", "POST"]
    body = json.loads(requests[1].content)
    assert body["encrypted"] is True
    assert body["mode"] == "cloud_writer"
    opened = suite.decrypt(
        base64.b64decode(body["sealed_material"], validate=True),
        private_key,
        b"orecloud/store-create/v1|org=acme|env=prod|store=user-42",
    )
    assert opened == material


def test_encrypted_create_rejects_invalid_material_before_a_request():
    """Cheap material validation avoids fetching an HPKE recipient needlessly."""

    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """Fail if a client-side validation case makes an HTTP request."""

        nonlocal calls
        calls += 1
        return httpx.Response(500)

    with Client(
        token="ore_sk_test",
        host="http://testserver",
        org="acme",
        environment="prod",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(ValueError, match="only to encrypted"):
            client.create_store("user-42", key_material=b"m" * 32)
        with pytest.raises(ValueError, match="exactly 32 bytes"):
            client.create_store("user-42", encrypted=True, key_material=b"short")
        with pytest.raises(ValueError, match="exactly 32 bytes"):
            client.create_store("user-42", encrypted=True)

    assert calls == 0


def test_unseal_uses_the_server_info_verbatim_and_returns_an_aware_expiry(hpke_suite):
    """Unseal echoes the standard nonce but seals against returned info bytes."""

    suite, serialization, x25519 = hpke_suite
    private_key = x25519.X25519PrivateKey.generate()
    recipient_public_key = _recipient_public_key(private_key, serialization)
    nonce = base64.b64encode(b"\xff" * 32).decode()
    challenge_info = b"orecloud/unseal/v1|db=another-store|nonce=__8="
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Serve one intentionally non-rebuilt challenge then accept unseal."""

        seen.append(request)
        if request.url.path.endswith("/challenge"):
            return httpx.Response(
                200,
                json={
                    "recipient_public_key": recipient_public_key,
                    "nonce": nonce,
                    "info": base64.b64encode(challenge_info).decode(),
                },
            )
        return httpx.Response(
            200,
            json={"store_id": "store-id", "expires_at": "2026-07-23T12:30:00Z"},
        )

    material = b"u" * 32
    with Client(
        token="ore_sk_test",
        host="http://testserver",
        org="acme",
        environment="prod",
        transport=httpx.MockTransport(handler),
    ) as client:
        expires_at = client.unseal_store("user-42", material, ttl_seconds=90)

    assert expires_at.tzinfo is not None and expires_at.utcoffset() is not None
    assert [request.method for request in seen] == ["POST", "POST"]
    body = json.loads(seen[1].content)
    assert body["nonce"] == nonce
    assert body["ttl_seconds"] == 90
    assert (
        suite.decrypt(
            base64.b64decode(body["sealed_material"], validate=True), private_key, challenge_info
        )
        == material
    )


class _ResealStub:
    """Duck-type only the composed client's reseal transport verb."""

    def __init__(self) -> None:
        """Record the tenancy and store passed to the transport verb."""

        self.calls: list[tuple[str, str, str]] = []

    def reseal_store(self, org: str, environment: str, store: str) -> dict:
        """Record a reseal request and answer that it removed a live grant."""

        self.calls.append((org, environment, store))
        return {"resealed": True}


def test_client_reseal_composes_over_the_thin_transport_stub():
    """The user-facing Client binds its tenancy before delegating reseal."""

    transport = _ResealStub()
    client = Client.__new__(Client)
    client.org = "acme"
    client.environment = "prod"
    client._client = transport

    assert client.reseal_store("user-42") is True
    assert transport.calls == [("acme", "prod", "user-42")]


def test_reseal_returns_the_server_bool_and_rotate_seals_fresh_material(hpke_suite):
    """Reseal is a thin bool result while rotation uses an unseal challenge."""

    suite, serialization, x25519 = hpke_suite
    private_key = x25519.X25519PrivateKey.generate()
    recipient_public_key = _recipient_public_key(private_key, serialization)
    challenge_info = b"orecloud/unseal/v1|db=store-id|nonce=dGVzdA=="
    nonce = base64.b64encode(b"test").decode()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Answer reseal, rotation challenge, and the rotation submission."""

        requests.append(request)
        raw_path = request.url.raw_path.decode()
        if raw_path.endswith("/stores/user%2F42/reseal"):
            return httpx.Response(200, json={"store_id": "store-id", "resealed": True})
        if raw_path.endswith("/stores/user%2F42/unseal/challenge"):
            return httpx.Response(
                200,
                json={
                    "recipient_public_key": recipient_public_key,
                    "nonce": nonce,
                    "info": base64.b64encode(challenge_info).decode(),
                },
            )
        assert raw_path.endswith("/stores/user%2F42/key/rotate")
        return httpx.Response(200, json={"store_id": "store-id"})

    new_material = b"r" * 32
    with Client(
        token="ore_sk_test",
        host="http://testserver",
        org="acme",
        environment="prod",
        transport=httpx.MockTransport(handler),
    ) as client:
        assert client.reseal_store("user/42") is True
        assert client.rotate_store_key("user/42", new_material) is None

    body = json.loads(requests[-1].content)
    assert body["nonce"] == nonce
    assert (
        suite.decrypt(
            base64.b64decode(body["sealed_material"], validate=True), private_key, challenge_info
        )
        == new_material
    )


def test_sealed_search_preserves_the_423_refusal_without_a_retry_loop():
    """A sealed data-plane read stays an inspectable CloudError for callers."""

    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """Return the deployed sealed-store refusal to one search request."""

        nonlocal calls
        calls += 1
        assert request.url.path.endswith("/stores/search")
        return httpx.Response(
            423,
            json={
                "detail": "store_sealed: this encrypted store is sealed; unseal it before querying"
            },
        )

    with CloudClient(
        "http://testserver", "ore_sk_test", transport=httpx.MockTransport(handler)
    ) as client:
        store = CloudStore(client, "acme", "prod", "user-42", owns_client=False)
        with pytest.raises(CloudError) as caught:
            store.search("hello")

    assert caught.value.status_code == 423
    assert caught.value.detail.startswith("store_sealed:")
    assert calls == 1


def test_missing_cryptography_names_the_cloud_sealed_install_extra(monkeypatch):
    """The sealing helper gives a targeted install hint when crypto is absent."""

    from lodedb.cloud import _sealing

    real_import_module = _sealing.importlib.import_module

    def unavailable(name: str, package: str | None = None):
        """Pretend every cryptography import is unavailable for this call."""

        if name.startswith("cryptography"):
            raise ImportError("cryptography unavailable")
        return real_import_module(name, package)

    monkeypatch.setattr(_sealing.importlib, "import_module", unavailable)
    with pytest.raises(ImportError, match=r"lodedb\[cloud-sealed\]"):
        _sealing.seal_material(b"m" * 32, base64.b64encode(b"p" * 32).decode(), b"info")


def test_cli_missing_cryptography_is_a_classified_error(monkeypatch):
    """The sealed-store CLI never leaks the optional-dependency traceback."""

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def store_create_challenge(self, _org: str, _environment: str) -> dict:
            return {"recipient_public_key": base64.b64encode(b"p" * 32).decode()}

        def create_store(self, *_args, **_kwargs):
            raise AssertionError("create must not run when sealing is unavailable")

    from lodedb.cloud import _sealing

    def unavailable(*_args, **_kwargs):
        raise ImportError("sealed-store support requires cryptography; run: cloud-sealed")

    monkeypatch.setattr(cli, "_client", FakeClient)
    monkeypatch.setattr(cli, "_tenancy", lambda *_args: ("acme", "prod"))
    monkeypatch.setattr(_sealing, "seal_material", unavailable)
    monkeypatch.setenv("SEALED_MATERIAL", base64.b64encode(b"m" * 32).decode())

    result = CliRunner().invoke(
        cli.app,
        [
            "store",
            "create",
            "user-42",
            "--encrypted",
            "--material-env",
            "SEALED_MATERIAL",
            "--no-connect-key",
        ],
    )

    assert result.exit_code == cli.EXIT_USAGE
    assert "error: sealed-store support requires cryptography" in result.output
    assert "Traceback" not in result.output
