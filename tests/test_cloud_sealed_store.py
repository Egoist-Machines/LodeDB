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


def _external_seal(material: bytes, recipient_public_key: str, info: bytes, suite, x25519) -> str:
    """Seal material as a separate holder would."""

    public_key = x25519.X25519PublicKey.from_public_bytes(
        base64.b64decode(recipient_public_key, validate=True)
    )
    return base64.b64encode(suite.encrypt(material, public_key, info)).decode()


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


def test_relayed_unseal_accepts_external_sealed_material_and_validates_expiry(hpke_suite):
    """A relay can submit material sealed by a separate holder."""

    suite, serialization, x25519 = hpke_suite
    private_key = x25519.X25519PrivateKey.generate()
    recipient_public_key = _recipient_public_key(private_key, serialization)
    nonce = base64.b64encode(b"relay-nonce").decode()
    challenge_info = b"orecloud/unseal/v1|db=store-id|nonce=cmVsYXktbm9uY2U="
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Serve a relayed challenge and accept the submitted sealed blob."""

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

    material = b"e" * 32
    with Client(
        token="ore_sk_test",
        host="http://testserver",
        org="acme",
        environment="prod",
        transport=httpx.MockTransport(handler),
    ) as client:
        challenge = client.unseal_challenge("user-42")
        sealed_material = _external_seal(
            material,
            challenge["recipient_public_key"],
            base64.b64decode(challenge["info"], validate=True),
            suite,
            x25519,
        )
        expires_at = client.unseal_store_sealed(
            "user-42", sealed_material, challenge["nonce"], ttl_seconds=120
        )

    assert expires_at.tzinfo is not None and expires_at.utcoffset() is not None
    assert [request.method for request in seen] == ["POST", "POST"]
    body = json.loads(seen[1].content)
    assert body["nonce"] == nonce
    assert body["ttl_seconds"] == 120
    assert (
        suite.decrypt(
            base64.b64decode(body["sealed_material"], validate=True), private_key, challenge_info
        )
        == material
    )

    def naive_handler(request: httpx.Request) -> httpx.Response:
        """Return an invalid naive expiry for a relayed submission."""

        assert request.url.path.endswith("/unseal")
        return httpx.Response(
            200,
            json={"store_id": "store-id", "expires_at": "2026-07-23T12:30:00"},
        )

    with Client(
        token="ore_sk_test",
        host="http://testserver",
        org="acme",
        environment="prod",
        transport=httpx.MockTransport(naive_handler),
    ) as client:
        with pytest.raises(ValueError, match="unseal response returned a naive expires_at"):
            client.unseal_store_sealed("user-42", "sealed", nonce)


@pytest.mark.parametrize("missing", ["recipient_public_key", "nonce", "info"])
def test_unseal_challenge_missing_field_names_it(missing):
    """A relayed challenge refusal points at the absent field."""

    challenge = {
        "recipient_public_key": base64.b64encode(b"p" * 32).decode(),
        "nonce": base64.b64encode(b"nonce").decode(),
        "info": base64.b64encode(b"info").decode(),
    }
    del challenge[missing]

    def handler(_request: httpx.Request) -> httpx.Response:
        """Return one incomplete challenge."""

        return httpx.Response(200, json=challenge)

    with Client(
        token="ore_sk_test",
        host="http://testserver",
        org="acme",
        environment="prod",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(ValueError, match=missing):
            client.unseal_challenge("user-42")


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


def test_sealed_store_round_trip_reseal_returns_423_without_retry(hpke_suite):
    """Create, unseal, query, reseal, then surface one sealed refusal."""

    from lodedb.cloud._sealing import create_info

    suite, serialization, x25519 = hpke_suite
    private_key = x25519.X25519PrivateKey.generate()
    recipient_public_key = _recipient_public_key(private_key, serialization)
    material = b"s" * 32
    nonce = base64.b64encode(b"round-trip-nonce").decode()
    challenge_info = b"orecloud/unseal/v1|db=store-id|nonce=cm91bmQtdHJpcC1ub25jZQ=="
    state = {"created": False, "grant": False, "sealed_searches": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        """Act like a sealed store through create, grant, read, and reseal."""

        raw_path = request.url.raw_path.decode()
        if request.method == "GET" and raw_path.endswith("/stores/create-challenge"):
            return httpx.Response(200, json={"recipient_public_key": recipient_public_key})
        if request.method == "POST" and raw_path.endswith("/stores"):
            body = json.loads(request.content)
            assert (
                suite.decrypt(
                    base64.b64decode(body["sealed_material"], validate=True),
                    private_key,
                    create_info("acme", "prod", "user-42"),
                )
                == material
            )
            state["created"] = True
            return httpx.Response(
                201,
                json={
                    "store": "user-42",
                    "key": "memory",
                    "mode": "cloud_writer",
                    "encrypted": True,
                },
            )
        if raw_path.endswith("/stores/user-42/unseal/challenge"):
            return httpx.Response(
                200,
                json={
                    "recipient_public_key": recipient_public_key,
                    "nonce": nonce,
                    "info": base64.b64encode(challenge_info).decode(),
                },
            )
        if raw_path.endswith("/stores/user-42/unseal"):
            body = json.loads(request.content)
            assert body["nonce"] == nonce
            assert (
                suite.decrypt(
                    base64.b64decode(body["sealed_material"], validate=True),
                    private_key,
                    challenge_info,
                )
                == material
            )
            state["grant"] = True
            return httpx.Response(
                200,
                json={"store_id": "store-id", "expires_at": "2026-07-23T12:30:00Z"},
            )
        if raw_path.endswith("/stores/search"):
            body = json.loads(request.content)
            assert state["created"] is True
            assert body["store"] == "user-42"
            if state["grant"]:
                return httpx.Response(
                    200,
                    json={"hits": [{"score": 0.9, "id": "doc-1", "metadata": {}}]},
                )
            state["sealed_searches"] += 1
            return httpx.Response(
                423,
                json={
                    "detail": (
                        "store_sealed: this encrypted store is sealed; "
                        "unseal it before querying"
                    )
                },
            )
        if raw_path.endswith("/stores/user-42/reseal"):
            state["grant"] = False
            return httpx.Response(200, json={"store_id": "store-id", "resealed": True})
        raise AssertionError(f"unexpected request {request.method} {raw_path}")

    with Client(
        token="ore_sk_test",
        host="http://testserver",
        org="acme",
        environment="prod",
        transport=httpx.MockTransport(handler),
    ) as client:
        created = client.create_store(
            "user-42", encrypted=True, key_material=material, preset="minilm"
        )
        assert created["encrypted"] is True
        client.unseal_store("user-42", material, ttl_seconds=90)
        store = client.store("user-42")
        assert store.search("hello")[0].id == "doc-1"
        assert client.reseal_store("user-42") is True
        with pytest.raises(CloudError) as caught:
            store.search("hello")

    assert caught.value.status_code == 423
    assert caught.value.detail.startswith("store_sealed:")
    assert state["sealed_searches"] == 1


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


def test_cli_rotate_seals_new_material_and_reports_success(monkeypatch, hpke_suite):
    """The rotate command seals the new material through the SDK facade."""

    suite, serialization, x25519 = hpke_suite
    private_key = x25519.X25519PrivateKey.generate()
    recipient_public_key = _recipient_public_key(private_key, serialization)
    challenge_info = b"orecloud/unseal/v1|db=store-id|nonce=cm90YXRl"
    nonce = base64.b64encode(b"rotate").decode()
    captured: dict[str, object] = {}

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def store_unseal_challenge(self, org: str, environment: str, store: str) -> dict:
            captured["challenge"] = (org, environment, store)
            return {
                "recipient_public_key": recipient_public_key,
                "nonce": nonce,
                "info": base64.b64encode(challenge_info).decode(),
            }

        def rotate_store_key(self, org: str, environment: str, store: str, payload: dict) -> dict:
            captured["rotate"] = (org, environment, store, payload)
            return {"store_id": "store-id"}

    new_material = b"n" * 32
    monkeypatch.setattr(cli, "_client", FakeClient)
    monkeypatch.setattr(cli, "_tenancy", lambda *_args: ("acme", "prod"))
    monkeypatch.setenv("NEW_MATERIAL", base64.b64encode(new_material).decode())

    result = CliRunner().invoke(
        cli.app,
        ["--no-json", "store", "rotate", "user-42", "--material-env", "NEW_MATERIAL"],
    )

    assert result.exit_code == 0, result.output
    assert "rotated user-42 key" in result.output
    assert captured["challenge"] == ("acme", "prod", "user-42")
    org, environment, store, payload = captured["rotate"]
    assert (org, environment, store) == ("acme", "prod", "user-42")
    assert payload["nonce"] == nonce
    assert (
        suite.decrypt(
            base64.b64decode(payload["sealed_material"], validate=True),
            private_key,
            challenge_info,
        )
        == new_material
    )


def test_cli_rotate_requires_material(monkeypatch):
    """Rotate needs either env-sourced or generated new material."""

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(cli, "_client", FakeClient)
    monkeypatch.setattr(cli, "_tenancy", lambda *_args: ("acme", "prod"))

    result = CliRunner().invoke(cli.app, ["store", "rotate", "user-42"])

    assert result.exit_code == cli.EXIT_USAGE
    assert "sealed stores need --material-env ENVVAR or --generate-material" in result.output


def test_cli_rotate_409_hints_to_unseal_first(monkeypatch, hpke_suite):
    """A rotate without a live grant is a refused command with an unseal hint."""

    _suite, serialization, x25519 = hpke_suite
    private_key = x25519.X25519PrivateKey.generate()
    recipient_public_key = _recipient_public_key(private_key, serialization)

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def store_unseal_challenge(self, _org: str, _environment: str, _store: str) -> dict:
            return {
                "recipient_public_key": recipient_public_key,
                "nonce": base64.b64encode(b"rotate").decode(),
                "info": base64.b64encode(b"info").decode(),
            }

        def rotate_store_key(self, *_args, **_kwargs) -> dict:
            raise CloudError(409, "store has no live unseal grant")

    monkeypatch.setattr(cli, "_client", FakeClient)
    monkeypatch.setattr(cli, "_tenancy", lambda *_args: ("acme", "prod"))
    monkeypatch.setenv("NEW_MATERIAL", base64.b64encode(b"n" * 32).decode())

    result = CliRunner().invoke(
        cli.app,
        ["store", "rotate", "user-42", "--material-env", "NEW_MATERIAL"],
    )

    assert result.exit_code == cli.EXIT_REFUSED
    assert "store has no live unseal grant (HTTP 409)" in result.output
    assert "hint: run `lodedb cloud store unseal` first" in result.output
    assert "Traceback" not in result.output


def test_cli_rotate_json_emits_the_server_row(monkeypatch, hpke_suite):
    """JSON mode passes the rotate response through, like unseal and reseal."""

    _suite, serialization, x25519 = hpke_suite
    private_key = x25519.X25519PrivateKey.generate()
    recipient_public_key = _recipient_public_key(private_key, serialization)

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def store_unseal_challenge(self, _org: str, _environment: str, _store: str) -> dict:
            return {
                "recipient_public_key": recipient_public_key,
                "nonce": base64.b64encode(b"rotate").decode(),
                "info": base64.b64encode(b"info").decode(),
            }

        def rotate_store_key(self, *_args, **_kwargs) -> dict:
            return {"store_id": "store-id"}

    monkeypatch.setattr(cli, "_client", FakeClient)
    monkeypatch.setattr(cli, "_tenancy", lambda *_args: ("acme", "prod"))
    monkeypatch.setenv("NEW_MATERIAL", base64.b64encode(b"j" * 32).decode())

    result = CliRunner().invoke(
        cli.app,
        ["--json", "store", "rotate", "user-42", "--material-env", "NEW_MATERIAL"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"store_id": "store-id"}


def test_cli_rotate_generated_material_failure_notes_no_effect(monkeypatch, hpke_suite):
    """A rotate that fails after printing generated material says so."""

    _suite, serialization, x25519 = hpke_suite
    private_key = x25519.X25519PrivateKey.generate()
    recipient_public_key = _recipient_public_key(private_key, serialization)

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def store_unseal_challenge(self, _org: str, _environment: str, _store: str) -> dict:
            return {
                "recipient_public_key": recipient_public_key,
                "nonce": base64.b64encode(b"rotate").decode(),
                "info": base64.b64encode(b"info").decode(),
            }

        def rotate_store_key(self, *_args, **_kwargs) -> dict:
            raise CloudError(409, "store has no live unseal grant")

    monkeypatch.setattr(cli, "_client", FakeClient)
    monkeypatch.setattr(cli, "_tenancy", lambda *_args: ("acme", "prod"))

    result = CliRunner().invoke(cli.app, ["store", "rotate", "user-42", "--generate-material"])

    assert result.exit_code == cli.EXIT_REFUSED
    assert "Keep this sealed-store material safe" in result.output
    assert "the printed material did not take effect" in result.output
    assert "hint: run `lodedb cloud store unseal` first" in result.output
    assert "Traceback" not in result.output


def test_cli_network_failure_is_a_classified_retry(monkeypatch):
    """A transport error prints one retryable error line, not a traceback."""

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def reseal_store(self, *_args, **_kwargs):
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(cli, "_client", FakeClient)
    monkeypatch.setattr(cli, "_tenancy", lambda *_args: ("acme", "prod"))

    result = CliRunner().invoke(cli.app, ["store", "reseal", "user-42"])

    assert result.exit_code == cli.EXIT_RETRY
    assert "error: could not reach the control plane: connection refused" in result.output
    assert "hint: check the network and ORECLOUD_HOST, then retry" in result.output
    assert "Traceback" not in result.output


def test_cli_network_failure_during_tenancy_is_classified(monkeypatch):
    """Tenancy resolution reaches the control plane before any verb; an
    unreachable host there is the same retryable refusal, not a traceback."""

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def token_self(self):
            raise httpx.ConnectError("nodename nor servname provided")

    monkeypatch.setattr(cli, "_client", FakeClient)

    result = CliRunner().invoke(cli.app, ["store", "reseal", "user-42"])

    assert result.exit_code == cli.EXIT_RETRY
    assert "error: could not reach the control plane" in result.output
    assert "Traceback" not in result.output


def test_cli_ambiguous_transport_failure_is_not_marked_retryable(monkeypatch):
    """A response lost mid-request may have been applied; no blind-retry hint."""

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def reseal_store(self, *_args, **_kwargs):
            raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(cli, "_client", FakeClient)
    monkeypatch.setattr(cli, "_tenancy", lambda *_args: ("acme", "prod"))

    result = CliRunner().invoke(cli.app, ["store", "reseal", "user-42"])

    assert result.exit_code == cli.EXIT_UNEXPECTED
    assert "error: the control plane connection failed mid-request" in result.output
    assert "may or may not have been applied" in result.output
    assert "Traceback" not in result.output


def test_cli_rotate_ambiguous_failure_says_keep_generated_material(monkeypatch, hpke_suite):
    """A lost rotate response must not tell the user to discard the material."""

    _suite, serialization, x25519 = hpke_suite
    private_key = x25519.X25519PrivateKey.generate()
    recipient_public_key = _recipient_public_key(private_key, serialization)

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def store_unseal_challenge(self, _org: str, _environment: str, _store: str) -> dict:
            return {
                "recipient_public_key": recipient_public_key,
                "nonce": base64.b64encode(b"rotate").decode(),
                "info": base64.b64encode(b"info").decode(),
            }

        def rotate_store_key(self, *_args, **_kwargs) -> dict:
            raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(cli, "_client", FakeClient)
    monkeypatch.setattr(cli, "_tenancy", lambda *_args: ("acme", "prod"))

    result = CliRunner().invoke(cli.app, ["store", "rotate", "user-42", "--generate-material"])

    assert result.exit_code == cli.EXIT_UNEXPECTED
    assert "Keep this sealed-store material safe" in result.output
    assert "the rotation outcome is unknown" in result.output
    assert "did not take effect" not in result.output
    assert "Traceback" not in result.output


def test_cli_interrupted_read_stays_retryable(monkeypatch):
    """A GET that dies mid-response changed nothing; blind retry is safe."""

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def token_self(self):
            raise httpx.ReadTimeout(
                "timed out", request=httpx.Request("GET", "http://testserver/v1/tokens/self")
            )

    monkeypatch.setattr(cli, "_client", FakeClient)

    result = CliRunner().invoke(cli.app, ["whoami"])

    assert result.exit_code == cli.EXIT_RETRY
    assert "error: could not reach the control plane" in result.output
    assert "Traceback" not in result.output


def test_cli_malformed_host_is_a_usage_error(monkeypatch):
    """A host without an http(s) scheme fails every retry; classify as usage."""

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def reseal_store(self, *_args, **_kwargs):
            raise httpx.UnsupportedProtocol(
                "Request URL is missing an 'http://' or 'https://' protocol."
            )

    monkeypatch.setattr(cli, "_client", FakeClient)
    monkeypatch.setattr(cli, "_tenancy", lambda *_args: ("acme", "prod"))

    result = CliRunner().invoke(cli.app, ["store", "reseal", "user-42"])

    assert result.exit_code == cli.EXIT_USAGE
    assert "error: invalid control-plane host" in result.output
    assert "Traceback" not in result.output


def test_cli_rotate_5xx_failure_says_keep_generated_material(monkeypatch, hpke_suite):
    """A 5xx rotation answer may follow a committed re-wrap; keep the material."""

    _suite, serialization, x25519 = hpke_suite
    private_key = x25519.X25519PrivateKey.generate()
    recipient_public_key = _recipient_public_key(private_key, serialization)

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def store_unseal_challenge(self, _org: str, _environment: str, _store: str) -> dict:
            return {
                "recipient_public_key": recipient_public_key,
                "nonce": base64.b64encode(b"rotate").decode(),
                "info": base64.b64encode(b"info").decode(),
            }

        def rotate_store_key(self, *_args, **_kwargs) -> dict:
            raise CloudError(503, "upstream unavailable")

    monkeypatch.setattr(cli, "_client", FakeClient)
    monkeypatch.setattr(cli, "_tenancy", lambda *_args: ("acme", "prod"))

    result = CliRunner().invoke(cli.app, ["store", "rotate", "user-42", "--generate-material"])

    assert result.exit_code == cli.EXIT_RETRY
    assert "Keep this sealed-store material safe" in result.output
    assert "the rotation outcome is unknown" in result.output
    assert "did not take effect" not in result.output
    assert "Traceback" not in result.output


def test_cli_syntactically_invalid_host_is_a_usage_error(monkeypatch):
    """httpx.InvalidURL strikes at client construction, outside every verb
    wrapper; it must still classify instead of printing a traceback."""

    creds = cli._config.Credentials(host="http://example.com:bad", token="ore_pat_x", source="env")
    monkeypatch.setattr(cli, "_load_credentials", lambda: creds)

    result = CliRunner().invoke(cli.app, ["store", "reseal", "user-42"])

    assert result.exit_code == cli.EXIT_USAGE
    assert "error: invalid control-plane host" in result.output
    assert "Traceback" not in result.output


def test_cli_proxy_failure_stays_retryable(monkeypatch):
    """A rejected proxy tunnel precedes submission; blind retry is safe."""

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def reseal_store(self, *_args, **_kwargs):
            raise httpx.ProxyError("407 Proxy Authentication Required")

    monkeypatch.setattr(cli, "_client", FakeClient)
    monkeypatch.setattr(cli, "_tenancy", lambda *_args: ("acme", "prod"))

    result = CliRunner().invoke(cli.app, ["store", "reseal", "user-42"])

    assert result.exit_code == cli.EXIT_RETRY
    assert "error: could not reach the control plane" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("host", ["http://", "example.com", "ftp://example.com", "http://[::1"])
def test_cli_authority_less_or_schemeless_host_is_a_usage_error(monkeypatch, host):
    """httpx accepts a bare `http://` base URL and only fails at the first
    request; the CLI must classify it as configuration, not transient."""

    creds = cli._config.Credentials(host=host, token="ore_pat_x", source="env")
    monkeypatch.setattr(cli, "_load_credentials", lambda: creds)

    result = CliRunner().invoke(cli.app, ["store", "reseal", "user-42"])

    assert result.exit_code == cli.EXIT_USAGE
    assert "error: invalid control-plane host" in result.output
    assert "Traceback" not in result.output
