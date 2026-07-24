"""Lazy HPKE helpers for delegated-custody sealed stores."""

from __future__ import annotations

import base64
import binascii
import importlib
import secrets
from typing import Any

_INSTALL_HINT = "pip install 'lodedb[cloud-sealed]'"


def _hpke_dependencies() -> tuple[Any, Any]:
    """Load the optional HPKE modules only when a caller seals material."""

    try:
        hpke = importlib.import_module("cryptography.hazmat.primitives.hpke")
        x25519 = importlib.import_module("cryptography.hazmat.primitives.asymmetric.x25519")
    except ImportError:
        raise ImportError(
            "sealed-store support requires cryptography; run: " + _INSTALL_HINT
        ) from None
    return hpke, x25519


def seal_material(material: bytes, recipient_public_key_b64: str, info: bytes) -> str:
    """HPKE-seal 32-byte caller-held material to a recipient and context."""

    if not isinstance(material, bytes) or len(material) != 32:
        raise ValueError("delegated material must be exactly 32 bytes")
    hpke, x25519 = _hpke_dependencies()
    try:
        raw_public_key = base64.b64decode(recipient_public_key_b64, validate=True)
        public_key = x25519.X25519PublicKey.from_public_bytes(raw_public_key)
    except (binascii.Error, TypeError, ValueError) as error:
        raise ValueError("recipient public key must be base64 raw 32 bytes") from error
    if len(raw_public_key) != 32:
        raise ValueError("recipient public key must be base64 raw 32 bytes")
    suite = hpke.Suite(hpke.KEM.X25519, hpke.KDF.HKDF_SHA256, hpke.AEAD.AES_256_GCM)
    return base64.b64encode(suite.encrypt(material, public_key, info)).decode()


def create_info(org: str, environment: str, store: str) -> bytes:
    """Build the server-defined HPKE context for encrypted store creation."""

    return (
        b"orecloud/store-create/v1|org="
        + org.encode()
        + b"|env="
        + environment.encode()
        + b"|store="
        + store.encode()
    )


def new_key_material() -> bytes:
    """Return fresh 32-byte delegated wrapping material for a new store."""

    return secrets.token_bytes(32)
