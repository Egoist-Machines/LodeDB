"""Local model presets mapped onto the existing engine route profiles.

The local layer reuses the engine's direct-TurboVec route policies verbatim
(model id, provider, task, native dim, quantization width, query/document
prefixes) so the compact storage and ``.tvim``/``.tvd``/``.jsd`` persistence are
byte-identical to the engine path. ``minilm`` is the fast default; ``bge`` is
the higher-quality preset.
"""

from __future__ import annotations

from dataclasses import dataclass

from lodedb.engine.core import EngineRoutePolicy
from lodedb.engine.route_profiles import (
    BGE_BASE_QUERY_PREFIX,
    BGE_TURBOVEC_ROUTE_PROFILE,
    MINILM_TURBOVEC_ROUTE_PROFILE,
    route_policy_for_profile,
)


@dataclass(frozen=True)
class LocalModelPreset:
    """Binds a friendly local model name to an engine route profile + prefixes."""

    name: str
    route_profile: str
    query_prefix: str
    document_prefix: str
    description: str
    # Sentence pooling the model was trained with, used by the ONNX runtime to
    # reproduce the sentence-transformers vector. The torch path reads pooling
    # from the model's own config, so this only matters for the ONNX backend.
    pooling: str = "mean"

    @property
    def route_policy(self) -> EngineRoutePolicy:
        """Returns the underlying engine route policy (reused, not redefined)."""

        return route_policy_for_profile(self.route_profile)

    @property
    def model_name(self) -> str:
        """Returns the HuggingFace model id required by this route profile."""

        return self.route_policy.model

    @property
    def native_dim(self) -> int:
        """Returns the embedding dimension fixed by this route profile."""

        return self.route_policy.native_dim

    @property
    def turbovec_bit_width(self) -> int:
        """Returns the TurboVec quantization width fixed by this route profile."""

        return int(self.route_policy.turbovec_bit_width or 4)


LOCAL_MODEL_PRESETS: dict[str, LocalModelPreset] = {
    "minilm": LocalModelPreset(
        name="minilm",
        route_profile=MINILM_TURBOVEC_ROUTE_PROFILE,
        query_prefix="",
        document_prefix="",
        description="Fast default: all-MiniLM-L6-v2, 384-dim, 4-bit TurboVec.",
        pooling="mean",
    ),
    "bge": LocalModelPreset(
        name="bge",
        route_profile=BGE_TURBOVEC_ROUTE_PROFILE,
        # BGE asymmetric retrieval prefix, matching the engine's
        # _default_query_prefix_for_policy for the BGE route profile.
        query_prefix=BGE_BASE_QUERY_PREFIX,
        document_prefix="",
        description="Quality: BAAI/bge-base-en-v1.5, 768-dim, 4-bit TurboVec.",
        pooling="cls",
    ),
}


def resolve_preset(model: str) -> LocalModelPreset:
    """Returns a known local preset, raising a clear error for unknown names."""

    key = (model or "").strip().lower()
    if key not in LOCAL_MODEL_PRESETS:
        known = ", ".join(sorted(LOCAL_MODEL_PRESETS))
        raise ValueError(f"unknown local model preset {model!r}; choose one of: {known}")
    return LOCAL_MODEL_PRESETS[key]


VECTOR_ONLY_ROUTE_PROFILE = "vector-only"


def vector_only_route_policy(native_dim: int, *, bit_width: int = 4) -> EngineRoutePolicy:
    """Returns a route policy for a bring-your-own-vectors index (no embedder).

    Unlike the preset profiles, this is built on demand at a caller-chosen
    dimension and is not registered in the client route-policy manifest. It pins a
    stable, redacted identity (``model="external"``, ``task="vector-only"``) that
    the engine persists in the snapshot header and re-enforces on reopen, so a
    vector-only index stays self-consistent and cannot be silently reopened as a
    preset index (or vice versa) without the dim/identity round-trip catching it.
    The ``turbovec_direct`` backend keeps the TurboVec-availability guard and the
    O(changed) commit path identical to the preset routes.
    """

    return EngineRoutePolicy(
        profile=VECTOR_ONLY_ROUTE_PROFILE,
        label="External vectors (bring-your-own)",
        client_note="caller-supplied embeddings; no internal embedding model",
        model="external",
        provider="external",
        task="vector-only",
        native_dim=int(native_dim),
        method_template=f"direct_turbovec_full{int(native_dim)}_bw{int(bit_width)}",
        experimental=False,
        index_backend="turbovec_direct",
        turbovec_bit_width=int(bit_width),
    )
