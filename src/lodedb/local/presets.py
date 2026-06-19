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
    ),
    "bge": LocalModelPreset(
        name="bge",
        route_profile=BGE_TURBOVEC_ROUTE_PROFILE,
        # BGE asymmetric retrieval prefix, matching the engine's
        # _default_query_prefix_for_policy for the BGE route profile.
        query_prefix=BGE_BASE_QUERY_PREFIX,
        document_prefix="",
        description="Quality: BAAI/bge-base-en-v1.5, 768-dim, 4-bit TurboVec.",
    ),
}


def resolve_preset(model: str) -> LocalModelPreset:
    """Returns a known local preset, raising a clear error for unknown names."""

    key = (model or "").strip().lower()
    if key not in LOCAL_MODEL_PRESETS:
        known = ", ".join(sorted(LOCAL_MODEL_PRESETS))
        raise ValueError(f"unknown local model preset {model!r}; choose one of: {known}")
    return LOCAL_MODEL_PRESETS[key]
