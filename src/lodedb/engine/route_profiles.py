"""Direct-TurboVec route profiles for the LodeDB engine.

The local LodeDB presets bind these route policies verbatim (model id, provider,
task, native dimension, TurboVec quantization width), so the compact
``.tvim``/``.tvd``/``.jsd`` storage is byte-identical to the engine path. The
local product uses the 4-bit MiniLM/BGE routes; the 2-bit variants are an
opt-in, storage-constrained tier.
"""

from __future__ import annotations

from typing import Any

from lodedb.engine.core import EngineRoutePolicy

BGE_BASE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BGE_TURBOVEC_ROUTE_PROFILE = "bge-turbovec"
MINILM_TURBOVEC_ROUTE_PROFILE = "minilm-turbovec"
CLIP_TURBOVEC_ROUTE_PROFILE = "clip-turbovec"
BGE_TURBOVEC_2BIT_ROUTE_PROFILE = "bge-turbovec-2bit"
MINILM_TURBOVEC_2BIT_ROUTE_PROFILE = "minilm-turbovec-2bit"
DEFAULT_ROUTE_PROFILE = MINILM_TURBOVEC_ROUTE_PROFILE

CLIENT_ROUTE_POLICIES: dict[str, EngineRoutePolicy] = {
    BGE_TURBOVEC_ROUTE_PROFILE: EngineRoutePolicy(
        profile=BGE_TURBOVEC_ROUTE_PROFILE,
        label="BGE direct TurboVec 4-bit",
        client_note="storage-first direct full-dimensional TurboVec route without PCA or rerank",
        model="BAAI/bge-base-en-v1.5",
        provider="local_open",
        task="direct-turbovec",
        native_dim=768,
        method_template="direct_turbovec_full768_bw4",
        experimental=False,
        index_backend="turbovec_direct",
        turbovec_bit_width=4,
    ),
    MINILM_TURBOVEC_ROUTE_PROFILE: EngineRoutePolicy(
        profile=MINILM_TURBOVEC_ROUTE_PROFILE,
        label="MiniLM direct TurboVec 4-bit",
        client_note="storage-first direct full-dimensional TurboVec route without PCA or rerank",
        model="sentence-transformers/all-MiniLM-L6-v2",
        provider="local_open",
        task="direct-turbovec",
        native_dim=384,
        method_template="direct_turbovec_full384_bw4",
        experimental=False,
        index_backend="turbovec_direct",
        turbovec_bit_width=4,
    ),
    CLIP_TURBOVEC_ROUTE_PROFILE: EngineRoutePolicy(
        profile=CLIP_TURBOVEC_ROUTE_PROFILE,
        label="CLIP direct TurboVec 4-bit (image + text)",
        client_note=(
            "shared image/text CLIP space; cross-modal cosine over the direct "
            "full-dimensional TurboVec route without PCA or rerank"
        ),
        model="sentence-transformers/clip-ViT-B-32",
        provider="local_open",
        task="direct-turbovec",
        native_dim=512,
        method_template="direct_turbovec_full512_bw4",
        experimental=False,
        index_backend="turbovec_direct",
        turbovec_bit_width=4,
    ),
    BGE_TURBOVEC_2BIT_ROUTE_PROFILE: EngineRoutePolicy(
        profile=BGE_TURBOVEC_2BIT_ROUTE_PROFILE,
        label="BGE direct TurboVec 2-bit (storage-constrained opt-in)",
        client_note=(
            "half the code bytes of bge-turbovec (192 vs 384 B/embed) with a "
            "dense-reference recall discount (GovReport5K 0.8688 vs 0.9631 at "
            "top-100) but GovReport end-task qrel recall indistinguishable "
            "from 4-bit; opt-in tier for storage-constrained deployments"
        ),
        model="BAAI/bge-base-en-v1.5",
        provider="local_open",
        task="direct-turbovec",
        native_dim=768,
        method_template="direct_turbovec_full768_bw2",
        experimental=False,
        index_backend="turbovec_direct",
        turbovec_bit_width=2,
    ),
    MINILM_TURBOVEC_2BIT_ROUTE_PROFILE: EngineRoutePolicy(
        profile=MINILM_TURBOVEC_2BIT_ROUTE_PROFILE,
        label="MiniLM direct TurboVec 2-bit (storage-constrained opt-in)",
        client_note=(
            "half the code bytes of minilm-turbovec (96 vs 192 B/embed) with a "
            "dense-reference recall discount (GovReport5K 0.8662 vs 0.9612 at "
            "top-100) but GovReport end-task qrel recall indistinguishable "
            "from 4-bit; opt-in tier for storage-constrained deployments"
        ),
        model="sentence-transformers/all-MiniLM-L6-v2",
        provider="local_open",
        task="direct-turbovec",
        native_dim=384,
        method_template="direct_turbovec_full384_bw2",
        experimental=True,
        index_backend="turbovec_direct",
        turbovec_bit_width=2,
    ),
}

# Direct TurboVec routes lead the recommended path; the 2-bit tier stays an
# explicit storage-constrained opt-in.
CLIENT_ROUTE_PROFILE_ORDER = (
    MINILM_TURBOVEC_ROUTE_PROFILE,
    BGE_TURBOVEC_ROUTE_PROFILE,
    CLIP_TURBOVEC_ROUTE_PROFILE,
    MINILM_TURBOVEC_2BIT_ROUTE_PROFILE,
    BGE_TURBOVEC_2BIT_ROUTE_PROFILE,
)


def route_policy_for_profile(profile: str) -> EngineRoutePolicy:
    """Returns the route policy for a runtime profile name."""

    try:
        return CLIENT_ROUTE_POLICIES[profile]
    except KeyError as exc:
        allowed = ", ".join(CLIENT_ROUTE_PROFILE_ORDER)
        raise ValueError(f"route profile must be one of: {allowed}") from exc


def client_route_policy_manifest() -> list[dict[str, Any]]:
    """Returns the route policies in recommended order as plain dicts."""

    return [CLIENT_ROUTE_POLICIES[name].to_dict() for name in CLIENT_ROUTE_PROFILE_ORDER]
