"""GPU model server that reproduces Trieve's dense/sparse/rerank HTTP contracts.

One FastAPI/uvicorn process on :7070 serving every model Trieve calls at
request time, plus the static OIDC discovery stub that trieve-server discovers
at boot. All models run on CUDA.

Endpoints (shapes verified against the pinned Trieve source, SHA a99b21e2):

- ``POST /embeddings`` (OpenAI-style, note the ``?api-version`` query param Trieve
  always appends). ``input`` may be a bare string or an array of strings (Trieve's
  ``EmbeddingInput`` is a serde-untagged enum: doc batches send an array, single
  queries send a bare string with the query prefix prepended). The response MUST
  carry a top-level ``usage`` object; Trieve's ``DenseEmbedData.usage`` is not
  ``Option``, so a missing usage fails deserialization and triggers retries.
- ``POST /embed_sparse`` SPLADE. Request ``{inputs, encode_type, truncate}``;
  response is an outer array per input, each an inner list of ``{index, value}``
  (``index`` u32 vocab id, ``value`` f32). ``encode_type`` picks the doc vs query
  SPLADE checkpoint.
- ``POST /rerank`` cross-encoder. Request ``{query, texts, truncate}``; response is
  ``[{index, score}]`` where ``index`` is the position in the request ``texts``.
  Trieve maps scores back by that index (out-of-range panics its ``index_mut``) and
  sorts by score itself (NaN panics its sort), so every index in ``0..len(texts)``
  must appear exactly once with a finite score.
- ``GET /.well-known/openid-configuration`` + ``GET /jwks`` OIDC stub. Trieve's
  ``build_oidc_client`` runs ``CoreProviderMetadata::discover_async(...).unwrap()``
  at boot; discovery fetches the JWKS eagerly, so both must serve valid JSON and
  the ``issuer`` must byte-match ``OIDC_ISSUER_URL``.

The pure tensor transforms (dense normalize, SPLADE pool, rerank scoring, response
shaping) live in importable module-level functions so ``self_test.py`` can validate
the math on synthetic tensors without CUDA, real weights, or Modal.
"""

from __future__ import annotations

import os
from typing import Any

# Model ids Trieve's config points at (shared dense model + Trieve's SPLADE + reranker).
DENSE_MODEL = os.environ.get("DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SPLADE_DOC_MODEL = os.environ.get("SPLADE_DOC_MODEL", "naver/efficient-splade-VI-BT-large-doc")
SPLADE_QUERY_MODEL = os.environ.get(
    "SPLADE_QUERY_MODEL", "naver/efficient-splade-VI-BT-large-query"
)
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-base")
MODEL_SERVER_PORT = int(os.environ.get("MODEL_SERVER_PORT", "7070"))
# Trieve's OPENAI_API_KEY / issuer are read from the same env the orchestrator sets.
OIDC_ISSUER_URL = os.environ.get("OIDC_ISSUER_URL", f"http://localhost:{MODEL_SERVER_PORT}")
MAX_SEQ_LEN = int(os.environ.get("MODEL_SERVER_MAX_SEQ_LEN", "512"))


# -- pure transforms (self-testable without weights/CUDA) -------------------


def l2_normalize_rows(matrix: Any) -> Any:
    """Returns a row-wise L2-normalized float32 array (zero rows left unchanged)."""

    import numpy as np

    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (array / norms).astype(np.float32)


def dense_response(embeddings: Any) -> dict[str, Any]:
    """Shapes a normalized embedding matrix into Trieve's DenseEmbedData JSON.

    ``usage`` is mandatory (Trieve's struct field is not Option); the token counts
    are cosmetic for a local server, so zeros are fine as long as the keys exist.
    """

    normalized = l2_normalize_rows(embeddings)
    data = [{"embedding": [float(value) for value in row]} for row in normalized]
    return {"data": data, "usage": {"prompt_tokens": 0, "total_tokens": 0}}


def splade_pool(logits: Any, attention_mask: Any) -> Any:
    """Returns SPLADE weights: max-pool of log(1+ReLU(logits)) over live tokens.

    ``logits`` is (batch, seq_len, vocab) MLM output; ``attention_mask`` is
    (batch, seq_len). Padding positions are masked to zero before the max so
    padded tokens never contribute. Output is (batch, vocab), all non-negative.
    """

    import numpy as np

    logits_array = np.asarray(logits, dtype=np.float32)
    mask = np.asarray(attention_mask, dtype=np.float32)[:, :, None]
    activated = np.log1p(np.maximum(logits_array, 0.0)) * mask
    return np.max(activated, axis=1).astype(np.float32)


def splade_terms(weights_row: Any) -> list[dict[str, Any]]:
    """Returns the nonzero SPLADE terms of one row as [{index, value}] (u32/f32)."""

    import numpy as np

    row = np.asarray(weights_row, dtype=np.float32)
    nonzero = np.nonzero(row)[0]
    return [{"index": int(index), "value": float(row[index])} for index in nonzero]


def sparse_response(weights: Any) -> list[list[dict[str, Any]]]:
    """Shapes a SPLADE weight matrix into Trieve's outer-array-per-input JSON."""

    import numpy as np

    matrix = np.asarray(weights, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    return [splade_terms(row) for row in matrix]


def rerank_response(scores: Any) -> list[dict[str, Any]]:
    """Shapes cross-encoder scores into Trieve's [{index, score}] JSON.

    ``index`` is the position in the request ``texts``; every position appears
    exactly once. Non-finite scores are clamped to a finite floor because Trieve
    sorts with ``partial_cmp().unwrap()``, which panics on NaN.
    """

    import math

    result: list[dict[str, Any]] = []
    for index, score in enumerate(scores):
        value = float(score)
        if not math.isfinite(value):
            value = -1.0e30
        result.append({"index": index, "score": value})
    return result


def oidc_discovery_document(issuer: str, base_url: str) -> dict[str, Any]:
    """Returns a minimal OIDC discovery doc that openidconnect 3.x accepts at boot.

    The required (non-Option) CoreProviderMetadata keys are issuer,
    authorization_endpoint, jwks_uri, response_types_supported,
    subject_types_supported, id_token_signing_alg_values_supported. ``issuer`` must
    byte-match the requested issuer URL (trailing-slash sensitive), so it is passed
    through verbatim. token_endpoint/userinfo_endpoint are optional at boot but
    included so the later login flow has endpoints to call.
    """

    return {
        "issuer": issuer,
        "authorization_endpoint": f"{base_url}/authorize",
        "token_endpoint": f"{base_url}/token",
        "userinfo_endpoint": f"{base_url}/userinfo",
        "jwks_uri": f"{base_url}/jwks",
        "response_types_supported": ["code", "id_token", "token id_token"],
        "subject_types_supported": ["public"],
        "id_token_signing_alg_values_supported": ["RS256"],
        "scopes_supported": ["openid", "email", "profile"],
        "token_endpoint_auth_methods_supported": ["client_secret_basic", "client_secret_post"],
        "claims_supported": ["sub", "iss", "email", "name"],
    }


def normalize_input(value: Any) -> list[str]:
    """Coerces Trieve's ``input`` (bare string or list) into a list of strings."""

    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise ValueError("input must be a string or a list of strings")


# -- lazy model loading -----------------------------------------------------


class _Models:
    """Lazily loads and caches the three CUDA models on first request."""

    def __init__(self) -> None:
        self._dense: Any = None
        self._splade: dict[str, Any] = {}
        self._reranker: Any = None
        self._device = os.environ.get("MODEL_SERVER_DEVICE", "cuda")

    def dense(self) -> Any:
        if self._dense is None:
            from sentence_transformers import SentenceTransformer

            self._dense = SentenceTransformer(DENSE_MODEL, device=self._device)
        return self._dense

    def splade(self, encode_type: str) -> tuple[Any, Any]:
        model_name = SPLADE_QUERY_MODEL if encode_type == "query" else SPLADE_DOC_MODEL
        if model_name not in self._splade:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForMaskedLM.from_pretrained(model_name).to(self._device).eval()
            self._splade[model_name] = (tokenizer, model, torch)
        tokenizer, model, _torch = self._splade[model_name]
        return tokenizer, model

    def reranker(self) -> tuple[Any, Any]:
        if self._reranker is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL)
            model = (
                AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL)
                .to(self._device)
                .eval()
            )
            self._reranker = (tokenizer, model, torch)
        tokenizer, model, _torch = self._reranker
        return tokenizer, model


# -- model forward passes ---------------------------------------------------


def encode_dense(models: _Models, texts: list[str]) -> Any:
    """Encodes texts with the shared dense model (cosine-normalized fp32)."""

    model = models.dense()
    return model.encode(
        texts,
        batch_size=256,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def encode_sparse(models: _Models, texts: list[str], encode_type: str) -> Any:
    """Runs the SPLADE checkpoint and returns the (batch, vocab) weight matrix."""

    import numpy as np
    import torch

    tokenizer, model = models.splade(encode_type)
    device = next(model.parameters()).device
    rows: list[Any] = []
    for start in range(0, len(texts), 64):
        batch = texts[start : start + 64]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**encoded).logits
        weights = splade_pool(
            logits.detach().cpu().numpy(), encoded["attention_mask"].detach().cpu().numpy()
        )
        rows.append(weights)
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(rows, axis=0)


def score_rerank(models: _Models, query: str, texts: list[str]) -> list[float]:
    """Runs the cross-encoder over (query, text) pairs; returns raw relevance logits."""

    import torch

    if not texts:
        return []
    tokenizer, model = models.reranker()
    device = next(model.parameters()).device
    scores: list[float] = []
    for start in range(0, len(texts), 64):
        batch = texts[start : start + 64]
        pairs = [[query, text] for text in batch]
        encoded = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**encoded).logits
        scores.extend(logits.view(-1).detach().cpu().tolist())
    return scores


# -- FastAPI app ------------------------------------------------------------


def build_app() -> Any:
    """Builds the FastAPI app wiring the transforms to the Trieve contracts."""

    from fastapi import Body, FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI(title="trieve-bench-model-server")
    models = _Models()

    # Warm all four models before serving so the first real request does not pay the
    # download+load cost and blow Trieve's embedding-server timeout mid-benchmark.
    try:
        encode_dense(models, ["warmup"])
        encode_sparse(models, ["warmup"], "doc")
        encode_sparse(models, ["warmup"], "query")
        score_rerank(models, "warmup", ["warmup"])
        print("[model_server] models warmed (dense + splade doc/query + reranker)", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[model_server] warmup failed (will lazy-load): {exc}", flush=True)

    from fastapi.exceptions import RequestValidationError

    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(req: Any, exc: Any) -> Any:
        try:
            raw = (await req.body())[:600]
        except Exception:  # noqa: BLE001
            raw = b"<no body>"
        print(
            f"[model_server] 422 {req.method} {req.url.path}?{req.url.query} "
            f"errors={exc.errors()} body={raw!r}",
            flush=True,
        )
        return JSONResponse({"detail": exc.errors()}, status_code=422)

    @app.exception_handler(Exception)
    async def _on_unhandled(req: Any, exc: Any) -> Any:
        import traceback

        print(
            f"[model_server] 500 {req.url.path}: {exc}\n{traceback.format_exc()}",
            flush=True,
        )
        return JSONResponse({"error": str(exc)}, status_code=500)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/.well-known/openid-configuration")
    def openid_configuration() -> JSONResponse:
        base_url = OIDC_ISSUER_URL.rstrip("/")
        return JSONResponse(oidc_discovery_document(OIDC_ISSUER_URL, base_url))

    @app.get("/jwks")
    def jwks() -> dict[str, list[Any]]:
        # Empty key set is structurally valid; boot only needs discovery to parse.
        return {"keys": []}

    # NOTE: bind the JSON body via Body(...), NOT `request: Request`. The pinned
    # fastapi/starlette here does not special-case a `request: Request` param and
    # instead treats it as a required query field, 422-ing every POST pre-handler.
    # Sync `def` (not async): FastAPI runs these in a threadpool, so the blocking
    # torch/GPU forward passes never starve the event loop when Trieve fires the
    # dense + sparse + rerank calls for one hybrid query concurrently.
    @app.post("/embeddings")
    def embeddings(payload: dict = Body(...)) -> JSONResponse:  # noqa: B008 (FastAPI body binding)
        texts = normalize_input(payload.get("input"))
        vectors = encode_dense(models, texts)
        return JSONResponse(dense_response(vectors))

    @app.post("/embed_sparse")
    def embed_sparse(payload: dict = Body(...)) -> JSONResponse:  # noqa: B008 (FastAPI body binding)
        inputs = [str(item) for item in payload.get("inputs", [])]
        encode_type = str(payload.get("encode_type", "doc"))
        weights = encode_sparse(models, inputs, encode_type)
        return JSONResponse(sparse_response(weights))

    @app.post("/rerank")
    def rerank(payload: dict = Body(...)) -> JSONResponse:  # noqa: B008 (FastAPI body binding)
        query = str(payload.get("query", ""))
        texts = [str(item) for item in payload.get("texts", [])]
        scores = score_rerank(models, query, texts)
        return JSONResponse(rerank_response(scores))

    return app


def main() -> None:
    """Runs the model server on :7070 (all interfaces) under uvicorn."""

    import uvicorn

    uvicorn.run(build_app(), host="0.0.0.0", port=MODEL_SERVER_PORT, log_level="info")


if __name__ == "__main__":
    main()
