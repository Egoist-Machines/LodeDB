"""Self-test for the model server's math and response shapes (no Modal, no weights).

Validates the pure transforms against Trieve's wire contracts on synthetic tensors:

- dense: L2 normalization + DenseEmbedData shape with the required ``usage`` field
  and an ``input`` that is either a bare string or a list.
- SPLADE: log(1+ReLU(logits)) max-pool over live tokens (padding masked out),
  nonzero {index,value} extraction, outer-array-per-input shape.
- rerank: [{index, score}] covering every input position, NaN clamped to finite.
- OIDC: discovery doc carries the required non-Option CoreProviderMetadata keys and
  its issuer byte-matches the requested issuer URL.

Run:  python -m trieve_stack.self_test   (or  python self_test.py  from this dir)
"""

from __future__ import annotations

import math

import numpy as np

try:  # allow both `python -m trieve_stack.self_test` and `python self_test.py`
    from . import model_server as ms
except ImportError:  # pragma: no cover - direct-script fallback
    import model_server as ms


def test_l2_normalize_unit_rows() -> None:
    """Every normalized row has unit L2 norm; a zero row stays zero."""

    matrix = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    normalized = ms.l2_normalize_rows(matrix)
    norms = np.linalg.norm(normalized, axis=1)
    assert math.isclose(norms[0], 1.0, rel_tol=1e-6)
    assert norms[1] == 0.0  # zero row left unchanged, not divided by zero
    assert math.isclose(norms[2], 1.0, rel_tol=1e-6)


def test_dense_response_shape_and_usage() -> None:
    """dense_response emits data[].embedding + a required usage object, rows normalized."""

    vectors = np.array([[3.0, 4.0], [1.0, 1.0]], dtype=np.float32)
    payload = ms.dense_response(vectors)
    assert set(payload) == {"data", "usage"}
    assert set(payload["usage"]) == {"prompt_tokens", "total_tokens"}  # non-Option in Trieve
    assert len(payload["data"]) == 2
    first = payload["data"][0]["embedding"]
    assert math.isclose(math.hypot(*first), 1.0, rel_tol=1e-6)
    assert all(isinstance(value, float) for value in first)


def test_normalize_input_accepts_string_and_list() -> None:
    """Trieve sends input as a bare string (query) or a list (doc batch); both coerce."""

    assert ms.normalize_input("hello") == ["hello"]
    assert ms.normalize_input(["a", "b"]) == ["a", "b"]
    assert ms.normalize_input([1, 2]) == ["1", "2"]


def test_splade_pool_masks_padding_and_activates() -> None:
    """SPLADE pool = max over live tokens of log(1+ReLU); padding contributes nothing."""

    # batch=1, seq_len=2, vocab=3. Token 0 is live, token 1 is padding.
    logits = np.array([[[2.0, -5.0, 0.0], [100.0, 100.0, 100.0]]], dtype=np.float32)
    mask = np.array([[1.0, 0.0]], dtype=np.float32)
    weights = ms.splade_pool(logits, mask)
    expected0 = math.log1p(2.0)  # ReLU(2)=2 -> log(3)
    assert weights.shape == (1, 3)
    assert math.isclose(weights[0, 0], expected0, rel_tol=1e-6)
    assert weights[0, 1] == 0.0  # ReLU(-5)=0 -> log(1)=0, and padding masked anyway
    assert weights[0, 2] == 0.0  # ReLU(0)=0 -> log(1)=0
    # Padding token's huge logits must not leak through the max.
    assert weights.max() < math.log1p(100.0)


def test_splade_terms_only_nonzero() -> None:
    """Only nonzero vocab entries are emitted, as {index:int, value:float}."""

    row = np.array([0.0, 0.5, 0.0, 1.25], dtype=np.float32)
    terms = ms.splade_terms(row)
    assert terms == [
        {"index": 1, "value": float(np.float32(0.5))},
        {"index": 3, "value": float(np.float32(1.25))},
    ]
    assert all(isinstance(term["index"], int) for term in terms)
    assert all(isinstance(term["value"], float) for term in terms)


def test_sparse_response_outer_array_per_input() -> None:
    """sparse_response returns one inner list per input row (Trieve Vec<Vec<..>>)."""

    weights = np.array([[0.0, 2.0], [1.0, 0.0]], dtype=np.float32)
    payload = ms.sparse_response(weights)
    assert isinstance(payload, list) and len(payload) == 2
    assert payload[0] == [{"index": 1, "value": 2.0}]
    assert payload[1] == [{"index": 0, "value": 1.0}]


def test_rerank_response_indices_and_nan() -> None:
    """rerank_response covers every input position once; NaN is clamped finite."""

    scores = [0.3, float("nan"), -2.0]
    payload = ms.rerank_response(scores)
    assert [item["index"] for item in payload] == [0, 1, 2]  # position == request order
    assert math.isfinite(payload[1]["score"]) and payload[1]["score"] < 0  # NaN clamped
    assert payload[0]["score"] == 0.3


def test_oidc_document_required_keys_and_issuer_match() -> None:
    """Discovery doc has every required key and echoes the issuer byte-for-byte."""

    issuer = "http://localhost:7070"
    doc = ms.oidc_discovery_document(issuer, issuer)
    required = {
        "issuer",
        "authorization_endpoint",
        "jwks_uri",
        "response_types_supported",
        "subject_types_supported",
        "id_token_signing_alg_values_supported",
    }
    assert required <= set(doc)
    assert doc["issuer"] == issuer  # openidconnect validates exact issuer match
    assert doc["jwks_uri"].endswith("/jwks")


def test_real_forward_pass_shapes_if_transformers_available() -> None:
    """Optional: run tiny randomly-initialized models to confirm the pipeline shapes.

    Uses AutoModel* built from small configs (no pretrained download) to prove the
    SPLADE-pool and rerank-scoring plumbing produces the right shapes on real tensors.
    Skipped when torch/transformers are not importable.
    """

    try:
        import torch
        from transformers import (
            AutoModelForMaskedLM,
            AutoModelForSequenceClassification,
            BertConfig,
        )
    except Exception as exc:  # pragma: no cover - environment without torch
        print(f"[self-test] skipping real-forward-pass check: {exc}")
        return

    torch.manual_seed(0)
    vocab = 64
    config = BertConfig(
        vocab_size=vocab,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
    )

    # SPLADE path: MLM head -> (batch, seq, vocab) -> pool -> nonzero terms.
    mlm = AutoModelForMaskedLM.from_config(config).eval()
    batch, seq = 2, 5
    input_ids = torch.randint(0, vocab, (batch, seq))
    attention_mask = torch.ones(batch, seq, dtype=torch.long)
    attention_mask[0, 3:] = 0  # pad the tail of the first row
    with torch.no_grad():
        logits = mlm(input_ids=input_ids, attention_mask=attention_mask).logits
    assert logits.shape == (batch, seq, vocab)
    weights = ms.splade_pool(logits.numpy(), attention_mask.numpy())
    assert weights.shape == (batch, vocab)
    sparse = ms.sparse_response(weights)
    assert len(sparse) == batch

    # Rerank path: sequence-classification head -> per-pair logit -> {index,score}.
    seq_cfg = BertConfig(
        vocab_size=vocab,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
        num_labels=1,
    )
    reranker = AutoModelForSequenceClassification.from_config(seq_cfg).eval()
    pair_ids = torch.randint(0, vocab, (3, seq))
    with torch.no_grad():
        pair_logits = reranker(input_ids=pair_ids).logits.view(-1).tolist()
    reranked = ms.rerank_response(pair_logits)
    assert [item["index"] for item in reranked] == [0, 1, 2]
    print("[self-test] real forward-pass shape check passed (tiny random models)")


def main() -> int:
    """Runs every test function in this module and reports pass/fail."""

    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"[self-test] PASS {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"[self-test] FAIL {test.__name__}: {exc}")
    print(f"[self-test] {len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
