"""GovReport at scale: vanilla vs augmented TurboVec recall + speed, 100K -> 1M.

Embeds GovReport chunks (MiniLM, cosine) up to ~1M vectors on the GPU, then runs the
vanilla-vs-augmented cells across a corpus-size sweep — the scale + correctness evidence a
launch review flagged as missing for the CPU scan. Both paths read the SAME 4-bit index:

- **recall@k vs fp32 brute force** for the vanilla **uint8-LUT** scan and the augmented
  **fp16-reconstruction** scan, at each corpus size — does the scan still find the true
  nearest neighbour at 1M?
- **the CPU scan's practical ceiling** — vanilla single-thread and all-threads throughput
  (fresh ``RAYON_NUM_THREADS``-pinned subprocesses) vs the augmented GPU throughput, per
  corpus size and a batch sweep at the top size.

This is a dev-only benchmark (not part of the shipped ``lodedb`` package). It reuses the
vanilla-vs-augmented cell runner from the sibling ``gpu_vanilla_vs_augmented`` benchmark
(only the data source is new). GPU-only — the augmented path needs CUDA, and 1M-vector
brute-force ground truth is not laptop-scale.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# The shared vanilla-vs-augmented cell runner lives in the sibling gpu benchmark dir
# (dev-only sibling scripts, not part of the shipped package); make it importable.
_VVA_DIR = str(Path(__file__).resolve().parent.parent / "gpu_vanilla_vs_augmented")
if _VVA_DIR not in sys.path:
    sys.path.insert(0, _VVA_DIR)

from turbovec_vva_bench import machine_info, run_cell  # noqa: E402
from turbovec_vva_runner import _subprocess_cell  # noqa: E402

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GOVREPORT_DATASET = "ccdv/govreport-summarization"


def _load_govreport_texts(
    *, max_corpus: int, n_query: int, chunk_character_limit: int
) -> tuple[list[str], list[str]]:
    """Streams GovReport and chunks each report into corpus + query (summary) texts.

    Reuses LodeDB's own ``chunk_text``; streams train/validation/test until the corpus
    reaches ``max_corpus`` chunks (chunk size sets the vector count, not scan correctness).
    """

    from datasets import load_dataset

    from lodedb.engine.core import chunk_text

    corpus_texts: list[str] = []
    query_texts: list[str] = []
    for split in ("train", "validation", "test"):
        if len(corpus_texts) >= max_corpus:
            break
        rows = load_dataset(GOVREPORT_DATASET, split=split, streaming=True)
        for row in rows:
            if len(corpus_texts) >= max_corpus:
                break
            report = str(row.get("report", "")).strip()
            summary = str(row.get("summary", "")).strip()
            if not report or not summary:
                continue
            chunks = chunk_text(report, chunk_character_limit)
            if not chunks:
                continue
            for body in chunks:
                if len(corpus_texts) >= max_corpus:
                    break
                corpus_texts.append(body)
            if len(query_texts) < n_query:
                query_texts.append(summary)
    return corpus_texts, query_texts


def _encode_blocks(
    model: Any, texts: list[str], *, batch_size: int, block: int, label: str
) -> NDArray[np.float32]:
    """Encodes texts in blocks (cosine-normalized fp32), printing progress."""

    if not texts:
        return np.zeros((0, int(model.get_sentence_embedding_dimension())), dtype=np.float32)
    parts: list[NDArray[np.float32]] = []
    for start in range(0, len(texts), block):
        emb = model.encode(
            texts[start : start + block],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        parts.append(np.asarray(emb, dtype=np.float32))
        print(
            f"[govreport-scale] embedded {label} {min(start + block, len(texts))}/{len(texts)}",
            flush=True,
        )
    return np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float32)


def embed_govreport(
    *,
    max_corpus: int,
    n_query: int,
    out_dir: str,
    chunk_character_limit: int = 480,
    model_name: str = DEFAULT_EMBED_MODEL,
    device: str = "cuda",
    batch_size: int = 512,
) -> dict[str, Any]:
    """Loads + chunks GovReport, embeds chunks + query summaries, saves .npy.

    Returns the dataset spec the vva cell runner consumes (name, dim, vectors_path,
    queries_path) plus realized counts and embed time.
    """

    from sentence_transformers import SentenceTransformer

    print(
        f"[govreport-scale] loading GovReport (max_corpus={max_corpus}, queries={n_query}, "
        f"chunk_chars={chunk_character_limit})",
        flush=True,
    )
    corpus_texts, query_texts = _load_govreport_texts(
        max_corpus=max_corpus, n_query=n_query, chunk_character_limit=chunk_character_limit
    )
    print(
        f"[govreport-scale] {len(corpus_texts)} chunks, {len(query_texts)} queries; "
        f"embedding with {model_name} on {device}",
        flush=True,
    )

    model = SentenceTransformer(model_name, device=device)
    started = time.perf_counter()
    corpus = _encode_blocks(
        model, corpus_texts, batch_size=batch_size, block=250_000, label="corpus"
    )
    queries = _encode_blocks(
        model, query_texts, batch_size=batch_size, block=250_000, label="query"
    )
    embed_seconds = time.perf_counter() - started

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    vec_path = str(Path(out_dir) / "govreport-minilm-corpus.npy")
    qry_path = str(Path(out_dir) / "govreport-minilm-queries.npy")
    np.save(vec_path, corpus)
    np.save(qry_path, queries)
    print(
        f"[govreport-scale] saved corpus={corpus.shape} queries={queries.shape} "
        f"in {embed_seconds:.0f}s",
        flush=True,
    )
    return {
        "name": "govreport-minilm",
        "dataset": GOVREPORT_DATASET,
        "chunk_character_limit": chunk_character_limit,
        "model": model_name,
        "dim": int(corpus.shape[1]),
        "vectors_path": vec_path,
        "queries_path": qry_path,
        "corpus_count": int(corpus.shape[0]),
        "query_count": int(queries.shape[0]),
        "embed_seconds": round(embed_seconds, 1),
    }


def run_govreport_scale(spec: dict[str, Any]) -> dict[str, Any]:
    """Runs recall + speed (vanilla ST/MT vs augmented GPU) over a corpus-size sweep."""

    dataset = spec["dataset"]
    dim = int(dataset["dim"])
    sizes = [int(n) for n in spec["sizes"]]
    bits = [int(b) for b in spec["bit_widths"]]
    queries = int(spec["queries"])
    k = int(spec["k"])
    repeats = int(spec["repeats"])
    batch_size = int(spec["batch_size"])
    include_gpu = bool(spec.get("include_gpu", True))
    ncpu = int(os.cpu_count() or 1)
    started = time.perf_counter()

    out: dict[str, Any] = {
        "machine": machine_info(),
        "spec": {key: value for key, value in spec.items() if key != "dataset"},
        "dataset": dataset,
        "recall": [],
        "speed": [],
        "batch": [],
    }

    def log(message: str) -> None:
        print(f"[govreport-scale] {message}", flush=True)

    paths = {
        "vectors_path": dataset["vectors_path"],
        "queries_path": dataset["queries_path"],
    }

    for n in sizes:
        for bit in bits:
            base = {
                "dim": dim, "bit_width": bit, "n": n, "queries": queries, "k": k,
                "seed": 0, "batch_size": batch_size, **paths,
            }
            log(f"recall n={n} bit={bit}")
            recall = run_cell({**base, "axis": "recall", "include_gpu": include_gpu})
            out["recall"].append({
                "n": n, "bit_width": bit,
                "vanilla": recall.get("recall_vanilla"),
                "augmented": recall.get("recall_augmented"),
            })

            log(f"speed n={n} bit={bit} (vanilla ST/MT + augmented)")
            speed_base = {**base, "axis": "speed", "repeats": repeats}
            st = _subprocess_cell(
                {**speed_base, "which": "vanilla", "include_gpu": False}, threads=1
            )
            mt = _subprocess_cell(
                {**speed_base, "which": "vanilla", "include_gpu": False}, threads=ncpu
            )
            augmented = (
                run_cell({**speed_base, "which": "augmented", "include_gpu": True})
                if include_gpu
                else {}
            )
            out["speed"].append({
                "n": n, "bit_width": bit,
                "vanilla_st": st.get("speed_vanilla"),
                "vanilla_mt": mt.get("speed_vanilla"),
                "augmented": augmented.get("speed_augmented"),
            })

    batch_sweep_n = int(spec.get("batch_sweep_n", 0))
    batch_sweep_bit = int(spec.get("batch_sweep_bit", 4))
    if batch_sweep_n and include_gpu:
        sweep_base = {
            "axis": "speed", "dim": dim, "bit_width": batch_sweep_bit, "n": batch_sweep_n,
            "queries": queries, "k": k, "repeats": repeats, "seed": 0, **paths,
        }
        reference = _subprocess_cell(
            {**sweep_base, "which": "vanilla", "include_gpu": False, "batch_size": 64},
            threads=ncpu,
        )
        for batch in spec.get("batch_sweep_sizes", [1, 16, 64, 256, 1024]):
            log(f"batch sweep bs={batch} n={batch_sweep_n}")
            augmented = run_cell(
                {**sweep_base, "which": "augmented", "include_gpu": True, "batch_size": batch}
            )
            out["batch"].append({
                "batch_size": batch, "n": batch_sweep_n, "bit_width": batch_sweep_bit,
                "augmented": augmented.get("speed_augmented"),
                "vanilla_mt": reference.get("speed_vanilla"),
            })

    out["wall_seconds"] = time.perf_counter() - started
    return out


def govreport_scale_spec() -> dict[str, Any]:
    """Full run: GovReport at 100K/500K/1M, 4-bit, with a batch sweep at 1M."""

    return {
        # ~480-char chunks so streamed GovReport reports yield ~1.1M chunks (capped at
        # max_corpus); chunk size only sets the vector count, not scan correctness.
        "chunk_character_limit": 480,
        "max_corpus": 1_000_000,
        "sizes": [100_000, 500_000, 1_000_000],
        "bit_widths": [4],
        "queries": 1000,
        "k": 64,
        "repeats": 3,
        "batch_size": 64,
        "include_gpu": True,
        "batch_sweep_n": 1_000_000,
        "batch_sweep_bit": 4,
        "batch_sweep_sizes": [1, 16, 64, 256, 1024],
    }


def govreport_scale_smoke_spec() -> dict[str, Any]:
    """Tiny Modal smoke: embed ~40K chunks, run two small sizes to validate the pipeline."""

    return {
        "chunk_character_limit": 480,
        "max_corpus": 40_000,
        "sizes": [10_000, 40_000],
        "bit_widths": [4],
        "queries": 200,
        "k": 64,
        "repeats": 2,
        "batch_size": 64,
        "include_gpu": True,
        "batch_sweep_n": 40_000,
        "batch_sweep_bit": 4,
        "batch_sweep_sizes": [1, 64, 256],
    }
