#!/usr/bin/env python3
"""MPS exact scan vs. TurboVec NEON scan, a local Apple-Silicon benchmark.

Compares the same vendored TurboVec index two ways (no Modal, no CUDA):

- **NEON**:       TurboVec's native CPU SIMD scan (``index.search(queries, k)``),
                  the default on Mac.
- **MPS exact**:  the opt-in
                  :class:`lodedb.engine.mps_turbovec.MpsDirectTurboVecSession`:
                  dequantized fp16 rows resident on the Apple GPU, scored with a
                  batched matmul + ``torch.topk``.

It reports search throughput (q/s) across batch sizes and recall (R@1-within-top-k
vs exact fp32 ground truth) for both. On the M1 we measured, NEON wins across
batch sizes; the open question is whether a much stronger Apple GPU (M-series
Pro/Max, M5+) moves the crossover, so run it on your own hardware.

    python benchmarks/mps_vs_neon/run.py --n 100000 --dim 384 --queries 1000

The NEON scan uses all cores (rayon); pin RAYON_NUM_THREADS to vary that.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

K_RECALL = (1, 8, 64)


def _unit(n: int, dim: int, *, seed: int) -> np.ndarray:
    """Returns n unit-norm Gaussian fp32 vectors (the data-oblivious regime)."""

    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim), dtype=np.float32)
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    return np.ascontiguousarray(v)


def _exact_top1(vecs: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Exact fp32 top-1 stable id (1-based) per query, tiled to bound memory."""

    q = queries.shape[0]
    best = np.full(q, -np.inf, np.float32)
    ids = np.zeros(q, np.uint64)
    tile = max(1, (1 << 26) // max(1, q))
    for s in range(0, vecs.shape[0], tile):
        block = vecs[s : s + tile]
        scores = queries @ block.T
        local = scores.argmax(axis=1)
        local_val = scores[np.arange(q), local]
        better = local_val > best
        best = np.where(better, local_val, best)
        ids = np.where(better, (s + local + 1).astype(np.uint64), ids)
    return ids


def _recall_curve(found: np.ndarray, truth_top1: np.ndarray) -> dict[str, float]:
    """R@1-within-top-k: fraction of queries whose true top-1 is in the top-k."""

    return {
        str(k): float((found[:, :k] == truth_top1[:, None]).any(axis=1).mean())
        for k in K_RECALL
        if found.shape[1] >= k
    }


def _median_qps(fn, batch: int, repeats: int) -> float:
    """Returns median throughput (queries/sec) over `repeats` warmed passes."""

    fn()  # warm up
    durations = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - t0)
    median_s = float(np.median(durations))
    return batch / median_s if median_s > 0 else 0.0


def main() -> None:
    """Runs the NEON-vs-MPS speed + recall comparison and writes a results JSON."""

    parser = argparse.ArgumentParser(description="MPS exact scan vs TurboVec NEON scan")
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--bit-width", type=int, default=4)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--batches", default="1,16,64,256,1024")
    parser.add_argument("--repeats", type=int, default=3)
    here = Path(__file__).resolve().parent
    parser.add_argument("--out", default=str(here / "results" / "mps_vs_neon_m1.json"))
    args = parser.parse_args()

    import turbovec

    from lodedb.engine.mps_turbovec import MpsDirectTurboVecSession, mps_exact_scan_available

    mps_ok, mps_reason = mps_exact_scan_available()
    print(f"[mps-vs-neon] mps_available={mps_ok} {mps_reason}", file=sys.stderr)

    vecs = _unit(args.n, args.dim, seed=0)
    qpool = _unit(args.queries, args.dim, seed=1)
    index = turbovec.IdMapIndex(dim=args.dim, bit_width=args.bit_width)
    index.add_with_ids(vecs, np.arange(1, args.n + 1, dtype=np.uint64))
    session = MpsDirectTurboVecSession.build(index=index) if mps_ok else None

    speed = []
    for batch in (int(b) for b in args.batches.split(",") if b):
        qb = np.ascontiguousarray(qpool[np.arange(batch) % args.queries])
        neon_qps = _median_qps(lambda q=qb: index.search(q, args.k), batch, args.repeats)
        mps_qps = (
            _median_qps(lambda q=qb: session.search_batch(q, top_k=args.k), batch, args.repeats)
            if session is not None
            else 0.0
        )
        ratio = round(mps_qps / neon_qps, 3) if neon_qps else 0.0
        speed.append(
            {"batch": batch, "neon_qps": round(neon_qps, 1), "mps_qps": round(mps_qps, 1),
             "mps_over_neon": ratio}
        )
        print(f"[mps-vs-neon] batch={batch:5d} NEON={neon_qps:9.0f} q/s "
              f"MPS={mps_qps:9.0f} q/s ({ratio}x)", file=sys.stderr)

    truth = _exact_top1(vecs, qpool)
    _scores, neon_ids = index.search(qpool, args.k)
    recall = {"neon": _recall_curve(neon_ids, truth)}
    if session is not None:
        mps_ids = session.search_batch(qpool, top_k=args.k).stable_ids
        recall["mps_exact"] = _recall_curve(mps_ids, truth)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "machine": {"platform": platform.platform(), "machine": platform.machine()},
                "mps_available": mps_ok,
                "mps_reason": mps_reason,
                "config": {
                    "n": args.n, "dim": args.dim, "bit_width": args.bit_width,
                    "queries": args.queries, "k": args.k, "repeats": args.repeats,
                },
                "build_ms": round(session.upload_build_ms, 2) if session else None,
                "resident_bytes": session.resident_bytes if session else None,
                "speed": speed,
                "recall": recall,
            },
            indent=2,
        )
    )
    print(f"[mps-vs-neon] wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
