"""Run the reproducible wiki_dpr disk-rescore configuration matrix in process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .lodedb_bench import EngineFeatureRequired, run_benchmark
except ImportError:  # Direct execution from this directory.
    from lodedb_bench import EngineFeatureRequired, run_benchmark  # type: ignore[no-redef]


STORES: dict[str, dict[str, Any]] = {
    "exact_bw4": {"bit_width": 4},
    "exact_rs_fp16": {
        "rescore": "fp16",
        "oversample": 4.0,
        "requires_engine": True,
    },
    "exact_rs_fp32": {
        "rescore": "fp32",
        "oversample": 4.0,
        "requires_engine": True,
    },
    "ann1000": {"ann_clusters": 1000, "ann_nprobe": 16},
    # Reuses the store directory built before the cluster-contiguous layout
    # change (same create parameters as ann1000). Serving it isolates the
    # physical-layout effect: identical engine, clustering, and codes, but
    # insertion-ordered rows.
    "ann1000_np16": {"ann_clusters": 1000, "ann_nprobe": 16},
    "ann4096": {"ann_clusters": 4096, "ann_nprobe": 64},
    "ann4096_rs_fp16": {
        "ann_clusters": 4096,
        "ann_nprobe": 64,
        "rescore": "fp16",
        "oversample": 4.0,
        "requires_engine": True,
    },
}


SERVE_CONFIGS: list[dict[str, Any]] = [
    {"label": "exact_bw4", "store": "exact_bw4", "serve_overrides": {}},
    {
        "label": "exact_rs_fp16_ov2",
        "store": "exact_rs_fp16",
        "serve_overrides": {"oversample": 2.0},
        "requires_engine": True,
    },
    {
        "label": "exact_rs_fp16_ov4",
        "store": "exact_rs_fp16",
        "serve_overrides": {"oversample": 4.0},
        "requires_engine": True,
    },
    {
        "label": "exact_rs_fp16_ov8",
        "store": "exact_rs_fp16",
        "serve_overrides": {"oversample": 8.0},
        "requires_engine": True,
    },
    {
        "label": "exact_rs_fp32_ov4",
        "store": "exact_rs_fp32",
        "serve_overrides": {"oversample": 4.0},
        "requires_engine": True,
    },
    {
        "label": "ann1000_np8",
        "store": "ann1000",
        "serve_overrides": {"ann_nprobe": 8},
        "requires_engine": True,
    },
    {"label": "ann1000_np16", "store": "ann1000", "serve_overrides": {}},
    {
        "label": "ann1000_np32",
        "store": "ann1000",
        "serve_overrides": {"ann_nprobe": 32},
        "requires_engine": True,
    },
    {
        "label": "ann1000_np64",
        "store": "ann1000",
        "serve_overrides": {"ann_nprobe": 64},
        "requires_engine": True,
    },
    {
        "label": "ann1000_np128",
        "store": "ann1000",
        "serve_overrides": {"ann_nprobe": 128},
        "requires_engine": True,
    },
    {
        "label": "prechange_layout_np16",
        "store": "ann1000_np16",
        "serve_overrides": {},
    },
    {
        "label": "prechange_layout_np32",
        "store": "ann1000_np16",
        "serve_overrides": {"ann_nprobe": 32},
        "requires_engine": True,
    },
    {
        "label": "prechange_layout_np64",
        "store": "ann1000_np16",
        "serve_overrides": {"ann_nprobe": 64},
        "requires_engine": True,
    },
    {
        "label": "prechange_layout_np128",
        "store": "ann1000_np16",
        "serve_overrides": {"ann_nprobe": 128},
        "requires_engine": True,
    },
    {
        "label": "ann4096_np32",
        "store": "ann4096",
        "serve_overrides": {"ann_nprobe": 32},
        "requires_engine": True,
    },
    {"label": "ann4096_np64", "store": "ann4096", "serve_overrides": {}},
    {
        "label": "ann4096_np128",
        "store": "ann4096",
        "serve_overrides": {"ann_nprobe": 128},
        "requires_engine": True,
    },
    {
        "label": "ann4096_np256",
        "store": "ann4096",
        "serve_overrides": {"ann_nprobe": 256},
        "requires_engine": True,
    },
    {
        "label": "ann4096_rs_fp16_np64_ov4",
        "store": "ann4096_rs_fp16",
        "serve_overrides": {"ann_nprobe": 64, "oversample": 4.0},
        "requires_engine": True,
    },
    {
        "label": "ann4096_rs_fp16_np128_ov4",
        "store": "ann4096_rs_fp16",
        "serve_overrides": {"ann_nprobe": 128, "oversample": 4.0},
        "requires_engine": True,
    },
    {
        "label": "ann4096_rs_fp16_np256_ov4",
        "store": "ann4096_rs_fp16",
        "serve_overrides": {"ann_nprobe": 256, "oversample": 4.0},
        "requires_engine": True,
    },
    {
        "label": "ann4096_rs_fp16_np128_ov2",
        "store": "ann4096_rs_fp16",
        "serve_overrides": {"ann_nprobe": 128, "oversample": 2.0},
        "requires_engine": True,
    },
]

_STORE_CONFIG_NAME = "benchmark_store_config.json"
_ENGINE_NOTICE = "requires engine branch feat/cluster-layout-rescore"


def build_parser() -> argparse.ArgumentParser:
    """Builds the sweep command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--only", help="run one SERVE_CONFIGS label")
    parser.add_argument("--include-unsupported", action="store_true")
    parser.add_argument("--loop-seconds", type=float, default=20.0)
    parser.add_argument("--loop-concurrency", type=int, default=4)
    parser.add_argument("--ingest-batch", type=int, default=8192)
    return parser


def run_sweep(
    *,
    data_dir: str | Path,
    work_dir: str | Path,
    only: str | None = None,
    include_unsupported: bool = False,
    loop_seconds: float = 20.0,
    loop_concurrency: int = 4,
    ingest_batch: int = 8192,
) -> dict[str, Any]:
    """Runs missing configurations and returns the compact sweep summary."""

    work_dir = Path(work_dir)
    result_dir = work_dir / "results"
    stores_dir = work_dir / "stores"
    result_dir.mkdir(parents=True, exist_ok=True)
    stores_dir.mkdir(parents=True, exist_ok=True)
    known = {str(config["label"]) for config in SERVE_CONFIGS}
    if only is not None and only not in known:
        raise ValueError(f"unknown config {only!r}; choose one of {sorted(known)}")
    completed: list[str] = []
    skipped: list[dict[str, str]] = []
    selected = [
        config for config in SERVE_CONFIGS if only is None or config["label"] == only
    ]
    referenced_stores = list(dict.fromkeys(str(config["store"]) for config in selected))
    unavailable_stores: dict[str, str] = {}
    for store_label in referenced_stores:
        store = STORES[store_label]
        store_dir = stores_dir / store_label
        config_path = store_dir / _STORE_CONFIG_NAME
        if config_path.exists():
            print(f"[wiki-dpr] resume: keeping {config_path}", flush=True)
            continue
        if store.get("requires_engine") and not include_unsupported:
            unavailable_stores[store_label] = _ENGINE_NOTICE
            continue
        kwargs = dict(store)
        kwargs.pop("requires_engine", None)
        try:
            run_benchmark(
                data_dir=data_dir,
                store_dir=store_dir,
                label=store_label,
                loop_seconds=loop_seconds,
                loop_concurrency=loop_concurrency,
                ingest_batch=ingest_batch,
                build=True,
                serve=False,
                **kwargs,
            )
        except EngineFeatureRequired as exc:
            if not include_unsupported:
                raise
            unavailable_stores[store_label] = str(exc)
    for config in selected:
        label = str(config["label"])
        output = result_dir / f"{label}.json"
        if output.exists():
            print(f"[wiki-dpr] resume: keeping {output}", flush=True)
            completed.append(label)
            continue
        store_label = str(config["store"])
        if store_label in unavailable_stores:
            notice = unavailable_stores[store_label]
            print(f"[wiki-dpr] skip {label}: {notice}", flush=True)
            skipped.append({"label": label, "reason": notice})
            continue
        if config.get("requires_engine") and not include_unsupported:
            print(f"[wiki-dpr] skip {label}: {_ENGINE_NOTICE}", flush=True)
            skipped.append({"label": label, "reason": _ENGINE_NOTICE})
            continue
        kwargs = dict(STORES[store_label])
        kwargs.pop("requires_engine", None)
        overrides = dict(config["serve_overrides"])
        try:
            result = run_benchmark(
                data_dir=data_dir,
                store_dir=stores_dir / store_label,
                label=label,
                loop_seconds=loop_seconds,
                loop_concurrency=loop_concurrency,
                ingest_batch=ingest_batch,
                build=False,
                serve=True,
                serve_nprobe=overrides.get("ann_nprobe"),
                serve_oversample=overrides.get("oversample"),
                **kwargs,
            )
        except EngineFeatureRequired as exc:
            if not include_unsupported:
                raise
            print(f"[wiki-dpr] skip {label}: {exc}", flush=True)
            skipped.append({"label": label, "reason": str(exc)})
            continue
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        completed.append(label)
    summary = {"completed": completed, "skipped": skipped, "work": str(work_dir)}
    summary_path = result_dir / "sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: list[str] | None = None) -> None:
    """Runs the sweep CLI."""

    args = build_parser().parse_args(argv)
    run_sweep(
        data_dir=args.data,
        work_dir=args.work,
        only=args.only,
        include_unsupported=args.include_unsupported,
        loop_seconds=args.loop_seconds,
        loop_concurrency=args.loop_concurrency,
        ingest_batch=args.ingest_batch,
    )


if __name__ == "__main__":
    main()
