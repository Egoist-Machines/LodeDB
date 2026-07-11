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


CONFIGS: list[dict[str, Any]] = [
    {"label": "exact_bw4", "bit_width": 4},
    {"label": "ann1000_np16", "bit_width": 4, "ann_clusters": 1000, "ann_nprobe": 16},
    {
        "label": "exact_bw4_rescore_fp16",
        "bit_width": 4,
        "rescore": "fp16",
        "oversample": 4.0,
        "requires_engine": True,
    },
    {
        "label": "ann1000_np16_rescore_fp32_compact",
        "bit_width": 4,
        "ann_clusters": 1000,
        "ann_nprobe": 16,
        "rescore": "fp32",
        "oversample": 4.0,
        "compact": True,
        "requires_engine": True,
    },
]


def build_parser() -> argparse.ArgumentParser:
    """Builds the sweep command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--only", help="run one CONFIGS label")
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
    known = {str(config["label"]) for config in CONFIGS}
    if only is not None and only not in known:
        raise ValueError(f"unknown config {only!r}; choose one of {sorted(known)}")
    completed: list[str] = []
    skipped: list[dict[str, str]] = []
    for config in CONFIGS:
        label = str(config["label"])
        if only is not None and label != only:
            continue
        output = result_dir / f"{label}.json"
        if output.exists():
            print(f"[wiki-dpr] resume: keeping {output}", flush=True)
            completed.append(label)
            continue
        if config.get("requires_engine") and not include_unsupported:
            notice = "requires engine branch feat/cluster-layout-rescore"
            print(f"[wiki-dpr] skip {label}: {notice}", flush=True)
            skipped.append({"label": label, "reason": notice})
            continue
        kwargs = dict(config)
        kwargs.pop("label")
        kwargs.pop("requires_engine", None)
        try:
            result = run_benchmark(
                data_dir=data_dir,
                store_dir=stores_dir / label,
                label=label,
                loop_seconds=loop_seconds,
                loop_concurrency=loop_concurrency,
                ingest_batch=ingest_batch,
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
