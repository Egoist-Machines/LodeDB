"""Local CPU-only convenience wrapper for the vanilla-vs-augmented benchmark.

The measurement core lives beside this file (``turbovec_vva_bench`` +
``turbovec_vva_runner``) as dev-only scripts, not part of the shipped ``lodedb``
package. This wrapper just makes those siblings importable and forwards to the
runner's CLI:

    python benchmarks/gpu_vanilla_vs_augmented/run_bench.py --smoke

Equivalent to running the runner directly from this directory:

    python benchmarks/gpu_vanilla_vs_augmented/turbovec_vva_runner.py --smoke

``lodedb`` itself must be importable (install it, e.g. ``pip install -e .`` /
``uv sync``, or set ``PYTHONPATH`` to the repo's ``src``). On a CPU-only host the
augmented GPU series records ``{"skipped": ...}`` and the CPU axes still run.
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Best-effort: if lodedb is not installed, fall back to the repo's src/ layout
# (benchmarks/gpu_vanilla_vs_augmented/ -> repo root -> src).
_SRC = _THIS_DIR.parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from turbovec_vva_runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
