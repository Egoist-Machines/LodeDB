"""Import-boundary guards for the public LodeDB package.

Two invariants, each checked in a **fresh subprocess** so the result is
independent of whatever the parent pytest process has already imported
(``sys.modules`` is process-global, so an in-process check could be polluted by
an earlier test):

1. The optional CUDA GPU scan (``gpu_turbovec``) stays lazy, so a plain
   ``import lodedb`` never requires CuPy or a GPU.
2. LodeDB's lean runtime set holds: importing the package must not pull any
   heavy dependency (faiss / modal / mteb / datasets / matplotlib / sklearn),
   which would otherwise have to be declared and shipped.
"""

from __future__ import annotations

import subprocess
import sys

# Imports LodeDB in a clean interpreter and reports whether the optional CUDA scan
# module was loaded eagerly. It ships (opt-in ``[gpu]`` extra) but must stay lazy
# (imported only inside the methods that use it) so a plain import never needs CuPy.
_GPU_LAZINESS_PROBE = """
import importlib, sys
for _m in ("lodedb.local.db", "lodedb.local.backends"):
    importlib.import_module(_m)
if "lodedb.engine.gpu_turbovec" in sys.modules:
    print("EAGER lodedb.engine.gpu_turbovec")
"""

# LodeDB declares a lean runtime set (turbovec, numpy, typer, sentence-transformers,
# pyyaml; extras [mcp]/[langchain]/[mem0]/[gpu]). None of the heavy deps below may load when
# the package is imported. Top-level roots are checked (e.g. `sklearn`, not
# `sklearn.x`) because importing any submodule pulls the heavy root. `scikit-learn`
# may be *installed* (transitive via sentence-transformers); this asserts it is not
# *imported* by simply importing LodeDB.
_LEAN_BOUNDARY_PROBE = """
import importlib, sys
for _m in ("lodedb", "lodedb.local.db", "lodedb.local.backends", "lodedb.local.cli"):
    importlib.import_module(_m)
_FORBIDDEN = ("faiss", "modal", "mteb", "datasets", "matplotlib", "sklearn")
_loaded = {_name.split(".", 1)[0] for _name in sys.modules}
for _root in _FORBIDDEN:
    if _root in _loaded:
        print("LOADED " + _root)
"""

_OPTIONAL_INTEGRATION_PROBE = """
import importlib, sys
importlib.import_module("lodedb")
_FORBIDDEN = ("langchain", "langchain_core", "llama_index", "mem0")
_loaded = {_name.split(".", 1)[0] for _name in sys.modules}
for _root in _FORBIDDEN:
    if _root in _loaded:
        print("LOADED " + _root)
"""


def _probe_lines(probe: str, marker: str) -> list[str]:
    """Runs a probe in a fresh interpreter and returns the names it flagged."""

    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    return [
        line.split(" ", 1)[1]
        for line in result.stdout.splitlines()
        if line.startswith(marker + " ")
    ]


def test_gpu_scan_stays_lazy_on_import():
    """Importing LodeDB must not eagerly load the CUDA GPU scan module."""

    assert _probe_lines(_GPU_LAZINESS_PROBE, "EAGER") == []


def test_import_loads_no_heavy_dependency():
    """Importing LodeDB must not pull any heavy dependency outside the lean set."""

    leaked = _probe_lines(_LEAN_BOUNDARY_PROBE, "LOADED")
    assert leaked == [], (
        f"heavy dependencies loaded on import (must stay out of the lean package): {leaked}"
    )


def test_import_does_not_load_optional_integrations():
    """Importing LodeDB must not import optional framework integrations."""

    assert _probe_lines(_OPTIONAL_INTEGRATION_PROBE, "LOADED") == []
