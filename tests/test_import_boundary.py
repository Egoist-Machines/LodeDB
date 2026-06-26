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

# LodeDB declares a lean runtime set (turbovec, numpy, typer, onnxruntime, transformers,
# sentence-transformers, pyyaml; extras [onnx-export]/[mcp]/[langchain]/[mem0]/[gpu]). None of
# the heavy deps below may load when the package is imported, and neither may the embedding
# runtimes themselves (onnxruntime/transformers/sentence-transformers are imported lazily, only
# when a backend is actually built). Top-level roots are checked (e.g. `sklearn`, not
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

# The optional framework adapters (langchain/llama-index/mem0) and the `lodedb migrate`
# source providers (psycopg/qdrant-client/chromadb/lancedb) are imported only inside the
# adapter or exporter that uses them. `import lodedb` reaches the `lodedb migrate` CLI
# sub-app (registered in lodedb.local.cli), so its source exporters must keep their provider
# imports function-local; none of these may load on a plain import.
_OPTIONAL_INTEGRATION_PROBE = """
import importlib, sys
for _m in ("lodedb", "lodedb.local.cli"):
    importlib.import_module(_m)
_FORBIDDEN = (
    "langchain", "langchain_core", "llama_index", "mem0",
    "psycopg", "psycopg2", "asyncpg", "qdrant_client", "chromadb", "lancedb",
)
_loaded = {_name.split(".", 1)[0] for _name in sys.modules}
for _root in _FORBIDDEN:
    if _root in _loaded:
        print("LOADED " + _root)
"""

# The optional image extra is Pillow (the "clip" preset / add_image path). The CLIP
# backend imports sentence-transformers and Pillow only inside the methods that
# encode, so importing LodeDB — including the embedding-backends, presets, and
# backends modules where the CLIP wiring lives — must not pull Pillow.
_IMAGE_EXTRA_LAZINESS_PROBE = """
import importlib, sys
for _m in (
    "lodedb",
    "lodedb.engine.embedding_backends",
    "lodedb.local.presets",
    "lodedb.local.backends",
):
    importlib.import_module(_m)
if "PIL" in {_name.split(".", 1)[0] for _name in sys.modules}:
    print("EAGER PIL")
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
    """Importing LodeDB must not import optional framework adapters or migrate providers."""

    leaked = _probe_lines(_OPTIONAL_INTEGRATION_PROBE, "LOADED")
    assert leaked == [], (
        f"optional integration/migration-source deps loaded on import (must stay lazy): {leaked}"
    )


def test_image_extra_stays_lazy_on_import():
    """Importing LodeDB must not eagerly load Pillow (the optional [image] extra)."""

    assert _probe_lines(_IMAGE_EXTRA_LAZINESS_PROBE, "EAGER") == []
