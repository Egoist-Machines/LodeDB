"""Import-boundary guards for the public LodeDB package.

Three invariants, each checked in a **fresh subprocess** so the result is
independent of whatever the parent pytest process has already imported
(``sys.modules`` is process-global, so an in-process check could be polluted by
an earlier test):

1. The optional CUDA GPU scan (``gpu_turbovec``) stays lazy, so a plain
   ``import lodedb`` never requires CuPy or a GPU.
2. LodeDB's lean runtime set holds: importing the package must not pull any
   heavy dependency (faiss / modal / mteb / datasets / matplotlib / sklearn),
   which would otherwise have to be declared and shipped.
3. Built-in text embedding is opt-in (the ``[embeddings]`` / ``[torch]`` extras),
   so importing the package must not pull any embedding runtime
   (onnxruntime / transformers / sentence-transformers / torch) — a base
   ``pip install lodedb`` is a vector store and must import cleanly without them.
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

# LodeDB declares a lean base runtime set (turbovec, numpy, typer, pyyaml; the embedding
# runtimes are the opt-in [embeddings]/[torch] extras, and [onnx-export]/[mcp]/[langchain]/
# [mem0]/[gpu] add the rest). None of the heavy deps below may load when the package is
# imported. Top-level roots are checked (e.g. `sklearn`, not `sklearn.x`) because importing any
# submodule pulls the heavy root. `scikit-learn` may be *installed* (transitive via
# sentence-transformers when the [torch] extra is present); this asserts it is not *imported* by
# simply importing LodeDB.
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

# The optional framework adapters (langchain/llama-index/mem0/cognee) and the `lodedb migrate`
# source providers (psycopg/qdrant-client/chromadb/lancedb) are imported only inside the
# adapter or exporter that uses them. `import lodedb` reaches the `lodedb migrate` CLI
# sub-app (registered in lodedb.local.cli), so its source exporters must keep their provider
# imports function-local; none of these may load on a plain import.
_OPTIONAL_INTEGRATION_PROBE = """
import importlib, sys
for _m in ("lodedb", "lodedb.local.cli", "lodedb.cloud"):
    importlib.import_module(_m)
_FORBIDDEN = (
    "langchain", "langchain_core", "llama_index", "mem0", "cognee",
    "psycopg", "psycopg2", "asyncpg", "qdrant_client", "chromadb", "lancedb",
    # The [cloud] extra's dependencies (httpx; pynacl imports as `nacl`): the
    # first-party cloud client (lodedb.cloud) reaches them only through its
    # lazy PEP 562 exports and the CLI trampoline, so a plain import — even of
    # lodedb.cloud itself — must stay network-free.
    "httpx", "nacl",
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

# Built-in text embedding is opt-in (the [embeddings]/[torch] extras): the embedding runtimes
# must not load on a plain import, so a base `pip install lodedb` (a vector store with no
# embedding deps) imports cleanly. onnxruntime/transformers/sentence-transformers/torch are
# imported only inside the methods that build a preset/CLIP backend; the bring-your-own-vectors
# and embedder= paths never build one. The modules below are where that wiring lives.
_EMBEDDING_RUNTIME_LAZINESS_PROBE = """
import importlib, sys
for _m in (
    "lodedb",
    "lodedb.local.cli",
    "lodedb.local.db",
    "lodedb.local.backends",
    "lodedb.local.onnx_artifacts",
    "lodedb.local.presets",
    "lodedb.engine.embedding_backends",
):
    importlib.import_module(_m)
_FORBIDDEN = ("onnxruntime", "transformers", "sentence_transformers", "torch")
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
    """Importing LodeDB must not import optional framework adapters or migrate providers."""

    leaked = _probe_lines(_OPTIONAL_INTEGRATION_PROBE, "LOADED")
    assert leaked == [], (
        f"optional integration/migration-source deps loaded on import (must stay lazy): {leaked}"
    )


def test_image_extra_stays_lazy_on_import():
    """Importing LodeDB must not eagerly load Pillow (the optional [image] extra)."""

    assert _probe_lines(_IMAGE_EXTRA_LAZINESS_PROBE, "EAGER") == []


def test_import_does_not_load_embedding_runtime():
    """Importing LodeDB must not pull any embedding runtime (opt-in [embeddings]/[torch] extras).

    Guards the slim base install: a bring-your-own-vectors / ``embedder=`` user has none of
    onnxruntime/transformers/sentence-transformers/torch installed, and ``import lodedb`` must
    still succeed.
    """

    leaked = _probe_lines(_EMBEDDING_RUNTIME_LAZINESS_PROBE, "LOADED")
    assert leaked == [], (
        f"embedding runtimes loaded on import (must stay lazy / opt-in): {leaked}"
    )
