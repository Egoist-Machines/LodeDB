"""LodeDB engine core (Apache-2.0).

The in-process index / storage / embedding / route engine that the local LodeDB
SDK binds: the TurboVec CPU scan + ``.tvim``/``.tvd``/``.jsd`` persistence, the
embedding backends, and the direct-TurboVec route policies.

Submodules are imported directly (no eager aggregation), so importing the engine
loads only what each entry point needs. The optional CUDA GPU scan stays lazy;
``tests/test_import_boundary.py`` guards that import boundary.
"""
