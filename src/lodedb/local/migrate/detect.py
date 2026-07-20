"""Project inspection: detect the framework / vector provider and the install plan.

The migrator's first step is to look at a project checkout and decide, without
running it, which migration path applies. Detection reads three kinds of signal:

- **Dependency files**: ``pyproject.toml`` / ``requirements*.txt`` / ``setup.py`` /
  ``package.json`` tell us which package manager to use and which frameworks /
  providers are even installed.
- **Config and infra files**: Docker Compose, ``settings.yaml``, ``.env.example``,
  Alembic migrations, and raw SQL hint at a provider (a ``qdrant`` service, a
  ``CREATE EXTENSION vector``) without any Python import.
- **Python source (AST)**: imported names and a few constructor/call patterns tell
  us which framework *owns* the vector store and which direct provider is wired.

The routing rule is the one both issues require: **framework detection wins over
direct-provider detection.** If LangChain / LlamaIndex / mem0 owns the store, we
route there (issue #34) even when the framework is backed by pgvector or Qdrant. A
direct provider (issue #35) is reported only when no framework owns the store. If
the signals are ambiguous (two live frameworks, or a framework and an unrelated
direct provider with no clear owner), detection stops and asks the caller to
disambiguate with ``--framework`` / ``--provider``.

Detection never executes project code; it parses files. Reports stay payload-free:
they carry detected names, counts of matching files, and the chosen install
command, never file contents.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Frameworks (issue #34). Order is the tie-break preference when, e.g., a project
# imports both langchain and llama-index but only one actually owns a store.
FRAMEWORKS = ("langchain", "llama-index", "mem0")
# Direct providers (issue #35).
DIRECT_PROVIDERS = ("pgvector", "qdrant", "chroma", "lancedb", "sqlite-vec", "faiss")

# Import-name fragments that signal each framework.
_FRAMEWORK_IMPORTS = {
    "langchain": ("langchain", "langchain_core", "langchain_community", "langchain_postgres"),
    "llama-index": ("llama_index",),
    "mem0": ("mem0",),
}
# Source substrings that signal each framework (covers call sites, not just imports).
_FRAMEWORK_SOURCE_SIGNALS = {
    "langchain": (
        "InMemoryVectorStore",
        "VectorStore",
        "QdrantVectorStore",
        "PGVector",
        "as_retriever",
    ),
    "llama-index": (
        "StorageContext",
        "VectorStoreIndex",
        "SimpleVectorStore",
        "persist_dir",
    ),
    "mem0": ("Memory.from_config", "MemoryConfig", "from mem0", "import mem0"),
}
# Import-name fragments + source substrings that signal each direct provider.
_PROVIDER_IMPORTS = {
    "pgvector": ("pgvector", "psycopg", "psycopg2", "asyncpg"),
    "qdrant": ("qdrant_client",),
    "chroma": ("chromadb",),
    "lancedb": ("lancedb",),
    "sqlite-vec": ("sqlite_vec",),
    "faiss": ("faiss",),
}
_PROVIDER_SOURCE_SIGNALS = {
    "pgvector": ("CREATE EXTENSION vector", "vector(", "VECTOR(", "embedding"),
    "qdrant": ("QdrantClient", "qdrant"),
    "chroma": ("chromadb", "PersistentClient", "persist_directory"),
    "lancedb": ("lancedb.connect", "LanceDB"),
    "sqlite-vec": ("vec0", "sqlite_vec"),
    "faiss": ("faiss.read_index", "IndexFlat", "save_local", "load_local"),
}

# The install extra for each framework / direct provider.
_FRAMEWORK_EXTRA = {
    "langchain": "lodedb[langchain]",
    "llama-index": "lodedb[llama-index]",
    "mem0": "lodedb[mem0]",
}

_PY_GLOB = "**/*.py"
_MAX_SOURCE_FILES = 2000
_SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    "site-packages",
}


@dataclass
class Detection:
    """The result of inspecting a project (payload-free)."""

    route: str  # "framework" | "provider" | "ambiguous" | "none"
    framework: str | None = None
    provider: str | None = None
    package_manager: str = "pip"
    install_command: str | None = None
    install_extra: str | None = None
    frameworks_seen: list[str] = field(default_factory=list)
    providers_seen: list[str] = field(default_factory=list)
    source_path: str | None = None
    collection: str | None = None
    table: str | None = None
    warnings: list[str] = field(default_factory=list)
    next_step: str | None = None
    signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Renders the detection as a payload-free JSON object."""

        out: dict[str, Any] = {
            "route": self.route,
            "framework": self.framework,
            "provider": self.provider,
            "package_manager": self.package_manager,
            "install_command": self.install_command,
            "install_extra": self.install_extra,
            "frameworks_seen": list(self.frameworks_seen),
            "providers_seen": list(self.providers_seen),
            "source_path": self.source_path,
            "collection": self.collection,
            "table": self.table,
            "warnings": list(self.warnings),
            "signals": dict(self.signals),
        }
        if self.next_step:
            out["next"] = self.next_step
        return out


def detect_package_manager(project: Path) -> tuple[str, str]:
    """Returns ``(manager, base_install_command)`` for a project checkout.

    Prefers the lockfile/manifest already present so the migration uses the
    project's existing toolchain rather than introducing a new one.
    """

    if (project / "uv.lock").is_file() or (project / "pyproject.toml").is_file():
        if (project / "uv.lock").is_file():
            return "uv", "uv add"
    if (project / "poetry.lock").is_file():
        return "poetry", "poetry add"
    if (project / "pyproject.toml").is_file():
        # A pyproject without a uv/poetry lock: pip against the project is the safe default.
        return "pip", "pip install"
    if (project / "requirements.txt").is_file():
        return "pip", "pip install"
    return "pip", "pip install"


def _iter_source_files(project: Path) -> list[Path]:
    """Returns project Python files, skipping vendored/build/venv directories."""

    files: list[Path] = []
    for path in project.glob(_PY_GLOB):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
        if len(files) >= _MAX_SOURCE_FILES:
            break
    return files


def _read_dependency_blobs(project: Path) -> str:
    """Returns the concatenated text of dependency/config files (lowercased search blob)."""

    candidates = [
        "pyproject.toml",
        "requirements.txt",
        "requirements-dev.txt",
        "setup.py",
        "setup.cfg",
        "Pipfile",
        "package.json",
        "settings.yaml",
        "settings.yml",
        ".env.example",
        "docker-compose.yml",
        "docker-compose.yaml",
        "compose.yml",
        "compose.yaml",
    ]
    blob: list[str] = []
    for name in candidates:
        path = project / name
        if path.is_file():
            try:
                blob.append(path.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                continue
    # Alembic migrations and raw SQL are strong pgvector signals.
    for sql in list(project.glob("**/*.sql"))[:50]:
        if any(part in _SKIP_DIRS for part in sql.parts):
            continue
        try:
            blob.append(sql.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    return "\n".join(blob)


def _imported_modules(tree: ast.AST) -> set[str]:
    """Returns the set of top-level imported module roots in a parsed source file."""

    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
            modules.add(node.module)
    return modules


def _scan_sources(files: list[Path]) -> tuple[dict[str, int], dict[str, int], dict[str, Any]]:
    """Scans source files for framework/provider signals.

    Returns ``(framework_hits, provider_hits, evidence)`` where the hit maps count
    how many files matched each name (imports weighted over plain text), and
    ``evidence`` records cheap extracted hints (persist paths, table/collection
    names) without storing file contents.
    """

    framework_hits: dict[str, int] = dict.fromkeys(FRAMEWORKS, 0)
    provider_hits: dict[str, int] = dict.fromkeys(DIRECT_PROVIDERS, 0)
    evidence: dict[str, Any] = {}

    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        modules: set[str] = set()
        try:
            modules = _imported_modules(ast.parse(text))
        except SyntaxError:
            modules = set()

        for framework, import_names in _FRAMEWORK_IMPORTS.items():
            if modules.intersection(import_names):
                framework_hits[framework] += 2
            elif any(sig in text for sig in _FRAMEWORK_SOURCE_SIGNALS[framework]):
                framework_hits[framework] += 1

        for provider, import_names in _PROVIDER_IMPORTS.items():
            if modules.intersection(import_names):
                provider_hits[provider] += 2
            elif any(sig in text for sig in _PROVIDER_SOURCE_SIGNALS[provider]):
                provider_hits[provider] += 1

    return framework_hits, provider_hits, evidence


def _blob_hits(blob: str) -> tuple[set[str], set[str]]:
    """Returns the frameworks/providers named in the dependency/config blob."""

    lowered = blob.lower()
    frameworks = {
        fw
        for fw, names in _FRAMEWORK_IMPORTS.items()
        if any(name.lower() in lowered for name in names)
    }
    if "mem0ai" in lowered:
        frameworks.add("mem0")
    providers: set[str] = set()
    for provider, names in _PROVIDER_IMPORTS.items():
        if any(name.lower() in lowered for name in names):
            providers.add(provider)
    for provider, signals in _PROVIDER_SOURCE_SIGNALS.items():
        if any(sig.lower() in lowered for sig in signals):
            providers.add(provider)
    return frameworks, providers


def inspect_project(
    project: str | Path,
    *,
    framework: str | None = None,
    provider: str | None = None,
) -> Detection:
    """Inspects a project and returns a routing :class:`Detection`.

    ``framework`` / ``provider`` are explicit overrides that pin the route and skip
    disambiguation (the values still validated against the known sets). Detection
    is read-only and payload-free.
    """

    root = Path(project)
    manager, base_cmd = detect_package_manager(root)

    files = _iter_source_files(root)
    framework_hits, provider_hits, evidence = _scan_sources(files)
    dep_frameworks, dep_providers = _blob_hits(_read_dependency_blobs(root))

    frameworks_seen = sorted(
        {fw for fw, n in framework_hits.items() if n > 0} | dep_frameworks,
        key=lambda fw: (-framework_hits.get(fw, 0), FRAMEWORKS.index(fw)),
    )
    providers_seen = sorted(
        {p for p, n in provider_hits.items() if n > 0} | dep_providers,
        key=lambda p: (-provider_hits.get(p, 0), DIRECT_PROVIDERS.index(p)),
    )

    detection = Detection(
        route="none",
        package_manager=manager,
        frameworks_seen=frameworks_seen,
        providers_seen=providers_seen,
        signals={
            "framework_file_hits": {k: v for k, v in framework_hits.items() if v},
            "provider_file_hits": {k: v for k, v in provider_hits.items() if v},
            "source_files_scanned": len(files),
            **evidence,
        },
    )

    # Explicit framework override wins outright (this is the #35 -> #34 handoff a
    # caller can force, and the disambiguation answer for the #34 path).
    if framework is not None:
        framework = _normalize_framework(framework)
        return _as_framework_route(detection, framework, base_cmd)

    if provider is not None and provider != "auto":
        provider = _normalize_provider(provider)
        return _as_provider_route(detection, provider, base_cmd)

    # Routing rule: a framework owner wins over any direct provider beneath it.
    if len(frameworks_seen) == 1:
        return _as_framework_route(detection, frameworks_seen[0], base_cmd)
    if len(frameworks_seen) > 1:
        detection.route = "ambiguous"
        detection.warnings.append(
            "multiple frameworks detected ("
            + ", ".join(frameworks_seen)
            + "); re-run with --framework to choose one"
        )
        detection.next_step = "pass --framework langchain|llama-index|mem0"
        return detection

    # No framework owner: fall to direct-provider detection (issue #35).
    if len(providers_seen) == 1:
        return _as_provider_route(detection, providers_seen[0], base_cmd)
    if len(providers_seen) > 1:
        detection.route = "ambiguous"
        detection.warnings.append(
            "multiple direct providers detected ("
            + ", ".join(providers_seen)
            + "); re-run with --provider to choose one"
        )
        detection.next_step = "pass --provider pgvector|qdrant|chroma|lancedb|sqlite-vec|faiss"
        return detection

    detection.route = "none"
    detection.warnings.append(
        "no LangChain/LlamaIndex/mem0 framework and no direct vector provider detected; "
        "pass --framework or --provider to migrate explicitly"
    )
    return detection


def _quote_spec(spec: str) -> str:
    """Quotes an install spec for the shell when it carries an extras bracket.

    ``lodedb[langchain]`` is quoted (the brackets are shell globs), while a bare
    ``lodedb`` is left unquoted, matching how the docs spell the commands.
    """

    return f"'{spec}'" if "[" in spec else spec


def _as_framework_route(detection: Detection, framework: str, base_cmd: str) -> Detection:
    """Fills a :class:`Detection` for the framework (issue #34) route."""

    detection.route = "framework"
    detection.framework = framework
    detection.install_extra = _FRAMEWORK_EXTRA[framework]
    detection.install_command = f"{base_cmd} {_quote_spec(_FRAMEWORK_EXTRA[framework])}"
    detection.next_step = "use the framework migration toolkit tracked in #34"
    return detection


def _as_provider_route(detection: Detection, provider: str, base_cmd: str) -> Detection:
    """Fills a :class:`Detection` for the direct-provider (issue #35) route."""

    detection.route = "provider"
    detection.provider = provider
    detection.install_extra = "lodedb"
    detection.install_command = f"{base_cmd} lodedb"
    if provider != "pgvector":
        detection.warnings.append(
            f"direct {provider} export is not implemented yet; pgvector is the supported direct "
            "provider in this release (other providers are tracked as follow-ups)"
        )
        detection.next_step = f"direct {provider} migration is a follow-up; pgvector is supported"
    else:
        detection.next_step = "use the direct pgvector migration path"
    return detection


def _normalize_framework(value: str) -> str:
    """Validates/normalizes a ``--framework`` value to the canonical name."""

    canonical = value.strip().lower().replace("_", "-")
    aliases = {"llamaindex": "llama-index", "llama_index": "llama-index"}
    canonical = aliases.get(canonical, canonical)
    if canonical not in FRAMEWORKS:
        raise ValueError(
            f"unknown framework {value!r}; choose one of: {', '.join(FRAMEWORKS)}"
        )
    return canonical


def _normalize_provider(value: str) -> str:
    """Validates/normalizes a ``--provider`` value to the canonical name."""

    canonical = value.strip().lower().replace("_", "-")
    aliases = {"sqlite_vec": "sqlite-vec", "sqlitevec": "sqlite-vec", "pg": "pgvector"}
    canonical = aliases.get(canonical, canonical)
    if canonical not in DIRECT_PROVIDERS:
        raise ValueError(
            f"unknown provider {value!r}; choose one of: {', '.join(DIRECT_PROVIDERS)}"
        )
    return canonical
