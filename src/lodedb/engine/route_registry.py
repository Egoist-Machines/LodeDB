"""Route registry loading and fallback decisions for fixed cascade support."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lodedb.config import load_yaml_file

COMPRESSED_INT8_SUPPORTED = "compressed_int8_supported"
FIXED_FULL_SUPPORTED_INT8_FAILED = "fixed_full_supported_int8_failed"
REQUIRES_HIGHER_K_DIAGNOSTIC = "requires_higher_k_diagnostic"
FULL_DENSE_OR_FULL_VECTOR_FALLBACK = "full_dense_or_full_vector_fallback"
EXCLUDED_CANDIDATE_RECALL_BOTTLENECK = "excluded_candidate_recall_bottleneck"
SKIPPED_LOADER = "skipped_loader"
SKIPPED_BUDGET = "skipped_budget"

SUPPORTED_ROUTE_CLASSES = {
    COMPRESSED_INT8_SUPPORTED,
    FIXED_FULL_SUPPORTED_INT8_FAILED,
}
FALLBACK_ROUTE_CLASSES = {
    FULL_DENSE_OR_FULL_VECTOR_FALLBACK,
    EXCLUDED_CANDIDATE_RECALL_BOTTLENECK,
    REQUIRES_HIGHER_K_DIAGNOSTIC,
    SKIPPED_LOADER,
    SKIPPED_BUDGET,
}
KNOWN_ROUTE_CLASSES = SUPPORTED_ROUTE_CLASSES | FALLBACK_ROUTE_CLASSES


@dataclass(frozen=True)
class RouteRegistryEntry:
    """Stores one configured support decision for a model/task/provider pair."""

    model: str
    task: str
    route_classification: str
    provider: str = "any"
    route_decision: str = ""
    method_template: str = ""
    fallback: str = "full_dense_or_full_vector"
    reason: str = ""
    evidence: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> RouteRegistryEntry:
        """Builds one route entry from a YAML mapping and validates its class."""

        route_classification = str(payload["route_classification"])
        if route_classification not in KNOWN_ROUTE_CLASSES:
            raise ValueError(f"Unknown route classification: {route_classification!r}")
        return cls(
            model=str(payload["model"]),
            task=str(payload["task"]),
            provider=str(payload.get("provider", "any")),
            route_classification=route_classification,
            route_decision=str(payload.get("route_decision", "")),
            method_template=str(payload.get("method_template", "")),
            fallback=str(payload.get("fallback", "full_dense_or_full_vector")),
            reason=str(payload.get("reason", "")),
            evidence=dict(payload.get("evidence", {}) or {}),
        )

    def to_dict(self) -> dict[str, object]:
        """Serializes this registry entry for reports and audit artifacts."""

        return {
            "model": self.model,
            "provider": self.provider,
            "task": self.task,
            "route_classification": self.route_classification,
            "route_decision": self.route_decision,
            "method_template": self.method_template,
            "fallback": self.fallback,
            "reason": self.reason,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class DatasetRouteStatus:
    """Stores task-level support status when a dataset has no validated pair route."""

    task: str
    status: str
    reason: str
    segment: str = "unsupported"
    loader_supported: bool = False

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> DatasetRouteStatus:
        """Builds one task-level status from a YAML mapping."""

        status = str(payload["status"])
        if status not in KNOWN_ROUTE_CLASSES:
            raise ValueError(f"Unknown dataset route status: {status!r}")
        return cls(
            task=str(payload["task"]),
            segment=str(payload.get("segment", "unsupported")),
            status=status,
            reason=str(payload.get("reason", "")),
            loader_supported=bool(payload.get("loader_supported", False)),
        )

    def to_dict(self) -> dict[str, object]:
        """Serializes the dataset-level route status for reports."""

        return {
            "task": self.task,
            "segment": self.segment,
            "status": self.status,
            "loader_supported": self.loader_supported,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RouteDecision:
    """Stores the concrete route selected for one model/task lookup."""

    model: str
    task: str
    provider: str | None
    route_classification: str
    route_decision: str
    method_template: str
    fallback: str
    reason: str
    registry_hit: bool
    evidence: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serializes a route decision for JSON, CSV, or Markdown reports."""

        return {
            "model": self.model,
            "provider": self.provider,
            "task": self.task,
            "route_classification": self.route_classification,
            "route_decision": self.route_decision,
            "method_template": self.method_template,
            "fallback": self.fallback,
            "reason": self.reason,
            "registry_hit": self.registry_hit,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class RouteRegistry:
    """Indexes validated route decisions and task-level fallback statuses."""

    name: str
    default_route_classification: str
    default_route_decision: str
    default_fallback: str
    entries: tuple[RouteRegistryEntry, ...]
    dataset_statuses: tuple[DatasetRouteStatus, ...] = ()

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> RouteRegistry:
        """Builds a route registry from a loaded YAML mapping."""

        registry = payload.get("route_registry")
        if not isinstance(registry, dict):
            raise ValueError("route registry config must contain a 'route_registry' mapping")
        defaults = dict(registry.get("defaults", {}) or {})
        entries = tuple(
            RouteRegistryEntry.from_mapping(dict(item))
            for item in registry.get("entries", ())
            if isinstance(item, dict)
        )
        dataset_statuses = tuple(
            DatasetRouteStatus.from_mapping(dict(item))
            for item in registry.get("dataset_statuses", ())
            if isinstance(item, dict)
        )
        return cls(
            name=str(registry.get("name", "unnamed_route_registry")),
            default_route_classification=str(
                defaults.get("route_classification", FULL_DENSE_OR_FULL_VECTOR_FALLBACK)
            ),
            default_route_decision=str(
                defaults.get("route_decision", "use_full_dense_or_full_vector_fallback")
            ),
            default_fallback=str(defaults.get("fallback", "full_dense_or_full_vector")),
            entries=entries,
            dataset_statuses=dataset_statuses,
        )

    def select_route(
        self,
        *,
        model: str,
        task: str,
        provider: str | None = None,
        fixed_full_passed: bool | None = None,
        compressed_int8_passed: bool | None = None,
        requires_higher_k: bool = False,
        failure_mode: str | None = None,
        drifted: bool = False,
        failed: bool = False,
        high_risk: bool = False,
    ) -> RouteDecision:
        """Returns the product route after applying runtime fallback overrides."""

        if drifted:
            return self._runtime_fallback(model, task, provider, "drift_gate_requires_fallback")
        if failed:
            return self._runtime_fallback(
                model, task, provider, "runtime_failure_requires_fallback"
            )
        if high_risk:
            return self._runtime_fallback(model, task, provider, "high_risk_pair_requires_fallback")
        if requires_higher_k:
            return self._fallback_decision(
                model=model,
                task=task,
                provider=provider,
                route_classification=REQUIRES_HIGHER_K_DIAGNOSTIC,
                route_decision="diagnostic_higher_k_only_keep_default_unsupported",
                reason="higher K is diagnostic only and is not a default product route",
                registry_hit=True,
            )

        entry = self.find_entry(model=model, task=task, provider=provider)
        if entry is None:
            dataset_status = self.dataset_status_for(task)
            if dataset_status is not None:
                return self._fallback_decision(
                    model=model,
                    task=task,
                    provider=provider,
                    route_classification=dataset_status.status,
                    route_decision=self._route_decision_for_status(dataset_status.status),
                    reason=dataset_status.reason,
                    registry_hit=True,
                )
            return self._fallback_decision(
                model=model,
                task=task,
                provider=provider,
                route_classification=self.default_route_classification,
                route_decision=self.default_route_decision,
                reason="unknown_model_task_pair",
                registry_hit=False,
            )

        if fixed_full_passed is False:
            if failure_mode == "candidate_recall":
                return self._fallback_decision(
                    model=model,
                    task=task,
                    provider=provider,
                    route_classification=EXCLUDED_CANDIDATE_RECALL_BOTTLENECK,
                    route_decision="exclude_pair_and_use_full_dense_or_fallback",
                    reason="candidate_recall_bottleneck",
                    registry_hit=True,
                    evidence=entry.evidence,
                )
            return self._runtime_fallback(model, task, provider, "fixed_full_rerank_failed")
        if fixed_full_passed is True and compressed_int8_passed is False:
            return self._fallback_decision(
                model=model,
                task=task,
                provider=provider,
                route_classification=FIXED_FULL_SUPPORTED_INT8_FAILED,
                route_decision="use_fixed_full_rerank_with_full_vector_fallback",
                reason="compressed_int8_failed_or_missing",
                registry_hit=True,
                evidence=entry.evidence,
            )
        return RouteDecision(
            model=model,
            provider=provider or entry.provider,
            task=task,
            route_classification=entry.route_classification,
            route_decision=entry.route_decision,
            method_template=entry.method_template,
            fallback=entry.fallback,
            reason=entry.reason,
            registry_hit=True,
            evidence=entry.evidence,
        )

    def find_entry(
        self,
        *,
        model: str,
        task: str,
        provider: str | None = None,
    ) -> RouteRegistryEntry | None:
        """Finds the best matching explicit model/task route entry."""

        provider_key = provider or "any"
        for entry in self.entries:
            if entry.model == model and entry.task == task and entry.provider == provider_key:
                return entry
        if provider is None:
            for entry in self.entries:
                if entry.model == model and entry.task == task:
                    return entry
        for entry in self.entries:
            if entry.model == model and entry.task == task and entry.provider == "any":
                return entry
        return None

    def dataset_status_for(self, task: str) -> DatasetRouteStatus | None:
        """Returns the configured task-level status for a dataset, if any."""

        for status in self.dataset_statuses:
            if status.task == task:
                return status
        return None

    def to_dict(self) -> dict[str, object]:
        """Serializes the whole registry for route-decision audit artifacts."""

        return {
            "name": self.name,
            "defaults": {
                "route_classification": self.default_route_classification,
                "route_decision": self.default_route_decision,
                "fallback": self.default_fallback,
            },
            "entries": [entry.to_dict() for entry in self.entries],
            "dataset_statuses": [status.to_dict() for status in self.dataset_statuses],
        }

    def _runtime_fallback(
        self,
        model: str,
        task: str,
        provider: str | None,
        reason: str,
    ) -> RouteDecision:
        """Builds a full dense/full-vector fallback decision for runtime risk."""

        return self._fallback_decision(
            model=model,
            task=task,
            provider=provider,
            route_classification=FULL_DENSE_OR_FULL_VECTOR_FALLBACK,
            route_decision="use_full_dense_or_full_vector_fallback",
            reason=reason,
            registry_hit=self.find_entry(model=model, task=task, provider=provider) is not None,
        )

    def _fallback_decision(
        self,
        *,
        model: str,
        task: str,
        provider: str | None,
        route_classification: str,
        route_decision: str,
        reason: str,
        registry_hit: bool,
        evidence: dict[str, object] | None = None,
    ) -> RouteDecision:
        """Builds a fallback-style route decision with the registry defaults."""

        return RouteDecision(
            model=model,
            provider=provider,
            task=task,
            route_classification=route_classification,
            route_decision=route_decision,
            method_template="",
            fallback=self.default_fallback,
            reason=reason,
            registry_hit=registry_hit,
            evidence=evidence or {},
        )

    @staticmethod
    def _route_decision_for_status(status: str) -> str:
        """Maps a task-level status into the corresponding product route decision."""

        if status == SKIPPED_LOADER:
            return "skip_until_loader_validated"
        if status == SKIPPED_BUDGET:
            return "skip_until_budget_authorizes_validation"
        if status == EXCLUDED_CANDIDATE_RECALL_BOTTLENECK:
            return "exclude_pair_and_use_full_dense_or_fallback"
        return "use_full_dense_or_full_vector_fallback"


def default_route_registry() -> RouteRegistry:
    """Returns the route registry the local LodeDB profile uses.

    The local DB serves with the direct-TurboVec route policy, whose
    ``index_backend == "turbovec_direct"`` skips the registry-classification
    check (see :meth:`EngineRoutePolicy.validate_index_request`). So a registry
    with no explicit entries — every lookup resolves to the full-vector default —
    is all the local path needs, and nothing has to be loaded from disk.
    """

    return RouteRegistry(
        name="lodedb_local_route_registry",
        default_route_classification=FULL_DENSE_OR_FULL_VECTOR_FALLBACK,
        default_route_decision="use_full_dense_or_full_vector_fallback",
        default_fallback="full_dense_or_full_vector",
        entries=(),
        dataset_statuses=(),
    )


def load_route_registry(path: str | Path) -> RouteRegistry:
    """Loads a route registry from a YAML file (for advanced custom registries)."""

    return RouteRegistry.from_mapping(load_yaml_file(path))


def route_decisions_to_dicts(decisions: tuple[RouteDecision, ...]) -> list[dict[str, object]]:
    """Serializes route decisions for stable report output."""

    return [decision.to_dict() for decision in decisions]
