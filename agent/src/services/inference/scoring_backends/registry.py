"""Scoring backend registry."""

from __future__ import annotations

from collections.abc import Callable

from agent.src.services.runtime_registry import RuntimeRegistry
from shared.src.contracts.registry_catalog_metadata import RegistryCatalogEntry
from shared.src.contracts.training_contracts import TrainingObjectiveConfig

from .base import ScoringBackend, ScoringBackendFactory

_SCORING_BACKEND_REGISTRY = RuntimeRegistry[ScoringBackendFactory](
    package_name="agent.src.services.inference.scoring_backends",
    item_kind="scoring backend",
)


def register_scoring_backend(
    *backend_names: str,
    catalog_entry: RegistryCatalogEntry,
    factory: ScoringBackendFactory | None = None,
) -> Callable[[ScoringBackendFactory], ScoringBackendFactory] | ScoringBackendFactory:
    """scoring backend factory 옆에서 runtime wiring을 등록한다."""

    return _SCORING_BACKEND_REGISTRY.register(
        *backend_names,
        catalog_entry=catalog_entry,
        factory=factory,
    )


def build_scoring_backend(
    backend_name: str,
    *,
    objective_config: TrainingObjectiveConfig,
    similarity_name: str = "cosine",
) -> ScoringBackend:
    """backend 이름과 objective config로 scoring backend를 조립한다."""

    factory, _catalog_entry = _SCORING_BACKEND_REGISTRY.get(backend_name)
    return factory(objective_config, similarity_name)


def list_registered_scoring_backend_names() -> tuple[str, ...]:
    """등록된 scoring backend 이름을 정렬된 tuple로 반환한다."""

    return _SCORING_BACKEND_REGISTRY.list_names()


def list_scoring_backend_catalog_entries() -> tuple[RegistryCatalogEntry, ...]:
    """등록된 scoring backend catalog entry를 canonical item 기준으로 반환한다."""

    return _SCORING_BACKEND_REGISTRY.list_catalog_entries()
