"""Scoring backend registry."""

from __future__ import annotations

from collections.abc import Callable

from agent.src.services.runtime_registry_imports import (
    import_runtime_module_for_name,
    import_runtime_package_modules,
)
from shared.src.config.registry_catalog_metadata import (
    RegistryCatalogEntry,
    dedupe_registry_catalog_entries,
)
from shared.src.contracts.training_contracts import TrainingObjectiveConfig

from .base import ScoringBackend, ScoringBackendFactory

_SCORING_BACKEND_REGISTRY: dict[
    str,
    tuple[ScoringBackendFactory, RegistryCatalogEntry],
] = {}


def register_scoring_backend(
    *backend_names: str,
    catalog_entry: RegistryCatalogEntry,
    factory: ScoringBackendFactory | None = None,
) -> Callable[[ScoringBackendFactory], ScoringBackendFactory] | ScoringBackendFactory:
    """scoring backend factory 옆에서 runtime wiring을 등록한다."""

    def _decorator(factory: ScoringBackendFactory) -> ScoringBackendFactory:
        registered_backend = (factory, catalog_entry)
        for backend_name in backend_names:
            _SCORING_BACKEND_REGISTRY[backend_name.strip().lower()] = (
                registered_backend
            )
        return factory

    if factory is not None:
        return _decorator(factory)
    return _decorator


def build_scoring_backend(
    backend_name: str,
    *,
    objective_config: TrainingObjectiveConfig,
    similarity_name: str = "cosine",
) -> ScoringBackend:
    """backend 이름과 objective config로 scoring backend를 조립한다."""

    normalized_name = backend_name.strip().lower()
    import_runtime_module_for_name(
        package_name="agent.src.services.inference.scoring_backends",
        registered_name=normalized_name,
    )
    registered_backend = _SCORING_BACKEND_REGISTRY.get(normalized_name)
    if registered_backend is not None:
        factory, _catalog_entry = registered_backend
        return factory(objective_config, similarity_name)
    raise ValueError(f"Unsupported scoring backend: {backend_name}.")


def list_registered_scoring_backend_names() -> tuple[str, ...]:
    """등록된 scoring backend 이름을 정렬된 tuple로 반환한다."""

    import_runtime_package_modules(
        package_name="agent.src.services.inference.scoring_backends"
    )
    return tuple(sorted(_SCORING_BACKEND_REGISTRY))


def list_scoring_backend_catalog_entries() -> tuple[RegistryCatalogEntry, ...]:
    """등록된 scoring backend catalog entry를 canonical item 기준으로 반환한다."""

    import_runtime_package_modules(
        package_name="agent.src.services.inference.scoring_backends"
    )
    return dedupe_registry_catalog_entries(
        catalog_entry for _factory, catalog_entry in _SCORING_BACKEND_REGISTRY.values()
    )
