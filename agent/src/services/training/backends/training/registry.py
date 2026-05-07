"""Training backend registry and resolver."""

from __future__ import annotations

from collections.abc import Callable

from shared.src.config.registry_catalog_metadata import (
    RegistryCatalogEntry,
    dedupe_registry_catalog_entries,
)

from .base import SharedAdapterTrainingBackend, TrainingBackendFactory

_TRAINING_BACKEND_REGISTRY: dict[
    str,
    tuple[TrainingBackendFactory, RegistryCatalogEntry],
] = {}


def register_shared_adapter_training_backend(
    *backend_names: str,
    catalog_entry: RegistryCatalogEntry,
    factory: TrainingBackendFactory | None = None,
) -> (
    Callable[[TrainingBackendFactory], TrainingBackendFactory]
    | TrainingBackendFactory
):
    """training backend factory 옆에서 runtime wiring을 등록한다."""

    def _decorator(factory: TrainingBackendFactory) -> TrainingBackendFactory:
        registered_backend = (factory, catalog_entry)
        for backend_name in backend_names:
            _TRAINING_BACKEND_REGISTRY[backend_name.strip().lower()] = (
                registered_backend
            )
        return factory

    if factory is not None:
        return _decorator(factory)
    return _decorator


def build_shared_adapter_training_backend(
    backend_name: str,
    *,
    objective_config=None,
) -> SharedAdapterTrainingBackend:
    """backend 이름으로 로컬 학습 backend를 생성한다."""

    _ensure_builtin_shared_adapter_training_backends_loaded()
    normalized_name = backend_name.strip().lower()
    registered_backend = _TRAINING_BACKEND_REGISTRY.get(normalized_name)
    if registered_backend is not None:
        factory, _catalog_entry = registered_backend
        return factory(objective_config)
    raise ValueError(f"Unsupported local training backend: {backend_name}.")


def list_registered_shared_adapter_training_backend_names() -> tuple[str, ...]:
    """등록된 로컬 training backend 이름을 정렬된 tuple로 반환한다."""

    _ensure_builtin_shared_adapter_training_backends_loaded()
    return tuple(sorted(_TRAINING_BACKEND_REGISTRY))


def list_shared_adapter_training_backend_catalog_entries() -> tuple[
    RegistryCatalogEntry, ...
]:
    """등록된 training backend catalog entry를 canonical item 기준으로 반환한다."""

    _ensure_builtin_shared_adapter_training_backends_loaded()
    return dedupe_registry_catalog_entries(
        catalog_entry for _factory, catalog_entry in _TRAINING_BACKEND_REGISTRY.values()
    )


def _ensure_builtin_shared_adapter_training_backends_loaded() -> None:
    from agent.src.services.training.backends.training.builtin_loader import (
        load_builtin_shared_adapter_training_backends,
    )

    load_builtin_shared_adapter_training_backends()
