"""Method-owned shared adapter scoring backend registry."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable

from shared.src.contracts.registry_catalog_metadata import (
    RegistryCatalogEntry,
    dedupe_registry_catalog_entries,
)
from shared.src.contracts.training_contracts import TrainingObjectiveConfig

from .scoring_backend import (
    SharedAdapterScoringBackend,
    SharedAdapterScoringBackendFactory,
)

_ADAPTATION_PACKAGE = "methods.adaptation"
_SCORING_BACKEND_REGISTRY: dict[
    str,
    tuple[SharedAdapterScoringBackendFactory, RegistryCatalogEntry],
] = {}


def register_shared_adapter_scoring_backend(
    *backend_names: str,
    catalog_entry: RegistryCatalogEntry,
    factory: SharedAdapterScoringBackendFactory | None = None,
) -> (
    Callable[
        [SharedAdapterScoringBackendFactory],
        SharedAdapterScoringBackendFactory,
    ]
    | SharedAdapterScoringBackendFactory
):
    """scoring backend 구현 옆에서 method-owned wiring을 등록한다."""

    def _decorator(
        factory: SharedAdapterScoringBackendFactory,
    ) -> SharedAdapterScoringBackendFactory:
        registered_item = (factory, catalog_entry)
        for backend_name in backend_names:
            _SCORING_BACKEND_REGISTRY[backend_name.strip().lower()] = registered_item
        return factory

    if factory is not None:
        return _decorator(factory)
    return _decorator


def build_shared_adapter_scoring_backend(
    backend_name: str,
    *,
    objective_config: TrainingObjectiveConfig,
    similarity_name: str = "cosine",
) -> SharedAdapterScoringBackend:
    """backend 이름과 objective config로 method-owned scoring backend를 조립한다."""

    _import_adaptation_scoring_modules()
    normalized_name = backend_name.strip().lower()
    registered_item = _SCORING_BACKEND_REGISTRY.get(normalized_name)
    if registered_item is not None:
        factory, _catalog_entry = registered_item
        return factory(objective_config, similarity_name)
    raise ValueError(f"Unsupported shared adapter scoring backend: {backend_name}.")


def list_shared_adapter_scoring_backend_catalog_entries() -> tuple[
    RegistryCatalogEntry, ...
]:
    """등록된 method-owned scoring backend catalog entry 목록."""

    _import_adaptation_scoring_modules()
    return dedupe_registry_catalog_entries(
        catalog_entry for _factory, catalog_entry in _SCORING_BACKEND_REGISTRY.values()
    )


def _import_adaptation_scoring_modules() -> None:
    package = importlib.import_module(_ADAPTATION_PACKAGE)
    package_paths = getattr(package, "__path__", None)
    if package_paths is None:
        return

    for module_info in pkgutil.walk_packages(
        package_paths,
        prefix=f"{_ADAPTATION_PACKAGE}.",
    ):
        if module_info.ispkg or not module_info.name.endswith(".scoring"):
            continue
        module_name = module_info.name
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name != module_name:
                raise
