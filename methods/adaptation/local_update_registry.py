"""Methods-owned local update backend registry."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable

from shared.src.contracts.registry_catalog_metadata import (
    RegistryCatalogEntry,
    dedupe_registry_catalog_entries,
)

from .local_update_backend import (
    SharedAdapterTrainingBackend,
    TrainingBackendFactory,
)

_ADAPTATION_PACKAGE = "methods.adaptation"
_TRAINING_BACKEND_MODULE_OVERRIDES = {
    "peft_classifier_trainer": (
        "methods.adaptation.text_classifier.peft_encoder.training_backend"
    ),
}
_LOCAL_UPDATE_BACKEND_REGISTRY: dict[
    str,
    tuple[TrainingBackendFactory, RegistryCatalogEntry],
] = {}


def register_shared_adapter_training_backend(
    *backend_names: str,
    catalog_entry: RegistryCatalogEntry,
    factory: TrainingBackendFactory | None = None,
) -> (
    Callable[[TrainingBackendFactory], TrainingBackendFactory] | TrainingBackendFactory
):
    """local update backend 구현 옆에서 factory/catalog를 등록한다."""

    def _decorator(factory: TrainingBackendFactory) -> TrainingBackendFactory:
        registered_backend = (factory, catalog_entry)
        for backend_name in backend_names:
            _LOCAL_UPDATE_BACKEND_REGISTRY[_normalize_backend_name(backend_name)] = (
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
    """backend 이름으로 local update backend를 생성한다."""

    normalized_name = _normalize_backend_name(backend_name)
    registered_backend = _LOCAL_UPDATE_BACKEND_REGISTRY.get(normalized_name)
    if registered_backend is None:
        _import_training_backend_module_for_name(normalized_name)
        registered_backend = _LOCAL_UPDATE_BACKEND_REGISTRY.get(normalized_name)
    if registered_backend is None:
        _import_all_training_backend_modules()
        registered_backend = _LOCAL_UPDATE_BACKEND_REGISTRY.get(normalized_name)
    if registered_backend is None:
        raise ValueError(f"Unsupported local update backend: {backend_name}.")
    factory, _catalog_entry = registered_backend
    return factory(objective_config)


def list_registered_shared_adapter_training_backend_names() -> tuple[str, ...]:
    """등록된 local update backend 이름을 정렬된 tuple로 반환한다."""

    _import_all_training_backend_modules()
    return tuple(sorted(_LOCAL_UPDATE_BACKEND_REGISTRY))


def list_shared_adapter_training_backend_catalog_entries() -> tuple[
    RegistryCatalogEntry, ...
]:
    """등록된 local update backend catalog entry를 canonical item 기준으로 반환한다."""

    _import_all_training_backend_modules()
    return dedupe_registry_catalog_entries(
        catalog_entry
        for _factory, catalog_entry in _LOCAL_UPDATE_BACKEND_REGISTRY.values()
    )


def _import_training_backend_module_for_name(normalized_name: str) -> None:
    override_module = _TRAINING_BACKEND_MODULE_OVERRIDES.get(normalized_name)
    if override_module is not None:
        _try_import_module(override_module)
        return
    parts = normalized_name.replace("-", "_").split("_")
    for end_index in range(len(parts), 0, -1):
        package_name = "_".join(parts[:end_index])
        if _try_import_module(f"{_ADAPTATION_PACKAGE}.{package_name}.training_backend"):
            return


def _import_all_training_backend_modules() -> None:
    package = importlib.import_module(_ADAPTATION_PACKAGE)
    package_paths = getattr(package, "__path__", None)
    if package_paths is None:
        return

    for module_info in pkgutil.iter_modules(package_paths):
        if not module_info.ispkg:
            continue
        _try_import_module(f"{_ADAPTATION_PACKAGE}.{module_info.name}.training_backend")


def _try_import_module(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name or module_name.startswith(f"{error.name}."):
            return False
        raise
    return True


def _normalize_backend_name(backend_name: str) -> str:
    normalized_name = backend_name.strip().lower()
    if not normalized_name:
        raise ValueError("local update backend name must not be empty.")
    return normalized_name
