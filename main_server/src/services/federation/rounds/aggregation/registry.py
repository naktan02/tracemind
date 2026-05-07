"""Aggregation backend registry."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Mapping

from main_server.src.services.federation.rounds.aggregation.models import (
    AggregationBackendFactory,
    SharedAdapterAggregationBackend,
)
from shared.src.config.registry_catalog_metadata import (
    RegistryCatalogEntry,
    dedupe_registry_catalog_entries,
)

from .diagonal_scale_defaults import AggregationConfigScalar

_AGGREGATION_PACKAGE = "main_server.src.services.federation.rounds.aggregation"
_SKIPPED_AGGREGATION_MODULES = frozenset(
    {
        "builtin_loader",
        "diagonal_scale_defaults",
        "models",
        "registry",
        "runtime_adapter",
    }
)

_AGGREGATION_BACKEND_REGISTRY: dict[
    tuple[str, str],
    tuple[AggregationBackendFactory, RegistryCatalogEntry],
] = {}


def register_shared_adapter_aggregation_backend(
    adapter_kind: str,
    *backend_names: str,
    catalog_entry: RegistryCatalogEntry,
    factory: AggregationBackendFactory | None = None,
) -> (
    Callable[[AggregationBackendFactory], AggregationBackendFactory]
    | AggregationBackendFactory
):
    """aggregation backend factory 옆에서 runtime wiring을 등록한다."""

    def _decorator(factory: AggregationBackendFactory) -> AggregationBackendFactory:
        normalized_adapter_kind = adapter_kind.strip().lower()
        registered_backend = (factory, catalog_entry)
        for backend_name in backend_names:
            normalized_backend_name = backend_name.strip().lower()
            _AGGREGATION_BACKEND_REGISTRY[
                (normalized_adapter_kind, normalized_backend_name)
            ] = registered_backend
        return factory

    if factory is not None:
        return _decorator(factory)
    return _decorator


def build_shared_adapter_aggregation_backend(
    *,
    adapter_kind: str,
    backend_name: str,
    overrides: Mapping[str, AggregationConfigScalar] | None = None,
) -> SharedAdapterAggregationBackend:
    """adapter family와 backend 이름으로 aggregation backend를 조립한다."""

    normalized_key = (adapter_kind.strip().lower(), backend_name.strip().lower())
    _import_aggregation_module_for_adapter_kind(normalized_key[0])
    registered_backend = _AGGREGATION_BACKEND_REGISTRY.get(normalized_key)
    if registered_backend is not None:
        factory, _catalog_entry = registered_backend
        return factory(overrides)
    raise ValueError(
        "Unsupported aggregation backend for adapter family: "
        f"adapter_kind={adapter_kind}, backend_name={backend_name}"
    )


def list_registered_shared_adapter_aggregation_backends(
    *,
    adapter_kind: str | None = None,
) -> tuple[tuple[str, str], ...]:
    """등록된 aggregation backend 키를 정렬된 tuple로 반환한다."""

    _import_aggregation_package_modules()
    normalized_adapter_kind = None
    if adapter_kind is not None:
        normalized_adapter_kind = adapter_kind.strip().lower()
    registered = sorted(_AGGREGATION_BACKEND_REGISTRY)
    if normalized_adapter_kind is None:
        return tuple(registered)
    return tuple(key for key in registered if key[0] == normalized_adapter_kind)


def list_shared_adapter_aggregation_backend_catalog_entries() -> tuple[
    RegistryCatalogEntry, ...
]:
    """등록된 aggregation backend catalog entry를 canonical item 기준으로 반환한다."""

    _import_aggregation_package_modules()
    return dedupe_registry_catalog_entries(
        catalog_entry
        for _factory, catalog_entry in _AGGREGATION_BACKEND_REGISTRY.values()
    )


def _import_aggregation_module_for_adapter_kind(normalized_adapter_kind: str) -> None:
    module_name = normalized_adapter_kind.replace("-", "_")
    try:
        importlib.import_module(f"{_AGGREGATION_PACKAGE}.{module_name}")
    except ModuleNotFoundError as error:
        expected_module = f"{_AGGREGATION_PACKAGE}.{module_name}"
        if error.name != expected_module:
            raise


def _import_aggregation_package_modules() -> None:
    package = importlib.import_module(_AGGREGATION_PACKAGE)
    package_paths = getattr(package, "__path__", None)
    if package_paths is None:
        return

    for module_info in pkgutil.iter_modules(package_paths):
        if module_info.name in _SKIPPED_AGGREGATION_MODULES:
            continue
        importlib.import_module(f"{_AGGREGATION_PACKAGE}.{module_info.name}")
