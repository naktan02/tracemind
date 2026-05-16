"""Aggregation backend registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from main_server.src.services.federation.rounds.aggregation.models import (
    AggregationBackendFactory,
    SharedAdapterAggregationBackend,
)
from methods.federated.aggregation.registry import (
    build_federated_aggregation_strategy,
    list_federated_aggregation_method_specs,
)
from shared.src.contracts.registry_catalog_metadata import (
    RegistryCatalogEntry,
    dedupe_registry_catalog_entries,
)

from .artifact_refs import AggregationArtifactStore
from .executor import MethodAggregationBackend
from .models import AggregationConfigScalar

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
    artifact_store: AggregationArtifactStore | None = None,
) -> SharedAdapterAggregationBackend:
    """adapter family와 backend 이름으로 aggregation backend를 조립한다."""

    normalized_key = (adapter_kind.strip().lower(), backend_name.strip().lower())
    registered_backend = _AGGREGATION_BACKEND_REGISTRY.get(normalized_key)
    if registered_backend is not None:
        factory, _catalog_entry = registered_backend
        return factory(overrides)
    return MethodAggregationBackend(
        strategy=build_federated_aggregation_strategy(
            adapter_kind=adapter_kind,
            method_name=backend_name,
            overrides=overrides,
        ),
        overrides=overrides,
        artifact_loader=artifact_store or AggregationArtifactStore(),
    )


def list_registered_shared_adapter_aggregation_backends(
    *,
    adapter_kind: str | None = None,
) -> tuple[tuple[str, str], ...]:
    """등록된 aggregation backend 키를 정렬된 tuple로 반환한다."""

    normalized_adapter_kind = None
    if adapter_kind is not None:
        normalized_adapter_kind = adapter_kind.strip().lower()
    registered = set(_AGGREGATION_BACKEND_REGISTRY)
    registered.update(
        (spec.adapter_kind, spec.method_name)
        for spec in list_federated_aggregation_method_specs(
            adapter_kind=normalized_adapter_kind
        )
    )
    sorted_registered = tuple(sorted(registered))
    if normalized_adapter_kind is None:
        return sorted_registered
    return tuple(key for key in sorted_registered if key[0] == normalized_adapter_kind)


def list_shared_adapter_aggregation_backend_catalog_entries() -> tuple[
    RegistryCatalogEntry, ...
]:
    """등록된 aggregation backend catalog entry를 canonical item 기준으로 반환한다."""

    return dedupe_registry_catalog_entries(
        (
            *(
                catalog_entry
                for _factory, catalog_entry in _AGGREGATION_BACKEND_REGISTRY.values()
            ),
            *(
                RegistryCatalogEntry(
                    item_name=f"{spec.adapter_kind}.{spec.method_name}",
                    display_name=spec.method_name,
                    implementation_module=spec.implementation_module,
                    core_method_name=spec.method_name,
                    family_name=spec.adapter_kind,
                    supported_adapter_kinds=(spec.adapter_kind,),
                    metadata=dict(spec.metadata or {}),
                )
                for spec in list_federated_aggregation_method_specs()
            ),
        )
    )
