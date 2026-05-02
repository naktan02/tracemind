"""Aggregation backend registry."""

from __future__ import annotations

from collections.abc import Mapping

from main_server.src.services.federation.rounds.aggregation.classifier_head import (
    ClassifierHeadFedAvgAggregationService,
)
from main_server.src.services.federation.rounds.aggregation.diagonal_scale import (
    DiagonalScaleAggregationService,
)
from main_server.src.services.federation.rounds.aggregation.models import (
    AggregationBackendFactory,
    SharedAdapterAggregationBackend,
)
from shared.src.config.adapter_family_metadata import (
    CLASSIFIER_HEAD_FAMILY_METADATA,
    DIAGONAL_SCALE_FAMILY_METADATA,
)
from shared.src.config.registry_catalog_metadata import (
    RegistryCatalogEntry,
    dedupe_registry_catalog_entries,
)

from .diagonal_scale_defaults import AggregationConfigScalar

_AGGREGATION_BACKEND_REGISTRY: dict[
    tuple[str, str],
    tuple[AggregationBackendFactory, RegistryCatalogEntry],
] = {}


def register_shared_adapter_aggregation_backend(
    adapter_kind: str,
    *backend_names: str,
    factory: AggregationBackendFactory,
    catalog_entry: RegistryCatalogEntry,
) -> None:
    """adapter family별 aggregation backend를 얇은 wiring registry에 등록한다."""

    normalized_adapter_kind = adapter_kind.strip().lower()
    registered_backend = (factory, catalog_entry)
    for backend_name in backend_names:
        normalized_backend_name = backend_name.strip().lower()
        _AGGREGATION_BACKEND_REGISTRY[
            (normalized_adapter_kind, normalized_backend_name)
        ] = registered_backend


def build_shared_adapter_aggregation_backend(
    *,
    adapter_kind: str,
    backend_name: str,
    overrides: Mapping[str, AggregationConfigScalar] | None = None,
) -> SharedAdapterAggregationBackend:
    """adapter family와 backend 이름으로 aggregation backend를 조립한다."""

    normalized_key = (adapter_kind.strip().lower(), backend_name.strip().lower())
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

    normalized_adapter_kind = None
    if adapter_kind is not None:
        normalized_adapter_kind = adapter_kind.strip().lower()
    registered = sorted(_AGGREGATION_BACKEND_REGISTRY)
    if normalized_adapter_kind is None:
        return tuple(registered)
    return tuple(key for key in registered if key[0] == normalized_adapter_kind)


def list_shared_adapter_aggregation_backend_catalog_entries(
) -> tuple[RegistryCatalogEntry, ...]:
    """등록된 aggregation backend catalog entry를 canonical item 기준으로 반환한다."""

    return dedupe_registry_catalog_entries(
        catalog_entry
        for _factory, catalog_entry in _AGGREGATION_BACKEND_REGISTRY.values()
    )


register_shared_adapter_aggregation_backend(
    DIAGONAL_SCALE_FAMILY_METADATA.adapter_kind,
    "fedavg",
    "diagonal_scale_fedavg",
    factory=DiagonalScaleAggregationService.from_mapping,
    catalog_entry=RegistryCatalogEntry(
        item_name=f"{DIAGONAL_SCALE_FAMILY_METADATA.adapter_kind}.fedavg",
        display_name="fedavg",
        implementation_module=DiagonalScaleAggregationService.__module__,
        core_method_name="fedavg",
        family_name=DIAGONAL_SCALE_FAMILY_METADATA.adapter_kind,
        supported_adapter_kinds=(DIAGONAL_SCALE_FAMILY_METADATA.adapter_kind,),
        metadata={"adapter_kind": DIAGONAL_SCALE_FAMILY_METADATA.adapter_kind},
    ),
)
register_shared_adapter_aggregation_backend(
    CLASSIFIER_HEAD_FAMILY_METADATA.adapter_kind,
    "fedavg",
    "classifier_head_fedavg",
    factory=lambda overrides: ClassifierHeadFedAvgAggregationService(),
    catalog_entry=RegistryCatalogEntry(
        item_name=f"{CLASSIFIER_HEAD_FAMILY_METADATA.adapter_kind}.fedavg",
        display_name="fedavg",
        implementation_module=ClassifierHeadFedAvgAggregationService.__module__,
        core_method_name="fedavg",
        family_name=CLASSIFIER_HEAD_FAMILY_METADATA.adapter_kind,
        supported_adapter_kinds=(CLASSIFIER_HEAD_FAMILY_METADATA.adapter_kind,),
        metadata={"adapter_kind": CLASSIFIER_HEAD_FAMILY_METADATA.adapter_kind},
    ),
)
