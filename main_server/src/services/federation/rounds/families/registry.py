"""Round family registry primitive."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from shared.src.contracts.adapter_family_metadata import (
    SharedAdapterFamilyMetadata,
    list_shared_adapter_family_metadata,
)

from ..aggregation.diagonal_scale_defaults import AggregationConfigScalar
from ..aggregation.registry import build_shared_adapter_aggregation_backend
from .models import (
    RoundFamilyFactory,
    SharedAdapterRoundFamily,
    SharedAdapterRoundFamilyRuntime,
)

_ROUND_FAMILY_REGISTRY: dict[str, RoundFamilyFactory] = {}


def register_shared_adapter_round_family(
    *family_names: str,
    factory: RoundFamilyFactory | None = None,
) -> (
    Callable[[RoundFamilyFactory], RoundFamilyFactory]
    | RoundFamilyFactory
):
    """adapter family factory 옆에서 round runtime wiring을 등록한다."""

    def _decorator(factory: RoundFamilyFactory) -> RoundFamilyFactory:
        for family_name in family_names:
            _ROUND_FAMILY_REGISTRY[family_name.strip().lower()] = factory
        return factory

    if factory is not None:
        return _decorator(factory)
    return _decorator


def build_shared_adapter_round_family(
    family_name: str,
    *,
    aggregation_backend_name: str,
    aggregation_backend_overrides: Mapping[str, AggregationConfigScalar] | None = None,
) -> SharedAdapterRoundFamily:
    """adapter family와 aggregation backend 이름으로 서버 조합 객체를 만든다."""

    normalized_family_name = family_name.strip().lower()
    factory = _ROUND_FAMILY_REGISTRY.get(normalized_family_name)
    if factory is not None:
        return factory(aggregation_backend_name, aggregation_backend_overrides)
    family_metadata = _resolve_shared_adapter_family_metadata(normalized_family_name)
    return SharedAdapterRoundFamilyRuntime(
        adapter_kind=family_metadata.adapter_kind,
        accepted_update_formats=family_metadata.accepted_update_payload_formats,
        aggregation_backend=build_shared_adapter_aggregation_backend(
            adapter_kind=family_metadata.adapter_kind,
            backend_name=aggregation_backend_name,
            overrides=aggregation_backend_overrides,
        ),
    )


def _resolve_shared_adapter_family_metadata(
    normalized_family_name: str,
) -> SharedAdapterFamilyMetadata:
    for family_metadata in list_shared_adapter_family_metadata():
        if (
            family_metadata.family_name.strip().lower() == normalized_family_name
            or family_metadata.adapter_kind.strip().lower() == normalized_family_name
        ):
            return family_metadata
    raise ValueError(
        f"Unsupported shared adapter family: {normalized_family_name}."
    )
