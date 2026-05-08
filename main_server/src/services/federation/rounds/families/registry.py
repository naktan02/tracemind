"""Round family registry primitive."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from shared.src.contracts.adapter_contract_families.registry import (
    get_shared_adapter_update_payload_formats,
    list_registered_shared_adapter_payload_adapter_kinds,
)

from ..aggregation.models import AggregationConfigScalar
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
) -> Callable[[RoundFamilyFactory], RoundFamilyFactory] | RoundFamilyFactory:
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
    adapter_kind = _resolve_shared_adapter_kind(normalized_family_name)
    return SharedAdapterRoundFamilyRuntime(
        adapter_kind=adapter_kind,
        accepted_update_formats=get_shared_adapter_update_payload_formats(adapter_kind),
        aggregation_backend=build_shared_adapter_aggregation_backend(
            adapter_kind=adapter_kind,
            backend_name=aggregation_backend_name,
            overrides=aggregation_backend_overrides,
        ),
    )


def _resolve_shared_adapter_kind(
    normalized_family_name: str,
) -> str:
    for adapter_kind in list_registered_shared_adapter_payload_adapter_kinds():
        if adapter_kind.strip().lower() == normalized_family_name:
            return adapter_kind
    raise ValueError(f"Unsupported shared adapter family: {normalized_family_name}.")
