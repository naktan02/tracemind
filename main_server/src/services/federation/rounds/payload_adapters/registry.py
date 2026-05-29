"""Round payload adapter registry primitive."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from shared.src.contracts.adapter_contract_families.registry import (
    get_shared_adapter_update_payload_formats,
    list_registered_shared_adapter_payload_adapter_kinds,
)

from ..aggregation.artifact_refs import AggregationArtifactStore
from ..aggregation.models import AggregationConfigScalar
from ..aggregation.registry import build_shared_adapter_aggregation_backend
from .models import (
    RoundPayloadAdapterFactory,
    SharedAdapterRoundPayloadAdapter,
    SharedAdapterRoundPayloadAdapterRuntime,
)

_ROUND_PAYLOAD_ADAPTER_REGISTRY: dict[str, RoundPayloadAdapterFactory] = {}


def register_shared_adapter_round_payload_adapter(
    *payload_adapter_kinds: str,
    factory: RoundPayloadAdapterFactory | None = None,
) -> (
    Callable[[RoundPayloadAdapterFactory], RoundPayloadAdapterFactory]
    | RoundPayloadAdapterFactory
):
    """payload adapter factory 옆에서 round runtime wiring을 등록한다."""

    def _decorator(factory: RoundPayloadAdapterFactory) -> RoundPayloadAdapterFactory:
        for payload_adapter_kind in payload_adapter_kinds:
            _ROUND_PAYLOAD_ADAPTER_REGISTRY[
                payload_adapter_kind.strip().lower()
            ] = factory
        return factory

    if factory is not None:
        return _decorator(factory)
    return _decorator


def build_shared_adapter_round_payload_adapter(
    payload_adapter_kind: str,
    *,
    aggregation_backend_name: str,
    aggregation_backend_overrides: Mapping[str, AggregationConfigScalar] | None = None,
    aggregation_artifact_store: AggregationArtifactStore | None = None,
) -> SharedAdapterRoundPayloadAdapter:
    """payload adapter와 aggregation backend 이름으로 서버 조합 객체를 만든다."""

    normalized_payload_adapter_kind = payload_adapter_kind.strip().lower()
    factory = _ROUND_PAYLOAD_ADAPTER_REGISTRY.get(normalized_payload_adapter_kind)
    if factory is not None:
        return factory(aggregation_backend_name, aggregation_backend_overrides)
    adapter_kind = _resolve_shared_adapter_kind(normalized_payload_adapter_kind)
    return SharedAdapterRoundPayloadAdapterRuntime(
        adapter_kind=adapter_kind,
        accepted_update_formats=get_shared_adapter_update_payload_formats(adapter_kind),
        aggregation_backend=build_shared_adapter_aggregation_backend(
            adapter_kind=adapter_kind,
            backend_name=aggregation_backend_name,
            overrides=aggregation_backend_overrides,
            artifact_store=aggregation_artifact_store,
        ),
    )


def _resolve_shared_adapter_kind(
    normalized_payload_adapter_kind: str,
) -> str:
    for adapter_kind in list_registered_shared_adapter_payload_adapter_kinds():
        if adapter_kind.strip().lower() == normalized_payload_adapter_kind:
            return adapter_kind
    raise ValueError(
        f"Unsupported shared payload adapter: {normalized_payload_adapter_kind}."
    )
