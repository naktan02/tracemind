"""Federated aggregation method metadata registry."""

from __future__ import annotations

from collections.abc import Iterable

from methods.federated.aggregation.base import FederatedAggregationMethodSpec
from methods.federated.aggregation.fedavg.classifier_head_fedavg import (
    compute_classifier_head_fedavg,
)
from methods.federated.aggregation.fedavg.diagonal_scale_fedavg import (
    compute_diagonal_scale_fedavg,
)
from methods.federated.aggregation.fedavg.lora_classifier_fedavg import (
    compute_lora_classifier_fedavg,
)
from shared.src.config.adapter_family_metadata import (
    CLASSIFIER_HEAD_FAMILY_METADATA,
    DIAGONAL_SCALE_FAMILY_METADATA,
    LORA_CLASSIFIER_FAMILY_METADATA,
)

_FEDERATED_AGGREGATION_METHOD_REGISTRY: dict[
    tuple[str, str],
    FederatedAggregationMethodSpec,
] = {}


def register_federated_aggregation_method(
    *,
    adapter_kind: str,
    method_name: str,
    implementation_module: str,
    core_function_name: str,
    aliases: Iterable[str] = (),
) -> FederatedAggregationMethodSpec:
    """adapter family와 method 이름으로 method metadata를 등록한다."""

    spec = FederatedAggregationMethodSpec(
        adapter_kind=adapter_kind.strip().lower(),
        method_name=method_name.strip().lower(),
        implementation_module=implementation_module,
        core_function_name=core_function_name,
        aliases=tuple(alias.strip().lower() for alias in aliases),
    )
    for name in (spec.method_name, *spec.aliases):
        _FEDERATED_AGGREGATION_METHOD_REGISTRY[(spec.adapter_kind, name)] = spec
    return spec


def get_federated_aggregation_method_spec(
    *,
    adapter_kind: str,
    method_name: str,
) -> FederatedAggregationMethodSpec:
    """adapter family와 method 이름에 맞는 method metadata를 반환한다."""

    normalized_key = (adapter_kind.strip().lower(), method_name.strip().lower())
    spec = _FEDERATED_AGGREGATION_METHOD_REGISTRY.get(normalized_key)
    if spec is None:
        raise ValueError(
            "Unsupported federated aggregation method: "
            f"adapter_kind={adapter_kind}, method_name={method_name}"
        )
    return spec


def list_federated_aggregation_method_specs(
    *,
    adapter_kind: str | None = None,
) -> tuple[FederatedAggregationMethodSpec, ...]:
    """등록된 method metadata를 중복 없이 반환한다."""

    normalized_adapter_kind = (
        adapter_kind.strip().lower() if adapter_kind is not None else None
    )
    specs = {
        spec
        for (registered_adapter_kind, _method_name), spec in (
            _FEDERATED_AGGREGATION_METHOD_REGISTRY.items()
        )
        if normalized_adapter_kind is None
        or registered_adapter_kind == normalized_adapter_kind
    }
    return tuple(sorted(specs, key=lambda spec: (spec.adapter_kind, spec.method_name)))


register_federated_aggregation_method(
    adapter_kind=DIAGONAL_SCALE_FAMILY_METADATA.adapter_kind,
    method_name="fedavg",
    implementation_module=compute_diagonal_scale_fedavg.__module__,
    core_function_name=compute_diagonal_scale_fedavg.__name__,
    aliases=("diagonal_scale_fedavg",),
)
register_federated_aggregation_method(
    adapter_kind=CLASSIFIER_HEAD_FAMILY_METADATA.adapter_kind,
    method_name="fedavg",
    implementation_module=compute_classifier_head_fedavg.__module__,
    core_function_name=compute_classifier_head_fedavg.__name__,
    aliases=("classifier_head_fedavg",),
)
register_federated_aggregation_method(
    adapter_kind=LORA_CLASSIFIER_FAMILY_METADATA.adapter_kind,
    method_name="fedavg",
    implementation_module=compute_lora_classifier_fedavg.__module__,
    core_function_name=compute_lora_classifier_fedavg.__name__,
    aliases=("lora_classifier_fedavg",),
)
