"""Federated aggregation method metadata registry."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from methods.federated.aggregation.base import FederatedAggregationMethodSpec

FederatedAggregationCoreFunction = Callable[..., object]

_FEDERATED_AGGREGATION_METHOD_REGISTRY: dict[
    tuple[str, str],
    FederatedAggregationMethodSpec,
] = {}


def register_federated_aggregation_method(
    *,
    adapter_kind: str,
    method_name: str,
    aliases: Iterable[str] = (),
) -> Callable[[FederatedAggregationCoreFunction], FederatedAggregationCoreFunction]:
    """core function 옆에서 adapter family별 aggregation method를 등록한다."""

    def _decorator(
        core_function: FederatedAggregationCoreFunction,
    ) -> FederatedAggregationCoreFunction:
        spec = FederatedAggregationMethodSpec(
            adapter_kind=adapter_kind.strip().lower(),
            method_name=method_name.strip().lower(),
            implementation_module=core_function.__module__,
            core_function_name=core_function.__name__,
            aliases=tuple(alias.strip().lower() for alias in aliases),
        )
        for name in (spec.method_name, *spec.aliases):
            _FEDERATED_AGGREGATION_METHOD_REGISTRY[(spec.adapter_kind, name)] = spec
        return core_function

    return _decorator


def get_federated_aggregation_method_spec(
    *,
    adapter_kind: str,
    method_name: str,
) -> FederatedAggregationMethodSpec:
    """adapter family와 method 이름에 맞는 method metadata를 반환한다."""

    _ensure_builtin_federated_aggregation_methods_loaded()
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

    _ensure_builtin_federated_aggregation_methods_loaded()
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


def _ensure_builtin_federated_aggregation_methods_loaded() -> None:
    from methods.federated.aggregation.builtin_loader import (
        load_builtin_federated_aggregation_methods,
    )

    load_builtin_federated_aggregation_methods()
