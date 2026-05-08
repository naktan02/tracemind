"""Federated aggregation method metadata and strategy registry."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Iterable, Mapping

from methods.federated.aggregation.base import (
    AggregationConfigScalar,
    FederatedAggregationMethodSpec,
    FederatedAggregationStrategy,
    FederatedAggregationStrategyFactory,
)

_AGGREGATION_PACKAGE = "methods.federated.aggregation"
_ADAPTATION_PACKAGE = "methods.adaptation"
_SKIPPED_AGGREGATION_MODULE_PARTS = frozenset(
    {
        "base",
        "builtin_loader",
        "registry",
    }
)

_FEDERATED_AGGREGATION_METHOD_REGISTRY: dict[
    tuple[str, str],
    FederatedAggregationMethodSpec,
] = {}
_FEDERATED_AGGREGATION_STRATEGY_REGISTRY: dict[
    tuple[str, str],
    tuple[FederatedAggregationStrategyFactory, FederatedAggregationMethodSpec],
] = {}


def register_federated_aggregation_strategy(
    *,
    adapter_kind: str,
    method_name: str,
    implementation_module: str,
    core_function_name: str,
    aliases: Iterable[str] = (),
    metadata: Mapping[str, AggregationConfigScalar | None] | None = None,
    factory: FederatedAggregationStrategyFactory,
) -> None:
    """strategy factory 옆에서 methods-owned aggregation 실행 surface를 등록한다."""

    spec = FederatedAggregationMethodSpec(
        adapter_kind=adapter_kind.strip().lower(),
        method_name=method_name.strip().lower(),
        implementation_module=implementation_module,
        core_function_name=core_function_name,
        aliases=tuple(alias.strip().lower() for alias in aliases),
        metadata=dict(metadata or {}),
    )
    for name in (spec.method_name, *spec.aliases):
        normalized_name = name.strip().lower()
        _FEDERATED_AGGREGATION_STRATEGY_REGISTRY[
            (spec.adapter_kind, normalized_name)
        ] = (factory, spec)
        _FEDERATED_AGGREGATION_METHOD_REGISTRY[(spec.adapter_kind, normalized_name)] = (
            spec
        )


def get_federated_aggregation_method_spec(
    *,
    adapter_kind: str,
    method_name: str,
) -> FederatedAggregationMethodSpec:
    """adapter family와 method 이름에 맞는 method metadata를 반환한다."""

    normalized_key = (adapter_kind.strip().lower(), method_name.strip().lower())
    if not _import_aggregation_strategy_module(
        normalized_method_name=normalized_key[1],
    ):
        _import_aggregation_package_modules()
    if not _import_adapter_projection_module(
        normalized_adapter_kind=normalized_key[0],
        normalized_method_name=normalized_key[1],
    ):
        _import_adaptation_projection_modules()
    spec = _FEDERATED_AGGREGATION_METHOD_REGISTRY.get(normalized_key)
    if spec is None:
        raise ValueError(
            "Unsupported federated aggregation method: "
            f"adapter_kind={adapter_kind}, method_name={method_name}"
        )
    return spec


def build_federated_aggregation_strategy(
    *,
    adapter_kind: str,
    method_name: str,
    overrides: Mapping[str, AggregationConfigScalar] | None = None,
) -> FederatedAggregationStrategy:
    """adapter family와 aggregation method 이름으로 methods-owned strategy를 만든다."""

    normalized_key = (adapter_kind.strip().lower(), method_name.strip().lower())
    if not _import_aggregation_strategy_module(
        normalized_method_name=normalized_key[1]
    ):
        _import_aggregation_package_modules()
    if not _import_adapter_projection_module(
        normalized_adapter_kind=normalized_key[0],
        normalized_method_name=normalized_key[1],
    ):
        _import_adaptation_projection_modules()
    registered_strategy = _FEDERATED_AGGREGATION_STRATEGY_REGISTRY.get(normalized_key)
    if registered_strategy is not None:
        factory, _spec = registered_strategy
        return factory(overrides)
    raise ValueError(
        "Unsupported federated aggregation strategy: "
        f"adapter_kind={adapter_kind}, method_name={method_name}"
    )


def list_federated_aggregation_method_specs(
    *,
    adapter_kind: str | None = None,
) -> tuple[FederatedAggregationMethodSpec, ...]:
    """등록된 method metadata를 중복 없이 반환한다."""

    _import_aggregation_package_modules()
    _import_adaptation_projection_modules()
    normalized_adapter_kind = (
        adapter_kind.strip().lower() if adapter_kind is not None else None
    )
    specs: dict[tuple[str, str], FederatedAggregationMethodSpec] = {}
    for registered_adapter_kind, _method_name in _FEDERATED_AGGREGATION_METHOD_REGISTRY:
        if (
            normalized_adapter_kind is not None
            and registered_adapter_kind != normalized_adapter_kind
        ):
            continue
        spec = _FEDERATED_AGGREGATION_METHOD_REGISTRY[
            (registered_adapter_kind, _method_name)
        ]
        specs[(spec.adapter_kind, spec.method_name)] = spec
    return tuple(
        sorted(specs.values(), key=lambda spec: (spec.adapter_kind, spec.method_name))
    )


def _import_aggregation_strategy_module(*, normalized_method_name: str) -> bool:
    method_package = f"{_AGGREGATION_PACKAGE}.{normalized_method_name}"
    try:
        importlib.import_module(f"{method_package}.strategy")
    except ModuleNotFoundError as error:
        expected_module = f"{method_package}.strategy"
        if error.name not in {method_package, expected_module}:
            raise
        return False
    return True


def _import_adapter_projection_module(
    *,
    normalized_adapter_kind: str,
    normalized_method_name: str,
) -> bool:
    module_name = (
        f"{_ADAPTATION_PACKAGE}."
        f"{normalized_adapter_kind.replace('-', '_')}."
        f"{normalized_method_name.replace('-', '_')}_projection"
    )
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        return False
    return True


def _import_aggregation_package_modules() -> None:
    package = importlib.import_module(_AGGREGATION_PACKAGE)
    package_paths = getattr(package, "__path__", None)
    if package_paths is None:
        return

    for module_info in pkgutil.walk_packages(
        package_paths,
        prefix=f"{_AGGREGATION_PACKAGE}.",
    ):
        relative_parts = module_info.name.removeprefix(
            f"{_AGGREGATION_PACKAGE}."
        ).split(".")
        if any(part in _SKIPPED_AGGREGATION_MODULE_PARTS for part in relative_parts):
            continue
        importlib.import_module(module_info.name)


def _import_adaptation_projection_modules() -> None:
    package = importlib.import_module(_ADAPTATION_PACKAGE)
    package_paths = getattr(package, "__path__", None)
    if package_paths is None:
        return

    for module_info in pkgutil.walk_packages(
        package_paths,
        prefix=f"{_ADAPTATION_PACKAGE}.",
    ):
        if module_info.name.endswith("_projection"):
            importlib.import_module(module_info.name)
