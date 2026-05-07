"""Federated aggregation method metadata registry."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Iterable

from methods.federated.aggregation.base import FederatedAggregationMethodSpec

FederatedAggregationCoreFunction = Callable[..., object]
_AGGREGATION_PACKAGE = "methods.federated.aggregation"
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

    normalized_key = (adapter_kind.strip().lower(), method_name.strip().lower())
    _import_aggregation_method_module(
        normalized_adapter_kind=normalized_key[0],
        normalized_method_name=normalized_key[1],
    )
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

    _import_aggregation_package_modules()
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


def _import_aggregation_method_module(
    *,
    normalized_adapter_kind: str,
    normalized_method_name: str,
) -> None:
    module_name = (
        f"{normalized_adapter_kind.replace('-', '_')}_"
        f"{normalized_method_name.replace('-', '_')}"
    )
    method_package = f"{_AGGREGATION_PACKAGE}.{normalized_method_name}"
    try:
        importlib.import_module(f"{method_package}.{module_name}")
    except ModuleNotFoundError as error:
        expected_module = f"{method_package}.{module_name}"
        if error.name != expected_module:
            raise


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
