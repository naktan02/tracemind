"""FL SSL method-local client diagnostic payload helpers."""

from __future__ import annotations

import pkgutil
from collections.abc import Callable, Iterable, Mapping
from importlib import import_module
from types import ModuleType

from methods.federated_ssl.method_module_resolution import import_method_family_module

NumericSummaryFn = Callable[[Iterable[float]], Mapping[str, float | int | None]]

_FEDERATED_SSL_PACKAGE = "methods.federated_ssl"


def extract_client_method_diagnostics(
    *,
    method_name: str | None,
    metrics: Mapping[str, object],
) -> dict[str, float]:
    """method 이름으로 해당 method가 report에 공개하는 client 진단값만 추출한다."""

    if method_name is None:
        return {}
    metric_names = _client_diagnostic_metric_names(method_name)
    return normalize_client_method_diagnostics(
        {
            metric_name: metrics[metric_name]
            for metric_name in metric_names
            if metric_name in metrics
        }
    )


def client_method_diagnostics_from_payload(
    payload: Mapping[str, object],
) -> dict[str, float]:
    """generic method_diagnostics와 legacy flat key를 canonical mapping으로 복원한다."""

    raw = dict(_mapping_payload(payload.get("method_diagnostics")))
    for metric_name in known_client_diagnostic_metric_names():
        if metric_name in payload:
            raw[metric_name] = payload[metric_name]
    return normalize_client_method_diagnostics(raw)


def normalize_client_method_diagnostics(
    diagnostics: Mapping[str, object],
) -> dict[str, float]:
    """client diagnostic mapping을 JSON/report 가능한 float mapping으로 정규화한다."""

    normalized: dict[str, float] = {}
    for raw_key, raw_value in diagnostics.items():
        if raw_value is None:
            continue
        key = str(raw_key).strip()
        if not key:
            continue
        value = float(raw_value)
        if value < 0.0 and key in known_non_negative_client_diagnostic_metric_names():
            raise ValueError(f"{key} must be non-negative.")
        normalized[key] = value
    return normalized


def client_method_diagnostics_payload(
    diagnostics: Mapping[str, float],
) -> dict[str, object]:
    """round report client payload에 넣을 generic/legacy-compatible 진단 payload."""

    normalized = normalize_client_method_diagnostics(diagnostics)
    payload: dict[str, object] = {
        "method_diagnostics": dict(sorted(normalized.items()))
    }
    payload.update(_legacy_flat_client_diagnostic_payload(normalized))
    return payload


def client_method_diagnostics_summary_payload(
    diagnostics_by_client: Iterable[Mapping[str, float]],
    *,
    numeric_summary: NumericSummaryFn,
) -> dict[str, object]:
    """aggregation diagnostics에 넣을 method-owned client 진단 summary를 만든다."""

    diagnostics = tuple(
        normalize_client_method_diagnostics(client_diagnostics)
        for client_diagnostics in diagnostics_by_client
    )
    payload: dict[str, object] = {}
    for module in _known_method_diagnostic_modules():
        builder = getattr(module, "client_diagnostic_summary_payload", None)
        if callable(builder):
            payload.update(builder(diagnostics, numeric_summary=numeric_summary))
    return payload


def known_client_diagnostic_metric_names() -> tuple[str, ...]:
    """현재 등록된 method들이 legacy flat payload로 노출하는 client metric key."""

    names: set[str] = set()
    for module in _known_method_diagnostic_modules():
        provider = getattr(module, "client_diagnostic_metric_names", None)
        if callable(provider):
            names.update(str(name) for name in provider())
    return tuple(sorted(names))


def known_non_negative_client_diagnostic_metric_names() -> frozenset[str]:
    """음수가 될 수 없는 method client diagnostic key 집합."""

    names: set[str] = set()
    for module in _known_method_diagnostic_modules():
        provider = getattr(module, "non_negative_client_diagnostic_metric_names", None)
        if callable(provider):
            names.update(str(name) for name in provider())
    return frozenset(names)


def _client_diagnostic_metric_names(method_name: str) -> tuple[str, ...]:
    module = _import_method_client_diagnostics_module(method_name)
    if module is None:
        return ()
    provider = getattr(module, "client_diagnostic_metric_names", None)
    if not callable(provider):
        return ()
    return tuple(str(name) for name in provider())


def _legacy_flat_client_diagnostic_payload(
    diagnostics: Mapping[str, float],
) -> dict[str, float]:
    return {
        metric_name: diagnostics[metric_name]
        for metric_name in known_client_diagnostic_metric_names()
        if metric_name in diagnostics
    }


def _mapping_payload(value: object) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("method_diagnostics must be a mapping.")
    return value


def _known_method_diagnostic_modules() -> tuple[ModuleType, ...]:
    modules: list[ModuleType] = []
    package = import_module(_FEDERATED_SSL_PACKAGE)
    package_paths = getattr(package, "__path__", None)
    if package_paths is None:
        return ()
    for module_info in pkgutil.iter_modules(package_paths):
        if not module_info.ispkg:
            continue
        method_name = module_info.name
        module = _import_method_client_diagnostics_module(method_name)
        if module is not None:
            modules.append(module)
    return tuple(modules)


def _import_method_client_diagnostics_module(method_name: str) -> ModuleType | None:
    return import_method_family_module(
        method_name=method_name,
        module_leaf="client_diagnostics",
    )
