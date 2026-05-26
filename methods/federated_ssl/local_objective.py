"""Method-owned FL SSL local objective port."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from types import ModuleType
from typing import Protocol


@dataclass(frozen=True, slots=True)
class FederatedSslLocalObjectiveSpec:
    """client local objective의 method-owned metadata."""

    objective_name: str
    required_batch_views: tuple[str, ...] = ()
    metric_prefix: str = "local_objective"
    parameters: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.objective_name.strip():
            raise ValueError("objective_name must not be empty.")
        if not self.metric_prefix.strip():
            raise ValueError("metric_prefix must not be empty.")


class FederatedSslLocalObjective(Protocol):
    """runtime adapter가 method-owned local objective를 호출하는 port."""

    spec: FederatedSslLocalObjectiveSpec


def requires_method_helper_probability_provider(
    *,
    method_name: str,
    local_ssl_policy_name: str,
) -> bool:
    """method-local runtime requirement가 helper probability provider를 요구하는지 반환한다."""

    requirements_module = _import_method_runtime_requirements_module(method_name)
    if requirements_module is None:
        return False
    resolver = getattr(requirements_module, "requires_helper_probability_provider", None)
    if resolver is None:
        return False
    return bool(resolver(local_ssl_policy_name=local_ssl_policy_name))


def _import_method_runtime_requirements_module(method_name: str) -> ModuleType | None:
    normalized_method_name = method_name.strip().lower().replace("-", "_")
    if not normalized_method_name:
        raise ValueError("method_name must not be empty.")
    module_name = (
        f"methods.federated_ssl.{normalized_method_name}.runtime_requirements"
    )
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name or module_name.startswith(f"{exc.name}."):
            return None
        raise
