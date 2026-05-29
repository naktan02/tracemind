"""Method-owned FL SSL server step policy specs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
from types import ModuleType


@dataclass(frozen=True, slots=True)
class FederatedSslServerStepPolicy:
    """server-side update policy metadata."""

    policy_name: str
    parameters: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.policy_name.strip():
            raise ValueError("policy_name must not be empty.")


@dataclass(frozen=True, slots=True)
class FederatedSslSupervisedSeedStepParameters:
    """server-side supervised seed step 실행 budget."""

    epochs: int
    batch_size: int

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("supervised_seed_step server epochs must be positive.")
        if self.batch_size <= 0:
            raise ValueError("supervised_seed_step server batch size must be positive.")


def resolve_method_supervised_seed_step_parameters(
    *,
    method_name: str,
    effective_parameters: Mapping[str, object] | None,
    default_epochs: int,
    default_batch_size: int,
    round_index: int,
) -> FederatedSslSupervisedSeedStepParameters:
    """method-local server seed parameter resolver를 convention으로 호출한다."""

    method_module = _import_method_server_step_parameters_module(method_name)
    resolver = getattr(method_module, "resolve_supervised_seed_step_parameters", None)
    if resolver is None:
        raise NotImplementedError(
            "Method-owned supervised seed step parameter resolver is not wired: "
            f"{method_name}"
        )
    resolved = resolver(
        effective_parameters=effective_parameters,
        default_epochs=default_epochs,
        default_batch_size=default_batch_size,
        round_index=round_index,
    )
    if not isinstance(resolved, FederatedSslSupervisedSeedStepParameters):
        raise TypeError(
            "method supervised seed step resolver must return "
            "FederatedSslSupervisedSeedStepParameters."
        )
    return resolved


def _import_method_server_step_parameters_module(method_name: str) -> ModuleType:
    normalized_method_name = method_name.strip().lower().replace("-", "_")
    if not normalized_method_name:
        raise ValueError("method_name must not be empty.")
    module_name = (
        f"methods.federated_ssl.{normalized_method_name}.server_step_parameters"
    )
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name or module_name.startswith(f"{exc.name}."):
            raise NotImplementedError(
                f"Method-owned server step parameter module is not wired: {method_name}"
            ) from exc
        raise
