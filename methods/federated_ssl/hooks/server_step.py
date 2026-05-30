"""Method-owned FL SSL server step policy specs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from methods.federated_ssl.method_module_resolution import (
    import_method_family_module,
)


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

    method_module = import_method_family_module(
        method_name=method_name,
        module_leaf="server_step_parameters",
    )
    if method_module is None:
        raise NotImplementedError(
            f"Method-owned server step parameter module is not wired: {method_name}"
        )
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
