"""FL SSL method descriptor contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


def _require_non_empty(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    return normalized


def _normalize_view_names(view_names: tuple[str, ...]) -> tuple[str, ...]:
    normalized_names = tuple(
        _require_non_empty(str(view_name), field_name="view_names item")
        for view_name in view_names
    )
    if not normalized_names:
        raise ValueError("required view_names must not be empty.")
    if len(set(normalized_names)) != len(normalized_names):
        raise ValueError("required view_names must be unique.")
    return normalized_names


@dataclass(frozen=True, slots=True)
class FederatedSslRequiredViews:
    """FL SSL method가 client 입력 row에 요구하는 view surface."""

    view_names: tuple[str, ...]
    view_generator_name: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "view_names",
            _normalize_view_names(self.view_names),
        )
        object.__setattr__(
            self,
            "view_generator_name",
            _require_non_empty(
                self.view_generator_name,
                field_name="view_generator_name",
            ),
        )


@dataclass(frozen=True, slots=True)
class FederatedSslLocalStepSpec:
    """client local step 요구사항."""

    step_name: str
    client_trainer_name: str
    pseudo_labeler_name: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "step_name",
            _require_non_empty(self.step_name, field_name="step_name"),
        )
        object.__setattr__(
            self,
            "client_trainer_name",
            _require_non_empty(
                self.client_trainer_name,
                field_name="client_trainer_name",
            ),
        )
        object.__setattr__(
            self,
            "pseudo_labeler_name",
            _require_non_empty(
                self.pseudo_labeler_name,
                field_name="pseudo_labeler_name",
            ),
        )


@dataclass(frozen=True, slots=True)
class FederatedSslServerStepSpec:
    """server round/aggregation 요구사항."""

    server_aggregator_name: str
    round_policy_name: str
    server_aggregate_hint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "server_aggregator_name",
            _require_non_empty(
                self.server_aggregator_name,
                field_name="server_aggregator_name",
            ),
        )
        object.__setattr__(
            self,
            "round_policy_name",
            _require_non_empty(
                self.round_policy_name,
                field_name="round_policy_name",
            ),
        )
        object.__setattr__(
            self,
            "server_aggregate_hint",
            _require_non_empty(
                self.server_aggregate_hint,
                field_name="server_aggregate_hint",
            ),
        )


@dataclass(frozen=True, slots=True)
class FederatedSslRuntimeCapabilities:
    """method가 현재 지원하는 실행 표면."""

    simulation_supported: bool
    live_agent_supported: bool
    live_server_supported: bool
    requires_custom_client_runtime: bool = False
    requires_custom_server_runtime: bool = False


@dataclass(frozen=True, slots=True)
class FederatedSslMethodDescriptor:
    """FL SSL method의 canonical strategy spec."""

    name: str
    implementation_status: str
    required_views: FederatedSslRequiredViews
    local_step: FederatedSslLocalStepSpec
    server_step: FederatedSslServerStepSpec
    runtime_capabilities: FederatedSslRuntimeCapabilities

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _require_non_empty(self.name, field_name="name"),
        )
        object.__setattr__(
            self,
            "implementation_status",
            _require_non_empty(
                self.implementation_status,
                field_name="implementation_status",
            ),
        )

    @property
    def client_trainer_name(self) -> str:
        return self.local_step.client_trainer_name

    @property
    def pseudo_labeler_name(self) -> str:
        return self.local_step.pseudo_labeler_name

    @property
    def view_generator_name(self) -> str:
        return self.required_views.view_generator_name

    @property
    def server_aggregator_name(self) -> str:
        return self.server_step.server_aggregator_name

    @property
    def round_policy_name(self) -> str:
        return self.server_step.round_policy_name

    @property
    def requires_custom_client_runtime(self) -> bool:
        return self.runtime_capabilities.requires_custom_client_runtime

    @property
    def requires_custom_server_runtime(self) -> bool:
        return self.runtime_capabilities.requires_custom_server_runtime


class FederatedSslMethod(Protocol):
    """registry에 올릴 수 있는 FL SSL method module surface."""

    descriptor: FederatedSslMethodDescriptor
