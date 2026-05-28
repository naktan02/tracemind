"""FL SSL method descriptor contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

TRAINING_ROW_SOURCE_ALL_ROWS = "all_rows"
TRAINING_ROW_SOURCE_UNLABELED_POOL_WHEN_AVAILABLE = "unlabeled_pool_when_available"
TRAINING_ROW_SOURCE_LABELED_POOL_WHEN_AVAILABLE = "labeled_pool_when_available"
TRAINING_ROW_SOURCES = frozenset(
    {
        TRAINING_ROW_SOURCE_ALL_ROWS,
        TRAINING_ROW_SOURCE_UNLABELED_POOL_WHEN_AVAILABLE,
        TRAINING_ROW_SOURCE_LABELED_POOL_WHEN_AVAILABLE,
    }
)


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


def _normalize_unique_names(
    values: tuple[str, ...],
    *,
    field_name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    normalized = tuple(
        _require_non_empty(str(value), field_name=f"{field_name} item")
        for value in values
    )
    if not allow_empty and not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    if len({value.lower() for value in normalized}) != len(normalized):
        raise ValueError(f"{field_name} must be unique.")
    return normalized


def _set_non_empty(instance: object, field_name: str) -> None:
    object.__setattr__(
        instance,
        field_name,
        _require_non_empty(getattr(instance, field_name), field_name=field_name),
    )


def _set_unique_names(
    instance: object,
    field_name: str,
    *,
    allow_empty: bool = False,
) -> None:
    object.__setattr__(
        instance,
        field_name,
        _normalize_unique_names(
            getattr(instance, field_name),
            field_name=field_name,
            allow_empty=allow_empty,
        ),
    )


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
        _set_non_empty(self, "view_generator_name")


@dataclass(frozen=True, slots=True)
class FederatedSslLocalStepSpec:
    """client local step 요구사항."""

    step_name: str
    client_trainer_name: str
    pseudo_labeler_name: str
    training_row_source: str = TRAINING_ROW_SOURCE_ALL_ROWS
    runtime_entrypoint: str | None = None

    def __post_init__(self) -> None:
        _set_non_empty(self, "step_name")
        _set_non_empty(self, "client_trainer_name")
        _set_non_empty(self, "pseudo_labeler_name")
        _set_non_empty(self, "training_row_source")
        if self.runtime_entrypoint is not None:
            module_name, separator, function_name = self.runtime_entrypoint.partition(
                ":"
            )
            if not separator or not module_name.strip() or not function_name.strip():
                raise ValueError(
                    "runtime_entrypoint must use 'module:function' format."
                )
            object.__setattr__(
                self,
                "runtime_entrypoint",
                _require_non_empty(
                    self.runtime_entrypoint,
                    field_name="runtime_entrypoint",
                ),
            )
        if self.training_row_source not in TRAINING_ROW_SOURCES:
            raise ValueError(
                f"training_row_source must be one of {sorted(TRAINING_ROW_SOURCES)}."
            )


@dataclass(frozen=True, slots=True)
class FederatedSslServerStepSpec:
    """server round/aggregation 요구사항."""

    server_aggregator_name: str
    round_policy_name: str
    server_aggregate_hint: str

    def __post_init__(self) -> None:
        _set_non_empty(self, "server_aggregator_name")
        _set_non_empty(self, "round_policy_name")
        _set_non_empty(self, "server_aggregate_hint")


@dataclass(frozen=True, slots=True)
class FederatedSslRoundStateExchangeSpec:
    """server round가 client metric/state를 교환하는 방식."""

    exchange_name: str
    required_client_metric_keys: tuple[str, ...] = ()
    summary_metric_prefix: str = "round_state"
    requires_custom_exchange: bool = False

    def __post_init__(self) -> None:
        _set_non_empty(self, "exchange_name")
        _set_unique_names(self, "required_client_metric_keys", allow_empty=True)
        _set_non_empty(self, "summary_metric_prefix")


@dataclass(frozen=True, slots=True)
class FederatedSslRequiredCapabilities:
    """method-owned FL SSL method가 요구하는 공통 runtime capability."""

    labeled_exposure_policy_names: tuple[str, ...] = ()
    local_supervision_regime_names: tuple[str, ...] = ()
    server_step_policy_names: tuple[str, ...] = ()
    server_update_policy_names: tuple[str, ...] = ()
    peer_context_policy_names: tuple[str, ...] = ()
    update_partition_policy_names: tuple[str, ...] = ()
    local_ssl_policy_names: tuple[str, ...] = ()
    aggregation_weight_policy_names: tuple[str, ...] = ()
    query_multiview_source_names: tuple[str, ...] = ()
    client_participation_policy_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _set_unique_names(
            self,
            "labeled_exposure_policy_names",
            allow_empty=True,
        )
        _set_unique_names(
            self,
            "local_supervision_regime_names",
            allow_empty=True,
        )
        _set_unique_names(self, "server_step_policy_names", allow_empty=True)
        _set_unique_names(self, "server_update_policy_names", allow_empty=True)
        _set_unique_names(self, "peer_context_policy_names", allow_empty=True)
        _set_unique_names(self, "update_partition_policy_names", allow_empty=True)
        _set_unique_names(self, "local_ssl_policy_names", allow_empty=True)
        _set_unique_names(self, "aggregation_weight_policy_names", allow_empty=True)
        _set_unique_names(self, "query_multiview_source_names", allow_empty=True)
        _set_unique_names(self, "client_participation_policy_names", allow_empty=True)


@dataclass(frozen=True, slots=True)
class FederatedSslRuntimeCapabilities:
    """method가 현재 지원하는 실행 표면."""

    simulation_supported: bool
    live_agent_supported: bool
    live_server_supported: bool
    requires_custom_client_runtime: bool = False
    requires_custom_server_runtime: bool = False


@dataclass(frozen=True, slots=True)
class FederatedSslRuntimePair:
    """method가 지원하는 update family와 aggregation backend 조합."""

    update_family_name: str
    aggregation_backend_name: str

    def __post_init__(self) -> None:
        _set_non_empty(self, "update_family_name")
        _set_non_empty(self, "aggregation_backend_name")

    @property
    def normalized_key(self) -> tuple[str, str]:
        return (
            self.update_family_name.lower(),
            self.aggregation_backend_name.lower(),
        )


@dataclass(frozen=True, slots=True)
class FederatedSslMethodRecipe:
    """FL SSL method가 지원하는 profile/backend/family 조합표."""

    method_name: str
    supported_local_update_profile_names: tuple[str, ...] = ()
    supported_runtime_pairs: tuple[FederatedSslRuntimePair, ...] = ()

    def __post_init__(self) -> None:
        _set_non_empty(self, "method_name")
        _set_unique_names(
            self,
            "supported_local_update_profile_names",
            allow_empty=True,
        )
        if len(set(self._runtime_pair_keys())) != len(self.supported_runtime_pairs):
            raise ValueError("supported_runtime_pairs must be unique.")

    def supports_local_update_profile(self, profile_name: str) -> bool:
        """method가 local update profile을 허용하는지 확인한다."""

        if not self.supported_local_update_profile_names:
            return True
        normalized_profile_name = profile_name.strip().lower()
        return normalized_profile_name in {
            value.lower() for value in self.supported_local_update_profile_names
        }

    def supports_runtime_pair(
        self,
        *,
        update_family_name: str,
        aggregation_backend_name: str,
    ) -> bool:
        """method가 update family/backend 조합을 허용하는지 확인한다."""

        if not self.supported_runtime_pairs:
            return True
        normalized_key = (
            update_family_name.strip().lower(),
            aggregation_backend_name.strip().lower(),
        )
        return normalized_key in set(self._runtime_pair_keys())

    def _runtime_pair_keys(self) -> tuple[tuple[str, str], ...]:
        return tuple(pair.normalized_key for pair in self.supported_runtime_pairs)


@dataclass(frozen=True, slots=True)
class FederatedSslMethodDescriptor:
    """FL SSL method의 canonical strategy spec."""

    name: str
    implementation_status: str
    required_views: FederatedSslRequiredViews
    local_step: FederatedSslLocalStepSpec
    server_step: FederatedSslServerStepSpec
    runtime_capabilities: FederatedSslRuntimeCapabilities
    method_role: str = "method"
    round_state_exchange: FederatedSslRoundStateExchangeSpec | None = None
    recipe: FederatedSslMethodRecipe | None = None
    required_capabilities: FederatedSslRequiredCapabilities = field(
        default_factory=FederatedSslRequiredCapabilities
    )
    display_name: str | None = None

    def __post_init__(self) -> None:
        _set_non_empty(self, "name")
        _set_non_empty(self, "implementation_status")
        _set_non_empty(self, "method_role")
        if self.display_name is not None:
            object.__setattr__(
                self,
                "display_name",
                _require_non_empty(self.display_name, field_name="display_name"),
            )
        if self.recipe is not None and self.recipe.method_name != self.name:
            raise ValueError(
                "method descriptor recipe must use the same method name: "
                f"{self.recipe.method_name} != {self.name}."
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
    def round_state_exchange_name(self) -> str | None:
        if self.round_state_exchange is None:
            return None
        return self.round_state_exchange.exchange_name

    @property
    def requires_custom_client_runtime(self) -> bool:
        return self.runtime_capabilities.requires_custom_client_runtime

    @property
    def requires_custom_server_runtime(self) -> bool:
        return self.runtime_capabilities.requires_custom_server_runtime


class FederatedSslMethod(Protocol):
    """registry에 올릴 수 있는 FL SSL method module surface."""

    descriptor: FederatedSslMethodDescriptor


class FederatedSslLocalObjective(Protocol):
    """client local objective 의미를 method-local module이 설명하는 surface."""

    objective_name: str
    trainer_hint: str


class FederatedSslServerPolicy(Protocol):
    """method-specific server policy surface."""

    policy_name: str
    aggregation_hint: str


class FederatedSslRoundPolicy(Protocol):
    """method-specific round policy surface."""

    policy_name: str
    custom_round_policy_required: bool
