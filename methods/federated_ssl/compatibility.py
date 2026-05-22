"""FL SSL profile compatibility validation."""

from __future__ import annotations

from dataclasses import dataclass

from methods.common.config_reading import normalize_non_empty_str
from methods.federated.client_split import LABELED_EXPOSURE_SERVER_ONLY_SEED
from methods.federated_ssl.base import FederatedSslMethodDescriptor
from methods.federated_ssl.capability_plan import (
    LOCAL_SUPERVISION_CLIENT_UNLABELED_ONLY,
    LOCAL_SUPERVISION_SERVER_LABELED_ONLY,
    SERVER_STEP_NONE,
    SERVER_STEP_SUPERVISED_SEED,
    FederatedSslCapabilityPlan,
)
from methods.federated_ssl.local_update_profile import LocalUpdateProfile


@dataclass(frozen=True, slots=True)
class FederatedSslProfileCompatibilityContext:
    """method/local update/round runtime 조합 검증에 필요한 canonical 값."""

    method_descriptor: FederatedSslMethodDescriptor
    local_update_profile: LocalUpdateProfile | None
    local_update_adapter_kind: str
    round_adapter_family_name: str
    round_aggregation_backend_name: str
    capability_plan: FederatedSslCapabilityPlan | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "local_update_adapter_kind",
            normalize_non_empty_str(
                self.local_update_adapter_kind,
                field_name="local_update_adapter_kind",
            ),
        )
        object.__setattr__(
            self,
            "round_adapter_family_name",
            normalize_non_empty_str(
                self.round_adapter_family_name,
                field_name="round_adapter_family_name",
            ),
        )
        object.__setattr__(
            self,
            "round_aggregation_backend_name",
            normalize_non_empty_str(
                self.round_aggregation_backend_name,
                field_name="round_aggregation_backend_name",
            ),
        )


def validate_federated_ssl_profile_compatibility(
    context: FederatedSslProfileCompatibilityContext,
) -> None:
    """FL SSL 실행 조합이 bootstrap 전에 실패하도록 최소 compatibility를 검증한다."""

    if not context.method_descriptor.runtime_capabilities.simulation_supported:
        raise ValueError(
            "FL SSL compatibility failed: method_descriptor="
            f"{context.method_descriptor.name} does not support simulation runtime."
        )

    if (
        context.local_update_adapter_kind.lower()
        != context.round_adapter_family_name.lower()
    ):
        profile_name = (
            None
            if context.local_update_profile is None
            else context.local_update_profile.algorithm_profile_name
        )
        raise ValueError(
            "FL SSL compatibility failed: local_update_profile and round_runtime "
            "must target the same adapter family: "
            f"local_update_profile={profile_name}, "
            f"local_update_adapter_kind={context.local_update_adapter_kind}, "
            f"round_adapter_family={context.round_adapter_family_name}."
        )

    recipe = context.method_descriptor.recipe
    if recipe is None:
        return

    if context.local_update_profile is not None:
        profile_name = context.local_update_profile.algorithm_profile_name
        if not recipe.supports_local_update_profile(profile_name):
            raise ValueError(
                "FL SSL compatibility failed: method recipe does not support "
                "local_update_profile="
                f"{profile_name} for method={context.method_descriptor.name}."
            )

    if not recipe.supports_runtime_pair(
        adapter_family_name=context.round_adapter_family_name,
        aggregation_backend_name=context.round_aggregation_backend_name,
    ):
        raise ValueError(
            "FL SSL compatibility failed: method recipe does not support "
            "round runtime pair: "
            f"method={context.method_descriptor.name}, "
            f"adapter_family={context.round_adapter_family_name}, "
            f"aggregation_backend={context.round_aggregation_backend_name}."
        )

    if context.capability_plan is not None:
        validate_federated_ssl_capability_compatibility(
            method_descriptor=context.method_descriptor,
            capability_plan=context.capability_plan,
        )


def validate_federated_ssl_capability_compatibility(
    *,
    method_descriptor: FederatedSslMethodDescriptor | None,
    capability_plan: FederatedSslCapabilityPlan,
) -> None:
    """data exposure/runtime capability 조합을 bootstrap 전에 검증한다."""

    _validate_server_only_semantics(
        method_descriptor=method_descriptor,
        capability_plan=capability_plan,
    )
    if method_descriptor is None:
        _validate_manual_capability_plan(capability_plan)
        return
    required = method_descriptor.required_capabilities
    _require_supported_capability(
        actual=capability_plan.labeled_exposure_policy_name,
        supported=required.labeled_exposure_policy_names,
        field_name="labeled_exposure_policy",
        method_name=method_descriptor.name,
    )
    _require_supported_capability(
        actual=capability_plan.local_supervision_regime_name,
        supported=required.local_supervision_regime_names,
        field_name="local_supervision_regime",
        method_name=method_descriptor.name,
    )
    _require_supported_capability(
        actual=capability_plan.server_step_policy_name,
        supported=required.server_step_policy_names,
        field_name="server_step_policy",
        method_name=method_descriptor.name,
    )
    _require_supported_capability(
        actual=capability_plan.peer_context_policy_name,
        supported=required.peer_context_policy_names,
        field_name="peer_context_policy",
        method_name=method_descriptor.name,
    )
    _require_supported_capability(
        actual=capability_plan.update_partition_policy_name,
        supported=required.update_partition_policy_names,
        field_name="update_partition_policy",
        method_name=method_descriptor.name,
    )
    _require_supported_capability(
        actual=capability_plan.aggregation_weight_policy.name,
        supported=required.aggregation_weight_policy_names,
        field_name="aggregation_weight_policy",
        method_name=method_descriptor.name,
    )
    _require_supported_capability(
        actual=capability_plan.query_multiview_source_name,
        supported=required.query_multiview_source_names,
        field_name="query_multiview_source",
        method_name=method_descriptor.name,
    )
    _require_supported_capability(
        actual=capability_plan.client_participation_policy.name,
        supported=required.client_participation_policy_names,
        field_name="client_participation_policy",
        method_name=method_descriptor.name,
    )


def _validate_manual_capability_plan(
    capability_plan: FederatedSslCapabilityPlan,
) -> None:
    if capability_plan.server_step_policy_name != SERVER_STEP_NONE:
        raise ValueError(
            "manual FL SSL baseline only supports server_step_policy=none."
        )


def _validate_server_only_semantics(
    *,
    method_descriptor: FederatedSslMethodDescriptor | None,
    capability_plan: FederatedSslCapabilityPlan,
) -> None:
    if (
        capability_plan.labeled_exposure_policy_name
        != LABELED_EXPOSURE_SERVER_ONLY_SEED
    ):
        return
    if capability_plan.server_step_policy_name != SERVER_STEP_SUPERVISED_SEED:
        raise ValueError(
            "server_only_seed requires server_step_policy=supervised_seed_step."
        )
    if capability_plan.local_supervision_regime_name not in {
        LOCAL_SUPERVISION_CLIENT_UNLABELED_ONLY,
        LOCAL_SUPERVISION_SERVER_LABELED_ONLY,
    }:
        raise ValueError(
            "server_only_seed requires a local supervision regime that does not "
            "expose labeled rows to clients."
        )
    if method_descriptor is None:
        raise ValueError("server_only_seed requires a method-owned FL SSL descriptor.")


def _require_supported_capability(
    *,
    actual: str,
    supported: tuple[str, ...],
    field_name: str,
    method_name: str,
) -> None:
    if not supported:
        return
    if actual.lower() in {value.lower() for value in supported}:
        return
    raise ValueError(
        "FL SSL compatibility failed: method does not support "
        f"{field_name}={actual!r}: method={method_name}, "
        f"supported={list(supported)}."
    )
