"""FL SSL profile compatibility validation."""

from __future__ import annotations

from dataclasses import dataclass

from methods.common.config_reading import normalize_non_empty_str
from methods.federated_ssl.base import FederatedSslMethodDescriptor
from methods.federated_ssl.local_update_profile import LocalUpdateProfile


@dataclass(frozen=True, slots=True)
class FederatedSslProfileCompatibilityContext:
    """method/local update/round runtime 조합 검증에 필요한 canonical 값."""

    method_descriptor: FederatedSslMethodDescriptor
    local_update_profile: LocalUpdateProfile | None
    local_update_adapter_kind: str
    round_adapter_family_name: str
    round_aggregation_backend_name: str

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
