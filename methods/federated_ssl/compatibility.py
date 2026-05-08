"""FL SSL profile compatibility validation."""

from __future__ import annotations

from dataclasses import dataclass

from methods.federated_ssl.base import FederatedSslMethodDescriptor
from methods.federated_ssl.experiment_profile import FederatedSslExperimentProfile
from methods.federated_ssl.local_update_profile import LocalUpdateProfile


@dataclass(frozen=True, slots=True)
class FederatedSslProfileCompatibilityContext:
    """method/local update/round runtime 조합 검증에 필요한 canonical 값."""

    method_descriptor: FederatedSslMethodDescriptor
    local_update_profile: LocalUpdateProfile | None
    local_update_adapter_kind: str
    round_adapter_family_name: str
    round_aggregation_backend_name: str
    experiment_profile: FederatedSslExperimentProfile | None = None
    round_runtime_profile_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "local_update_adapter_kind",
            _normalize_non_empty(
                self.local_update_adapter_kind,
                field_name="local_update_adapter_kind",
            ),
        )
        object.__setattr__(
            self,
            "round_adapter_family_name",
            _normalize_non_empty(
                self.round_adapter_family_name,
                field_name="round_adapter_family_name",
            ),
        )
        object.__setattr__(
            self,
            "round_aggregation_backend_name",
            _normalize_non_empty(
                self.round_aggregation_backend_name,
                field_name="round_aggregation_backend_name",
            ),
        )
        if self.round_runtime_profile_name is not None:
            object.__setattr__(
                self,
                "round_runtime_profile_name",
                _normalize_non_empty(
                    self.round_runtime_profile_name,
                    field_name="round_runtime_profile_name",
                ),
            )


def validate_federated_ssl_profile_compatibility(
    context: FederatedSslProfileCompatibilityContext,
) -> None:
    """FL SSL 실행 조합이 bootstrap 전에 실패하도록 최소 compatibility를 검증한다."""

    if not context.method_descriptor.runtime_capabilities.simulation_supported:
        raise ValueError(
            "FL profile compatibility failed: method_descriptor="
            f"{context.method_descriptor.name} does not support simulation runtime."
        )
    if context.experiment_profile is not None:
        _validate_experiment_profile_metadata(context)

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
            "FL profile compatibility failed: local_update_profile and "
            "round_runtime_profile must target the same adapter family: "
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
                "FL profile compatibility failed: method recipe does not support "
                "local_update_profile="
                f"{profile_name} for method={context.method_descriptor.name}."
            )
        if context.round_runtime_profile_name is not None and (
            not recipe.supports_profile_combination(
                local_update_profile_name=profile_name,
                round_runtime_profile_name=context.round_runtime_profile_name,
            )
        ):
            raise ValueError(
                "FL profile compatibility failed: method recipe does not support "
                "profile combination: "
                f"method={context.method_descriptor.name}, "
                f"local_update_profile={profile_name}, "
                f"round_runtime_profile={context.round_runtime_profile_name}."
            )

    if not recipe.supports_runtime_pair(
        adapter_family_name=context.round_adapter_family_name,
        aggregation_backend_name=context.round_aggregation_backend_name,
    ):
        raise ValueError(
            "FL profile compatibility failed: method recipe does not support "
            "round runtime pair: "
            f"method={context.method_descriptor.name}, "
            f"adapter_family={context.round_adapter_family_name}, "
            f"aggregation_backend={context.round_aggregation_backend_name}."
        )


def _validate_experiment_profile_metadata(
    context: FederatedSslProfileCompatibilityContext,
) -> None:
    profile = context.experiment_profile
    if profile is None:
        return

    _require_profile_value(
        profile.method_name,
        expected=context.method_descriptor.name,
        field_name="method_name",
        profile_name=profile.name,
    )
    if context.local_update_profile is not None:
        _require_profile_value(
            profile.local_update_profile_name,
            expected=context.local_update_profile.algorithm_profile_name,
            field_name="local_update_profile_name",
            profile_name=profile.name,
        )
    if context.round_runtime_profile_name is None:
        raise ValueError(
            "FL profile compatibility failed: experiment_profile requires "
            "round_runtime_profile_name to validate compose metadata."
        )
    _require_profile_value(
        profile.round_runtime_profile_name,
        expected=context.round_runtime_profile_name,
        field_name="round_runtime_profile_name",
        profile_name=profile.name,
    )
    _require_profile_value(
        profile.adapter_family_name,
        expected=context.round_adapter_family_name,
        field_name="adapter_family_name",
        profile_name=profile.name,
    )
    _require_profile_value(
        profile.aggregation_backend_name,
        expected=context.round_aggregation_backend_name,
        field_name="aggregation_backend_name",
        profile_name=profile.name,
    )


def _require_profile_value(
    actual: str,
    *,
    expected: str,
    field_name: str,
    profile_name: str,
) -> None:
    if actual.lower() == expected.lower():
        return
    raise ValueError(
        "FL profile compatibility failed: experiment_profile metadata drift: "
        f"profile={profile_name}, field={field_name}, "
        f"actual={actual}, expected={expected}."
    )


def _normalize_non_empty(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    return normalized
