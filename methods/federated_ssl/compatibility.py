"""FL SSL profile compatibility validation."""

from __future__ import annotations

from dataclasses import dataclass

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


def validate_federated_ssl_profile_compatibility(
    context: FederatedSslProfileCompatibilityContext,
) -> None:
    """FL SSL 실행 조합이 bootstrap 전에 실패하도록 최소 compatibility를 검증한다."""

    if not context.method_descriptor.runtime_capabilities.simulation_supported:
        raise ValueError(
            "FL profile compatibility failed: method_descriptor="
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
            "FL profile compatibility failed: local_update_profile and "
            "round_runtime_profile must target the same adapter family: "
            f"local_update_profile={profile_name}, "
            f"local_update_adapter_kind={context.local_update_adapter_kind}, "
            f"round_adapter_family={context.round_adapter_family_name}."
        )


def _normalize_non_empty(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    return normalized
