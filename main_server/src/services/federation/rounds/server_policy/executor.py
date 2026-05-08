"""method descriptor가 요구하는 server policy를 runtime capability로 검증한다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from main_server.src.services.federation.rounds.acceptance.errors import (
    RoundValidationError,
)
from methods.federated_ssl.base import FederatedSslMethodDescriptor

ROUND_RUNTIME_AGGREGATION_BACKEND_POLICY_NAME = "round_runtime_aggregation_backend"
ROUND_ACTIVE_PAIR_ONLY_POLICY_NAME = "round_active_pair_only"
ROUND_RUNTIME_AGGREGATION_BACKEND_HINT = "use_round_runtime_aggregation_backend"


@dataclass(frozen=True, slots=True)
class ServerPolicyExecutionSummary:
    """현재 server runtime이 method server policy를 해석한 결과."""

    method_name: str
    round_id: str
    server_aggregator_name: str
    round_policy_name: str
    server_aggregate_hint: str
    update_count: int


class ServerPolicyExecutor(Protocol):
    """main_server가 제공하는 method-agnostic server policy capability."""

    def prepare_finalize(
        self,
        *,
        method_descriptor: FederatedSslMethodDescriptor,
        round_id: str,
        update_count: int,
    ) -> ServerPolicyExecutionSummary:
        """round finalize 전에 method의 server policy 요구사항을 검증한다."""


@dataclass(frozen=True, slots=True)
class DefaultServerPolicyExecutor:
    """기본 live runtime policy: 기존 round aggregation backend를 그대로 사용한다."""

    def prepare_finalize(
        self,
        *,
        method_descriptor: FederatedSslMethodDescriptor,
        round_id: str,
        update_count: int,
    ) -> ServerPolicyExecutionSummary:
        """현재 generic finalize 흐름이 method server policy를 만족하는지 확인한다."""

        if not method_descriptor.runtime_capabilities.live_server_supported:
            raise RoundValidationError(
                "Configured FL SSL method does not support live server runtime: "
                f"{method_descriptor.name}."
            )
        if method_descriptor.requires_custom_server_runtime:
            raise RoundValidationError(
                "Configured FL SSL method requires a custom server runtime "
                "capability, but only the default round aggregation backend "
                f"policy is wired: {method_descriptor.name}."
            )

        server_step = method_descriptor.server_step
        if (
            server_step.server_aggregator_name
            != ROUND_RUNTIME_AGGREGATION_BACKEND_POLICY_NAME
        ):
            raise RoundValidationError(
                "Unsupported server aggregation policy for default live runtime: "
                f"{server_step.server_aggregator_name}."
            )
        if server_step.round_policy_name != ROUND_ACTIVE_PAIR_ONLY_POLICY_NAME:
            raise RoundValidationError(
                "Unsupported round policy for default live runtime: "
                f"{server_step.round_policy_name}."
            )
        if server_step.server_aggregate_hint != ROUND_RUNTIME_AGGREGATION_BACKEND_HINT:
            raise RoundValidationError(
                "Unsupported server aggregate hint for default live runtime: "
                f"{server_step.server_aggregate_hint}."
            )

        return ServerPolicyExecutionSummary(
            method_name=method_descriptor.name,
            round_id=round_id,
            server_aggregator_name=server_step.server_aggregator_name,
            round_policy_name=server_step.round_policy_name,
            server_aggregate_hint=server_step.server_aggregate_hint,
            update_count=update_count,
        )
