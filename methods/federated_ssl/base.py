"""FL SSL method descriptor contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FederatedSslMethodDescriptor:
    """FL SSL method가 사용하는 runtime 조합을 설명한다."""

    name: str
    implementation_status: str
    client_trainer_name: str
    pseudo_labeler_name: str
    view_generator_name: str
    server_aggregator_name: str
    round_policy_name: str
    requires_custom_client_runtime: bool
    requires_custom_server_runtime: bool
