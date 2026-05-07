"""FedAvg pseudo-label server policy seam."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FedAvgPseudoLabelServerPolicy:
    """현재는 round runtime aggregation backend를 그대로 사용하는 server policy."""

    policy_name: str = "round_runtime_aggregation_backend"
    aggregation_hint: str = "use_round_runtime_aggregation_backend"
    custom_server_runtime_required: bool = False


FEDAVG_PSEUDO_LABEL_SERVER_POLICY = FedAvgPseudoLabelServerPolicy()
