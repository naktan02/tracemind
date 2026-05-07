"""FedAvg pseudo-label round policy seam."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FedAvgPseudoLabelRoundPolicy:
    """현재는 active pair가 있는 client update만 aggregate하는 기본 round policy."""

    policy_name: str = "round_active_pair_only"
    custom_round_policy_required: bool = False


FEDAVG_PSEUDO_LABEL_ROUND_POLICY = FedAvgPseudoLabelRoundPolicy()
