"""Pseudo-label acceptance policy package."""

from .base import AcceptanceDecision, PseudoLabelAcceptancePolicy
from .registry import (
    build_pseudo_label_acceptance_policy,
    register_pseudo_label_acceptance_policy,
)
from .top1 import (
    Top1ConfidenceOnlyAcceptancePolicy,
    Top1MarginThresholdAcceptancePolicy,
)

__all__ = [
    "AcceptanceDecision",
    "PseudoLabelAcceptancePolicy",
    "Top1ConfidenceOnlyAcceptancePolicy",
    "Top1MarginThresholdAcceptancePolicy",
    "build_pseudo_label_acceptance_policy",
    "register_pseudo_label_acceptance_policy",
]
