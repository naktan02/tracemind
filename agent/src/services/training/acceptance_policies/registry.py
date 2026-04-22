"""Acceptance policy registry."""

from __future__ import annotations

from collections.abc import Callable

from .base import PseudoLabelAcceptancePolicy
from .top1 import (
    Top1ConfidenceOnlyAcceptancePolicy,
    Top1MarginThresholdAcceptancePolicy,
)

AcceptancePolicyFactory = Callable[[], PseudoLabelAcceptancePolicy]

_ACCEPTANCE_POLICY_REGISTRY: dict[str, AcceptancePolicyFactory] = {}


def register_pseudo_label_acceptance_policy(
    *policy_names: str,
    factory: AcceptancePolicyFactory,
) -> None:
    """얇은 wiring registry에 pseudo-label acceptance policy를 등록한다."""

    for policy_name in policy_names:
        _ACCEPTANCE_POLICY_REGISTRY[policy_name.strip().lower()] = factory


def build_pseudo_label_acceptance_policy(
    policy_name: str,
) -> PseudoLabelAcceptancePolicy:
    """정책 이름으로 acceptance policy를 생성한다."""

    normalized_name = policy_name.strip().lower()
    factory = _ACCEPTANCE_POLICY_REGISTRY.get(normalized_name)
    if factory is not None:
        return factory()
    raise ValueError(f"Unsupported pseudo-label acceptance policy: {policy_name}.")


def list_registered_pseudo_label_acceptance_policy_names() -> tuple[str, ...]:
    """등록된 acceptance policy 이름을 정렬된 tuple로 반환한다."""

    return tuple(sorted(_ACCEPTANCE_POLICY_REGISTRY))


register_pseudo_label_acceptance_policy(
    "top1_margin_threshold",
    factory=Top1MarginThresholdAcceptancePolicy,
)
register_pseudo_label_acceptance_policy(
    "top1_confidence_only",
    factory=Top1ConfidenceOnlyAcceptancePolicy,
)


__all__ = [
    "AcceptancePolicyFactory",
    "build_pseudo_label_acceptance_policy",
    "list_registered_pseudo_label_acceptance_policy_names",
    "register_pseudo_label_acceptance_policy",
]
