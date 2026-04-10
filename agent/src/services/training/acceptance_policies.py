"""Pseudo-label acceptance 정책."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from shared.src.domain.entities.training.pseudo_label_evidence import (
    PseudoLabelEvidence,
)


@dataclass(slots=True)
class AcceptanceDecision:
    """Pseudo-label acceptance 판단 결과."""

    label: str
    confidence: float
    confidence_kind: str | None
    margin: float
    accepted: bool
    runner_up_label: str | None = None
    runner_up_score: float | None = None
    sample_weight: float = 1.0


class PseudoLabelAcceptancePolicy(Protocol):
    """Evidence를 pseudo-label 후보로 해석하는 정책."""

    policy_name: str
    supported_adapter_kinds: tuple[str, ...]

    def evaluate(
        self,
        *,
        evidence: PseudoLabelEvidence,
        confidence_threshold: float,
        margin_threshold: float,
    ) -> AcceptanceDecision:
        """Evidence를 해석해 acceptance 결과를 만든다."""


AcceptancePolicyFactory = Callable[[], PseudoLabelAcceptancePolicy]


@dataclass(slots=True)
class Top1MarginThresholdAcceptancePolicy:
    """Top1 confidence와 top1-top2 margin을 함께 본다."""

    policy_name: str = "top1_margin_threshold"
    supported_adapter_kinds: tuple[str, ...] = ("*",)

    def evaluate(
        self,
        *,
        evidence: PseudoLabelEvidence,
        confidence_threshold: float,
        margin_threshold: float,
    ) -> AcceptanceDecision:
        return _build_acceptance_decision(
            evidence=evidence,
            accepted=(
                evidence.top1_score >= confidence_threshold
                and evidence.margin >= margin_threshold
            ),
        )


@dataclass(slots=True)
class Top1ConfidenceOnlyAcceptancePolicy:
    """Top1 confidence만으로 pseudo-label 채택 여부를 결정한다."""

    policy_name: str = "top1_confidence_only"
    supported_adapter_kinds: tuple[str, ...] = ("*",)

    def evaluate(
        self,
        *,
        evidence: PseudoLabelEvidence,
        confidence_threshold: float,
        margin_threshold: float,
    ) -> AcceptanceDecision:
        return _build_acceptance_decision(
            evidence=evidence,
            accepted=evidence.top1_score >= confidence_threshold,
        )


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


def _build_acceptance_decision(
    *,
    evidence: PseudoLabelEvidence,
    accepted: bool,
) -> AcceptanceDecision:
    return AcceptanceDecision(
        label=evidence.top1_label,
        confidence=evidence.top1_score,
        confidence_kind=evidence.confidence_kind,
        margin=evidence.margin,
        accepted=accepted,
        runner_up_label=evidence.top2_label,
        runner_up_score=evidence.top2_score,
        sample_weight=evidence.sample_weight,
    )


register_pseudo_label_acceptance_policy(
    "top1_margin_threshold",
    factory=Top1MarginThresholdAcceptancePolicy,
)
register_pseudo_label_acceptance_policy(
    "top1_confidence_only",
    factory=Top1ConfidenceOnlyAcceptancePolicy,
)
