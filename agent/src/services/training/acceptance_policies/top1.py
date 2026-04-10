"""Top1-based acceptance policies."""

from __future__ import annotations

from dataclasses import dataclass

from shared.src.domain.entities.training.pseudo_label_evidence import (
    PseudoLabelEvidence,
)

from .base import AcceptanceDecision
from .helpers import build_acceptance_decision


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
        return build_acceptance_decision(
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
        del margin_threshold
        return build_acceptance_decision(
            evidence=evidence,
            accepted=evidence.top1_score >= confidence_threshold,
        )


__all__ = [
    "Top1ConfidenceOnlyAcceptancePolicy",
    "Top1MarginThresholdAcceptancePolicy",
]
