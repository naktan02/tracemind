"""Acceptance policy shared helpers."""

from __future__ import annotations

from methods.ssl.pseudo_label_selection.base import PseudoLabelSelectionDecision
from shared.src.domain.entities.training.pseudo_label_evidence import (
    PseudoLabelEvidence,
)

from .base import AcceptanceDecision


def build_acceptance_decision(
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


def acceptance_decision_from_selection_decision(
    decision: PseudoLabelSelectionDecision,
) -> AcceptanceDecision:
    """methods selection decision을 agent-local acceptance decision으로 감싼다."""

    return AcceptanceDecision(
        label=decision.label,
        confidence=decision.confidence,
        confidence_kind=decision.confidence_kind,
        margin=decision.margin,
        accepted=decision.accepted,
        runner_up_label=decision.runner_up_label,
        runner_up_score=decision.runner_up_score,
        sample_weight=decision.sample_weight,
    )
