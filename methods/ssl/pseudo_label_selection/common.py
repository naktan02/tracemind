"""Pseudo-label selection 공통 helper."""

from __future__ import annotations

from methods.ssl.pseudo_label_selection.base import PseudoLabelSelectionDecision
from shared.src.domain.entities.training.pseudo_label_evidence import (
    PseudoLabelEvidence,
)


def build_pseudo_label_selection_decision(
    *,
    evidence: PseudoLabelEvidence,
    accepted: bool,
) -> PseudoLabelSelectionDecision:
    """Evidence의 top1 정보를 selection decision으로 정규화한다."""

    return PseudoLabelSelectionDecision(
        label=evidence.top1_label,
        confidence=evidence.top1_score,
        confidence_kind=evidence.confidence_kind,
        margin=evidence.margin,
        accepted=accepted,
        runner_up_label=evidence.top2_label,
        runner_up_score=evidence.top2_score,
        sample_weight=evidence.sample_weight,
    )
