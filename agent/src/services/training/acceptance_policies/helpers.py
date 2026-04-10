"""Acceptance policy shared helpers."""

from __future__ import annotations

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


__all__ = ["build_acceptance_decision"]
