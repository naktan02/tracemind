"""Pseudo-label evidence selection hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from shared.src.domain.entities.training.pseudo_label_evidence import (
    PseudoLabelEvidence,
)


@dataclass(frozen=True, slots=True)
class PseudoLabelSelectionConfig:
    """Selection hook이 해석할 threshold 입력."""

    confidence_threshold: float
    margin_threshold: float = 0.0


@dataclass(frozen=True, slots=True)
class PseudoLabelSelectionDecision:
    """순수 selection hook의 판단 결과."""

    label: str
    confidence: float
    confidence_kind: str | None
    margin: float
    accepted: bool
    runner_up_label: str | None = None
    runner_up_score: float | None = None
    sample_weight: float = 1.0


class PseudoLabelSelectionHook(Protocol):
    """중앙/FL SSL에서 재사용하는 pseudo-label selection hook."""

    hook_name: str

    def evaluate(
        self,
        *,
        evidence: PseudoLabelEvidence,
        config: PseudoLabelSelectionConfig,
    ) -> PseudoLabelSelectionDecision:
        """Evidence 하나를 selection decision 하나로 해석한다."""


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


@dataclass(slots=True)
class FixedConfidencePseudoLabelSelectionHook:
    """Top1 confidence threshold만 보는 selection hook."""

    hook_name: str = "top1_confidence_only"

    def evaluate(
        self,
        *,
        evidence: PseudoLabelEvidence,
        config: PseudoLabelSelectionConfig,
    ) -> PseudoLabelSelectionDecision:
        return build_pseudo_label_selection_decision(
            evidence=evidence,
            accepted=evidence.top1_score >= config.confidence_threshold,
        )


@dataclass(slots=True)
class MarginThresholdPseudoLabelSelectionHook:
    """Top1 confidence와 top1-top2 margin을 함께 보는 selection hook."""

    hook_name: str = "top1_margin_threshold"

    def evaluate(
        self,
        *,
        evidence: PseudoLabelEvidence,
        config: PseudoLabelSelectionConfig,
    ) -> PseudoLabelSelectionDecision:
        return build_pseudo_label_selection_decision(
            evidence=evidence,
            accepted=(
                evidence.top1_score >= config.confidence_threshold
                and evidence.margin >= config.margin_threshold
            ),
        )
