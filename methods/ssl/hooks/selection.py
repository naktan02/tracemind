"""Pseudo-label evidence selection hooks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from shared.src.domain.entities.training.pseudo_label_evidence import (
    PseudoLabelEvidence,
)

from .registry import register_pseudo_label_selection_hook


@dataclass(frozen=True, slots=True)
class PseudoLabelSelectionConfig:
    """Selection hook이 해석할 method-owned parameter 입력."""

    parameters: Mapping[str, float] = field(default_factory=dict)

    def require_float(self, key: str) -> float:
        """hook이 필요한 숫자 parameter를 명시적으로 요구한다."""

        value = self.parameters.get(key)
        if value is None:
            raise ValueError(f"Selection hook parameter is required: {key}")
        return float(value)


@dataclass(frozen=True, slots=True)
class PseudoLabelSelectionDecision:
    """순수 selection hook의 판단 결과."""

    label: str
    confidence: float
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
        margin=evidence.margin,
        accepted=accepted,
        runner_up_label=evidence.top2_label,
        runner_up_score=evidence.top2_score,
        sample_weight=evidence.sample_weight,
    )


@register_pseudo_label_selection_hook("top1_confidence_only")
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
        confidence_threshold = config.require_float("confidence_threshold")
        return build_pseudo_label_selection_decision(
            evidence=evidence,
            accepted=evidence.top1_score >= confidence_threshold,
        )


@register_pseudo_label_selection_hook("top1_margin_threshold")
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
        confidence_threshold = config.require_float("confidence_threshold")
        margin_threshold = config.require_float("margin_threshold")
        return build_pseudo_label_selection_decision(
            evidence=evidence,
            accepted=(
                evidence.top1_score >= confidence_threshold
                and evidence.margin >= margin_threshold
            ),
        )


@register_pseudo_label_selection_hook("top1_ranked")
@dataclass(slots=True)
class Top1RankedPseudoLabelSelectionHook:
    """Top1 evidence를 모두 후보로 열고 cap/ranking 정책에 선택을 맡긴다."""

    hook_name: str = "top1_ranked"

    def evaluate(
        self,
        *,
        evidence: PseudoLabelEvidence,
        config: PseudoLabelSelectionConfig,
    ) -> PseudoLabelSelectionDecision:
        _ = config
        return build_pseudo_label_selection_decision(
            evidence=evidence,
            accepted=True,
        )
