"""Pseudo-label acceptance 정책."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class AcceptanceDecision:
    """Pseudo-label acceptance 판단 결과."""

    label: str
    confidence: float
    margin: float
    accepted: bool
    runner_up_label: str | None = None
    runner_up_score: float | None = None


class PseudoLabelAcceptancePolicy(Protocol):
    """카테고리 score를 pseudo-label 후보로 해석하는 정책."""

    policy_name: str

    def evaluate(
        self,
        *,
        category_scores: Mapping[str, float],
        confidence_threshold: float,
        margin_threshold: float,
    ) -> AcceptanceDecision:
        """Score dict를 해석해 acceptance 결과를 만든다."""


@dataclass(slots=True)
class Top1MarginThresholdAcceptancePolicy:
    """Top1 confidence와 top1-top2 margin을 함께 본다."""

    policy_name: str = "top1_margin_threshold"

    def evaluate(
        self,
        *,
        category_scores: Mapping[str, float],
        confidence_threshold: float,
        margin_threshold: float,
    ) -> AcceptanceDecision:
        ranked_scores = _rank_category_scores(category_scores)
        top_label, top_score = ranked_scores[0]
        if len(ranked_scores) > 1:
            runner_up_label, runner_up_score = ranked_scores[1]
        else:
            runner_up_label, runner_up_score = None, 0.0
        margin = top_score - runner_up_score
        return AcceptanceDecision(
            label=top_label,
            confidence=top_score,
            margin=margin,
            accepted=(
                top_score >= confidence_threshold and margin >= margin_threshold
            ),
            runner_up_label=runner_up_label,
            runner_up_score=runner_up_score,
        )


@dataclass(slots=True)
class Top1ConfidenceOnlyAcceptancePolicy:
    """Top1 confidence만으로 pseudo-label 채택 여부를 결정한다."""

    policy_name: str = "top1_confidence_only"

    def evaluate(
        self,
        *,
        category_scores: Mapping[str, float],
        confidence_threshold: float,
        margin_threshold: float,
    ) -> AcceptanceDecision:
        ranked_scores = _rank_category_scores(category_scores)
        top_label, top_score = ranked_scores[0]
        if len(ranked_scores) > 1:
            runner_up_label, runner_up_score = ranked_scores[1]
        else:
            runner_up_label, runner_up_score = None, 0.0
        margin = top_score - runner_up_score
        return AcceptanceDecision(
            label=top_label,
            confidence=top_score,
            margin=margin,
            accepted=top_score >= confidence_threshold,
            runner_up_label=runner_up_label,
            runner_up_score=runner_up_score,
        )


def build_pseudo_label_acceptance_policy(
    policy_name: str,
) -> PseudoLabelAcceptancePolicy:
    """정책 이름으로 acceptance policy를 생성한다."""

    normalized_name = policy_name.strip().lower()
    if normalized_name == "top1_margin_threshold":
        return Top1MarginThresholdAcceptancePolicy()
    if normalized_name == "top1_confidence_only":
        return Top1ConfidenceOnlyAcceptancePolicy()
    raise ValueError(f"Unsupported pseudo-label acceptance policy: {policy_name}.")


def _rank_category_scores(
    category_scores: Mapping[str, float],
) -> list[tuple[str, float]]:
    ranked_scores = sorted(
        category_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    if not ranked_scores:
        raise ValueError("ScoredEvent must contain at least one category score.")
    return ranked_scores
