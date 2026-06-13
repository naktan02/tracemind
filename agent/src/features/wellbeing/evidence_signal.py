"""Wellbeing 판단에 쓰는 agent-local evidence signal."""

from __future__ import annotations

from dataclasses import dataclass

from shared.src.domain.entities.inference.events import AnalysisEvent

DIRECT_RISK_REASON = "direct_self_harm_or_suicide_expression"
_DIRECT_RISK_PHRASES = (
    "죽고 싶",
    "죽고싶",
    "자살",
    "자해",
    "극단",
    "suicide",
    "self harm",
    "selfharm",
    "kill myself",
)
_NORMAL_CATEGORY = "normal"


@dataclass(frozen=True, slots=True)
class WellbeingEvidenceSignal:
    """분석 점수와 원문 근거를 합친 wellbeing 판단 입력."""

    max_non_normal_score: float
    direct_risk: bool = False
    reason: str | None = None


def build_wellbeing_evidence_signal(
    *,
    analysis_event: AnalysisEvent,
    source_text: str,
) -> WellbeingEvidenceSignal:
    """analysis event와 agent-local 원문에서 canonical evidence signal을 만든다."""

    direct_risk = contains_direct_risk_expression(source_text)
    return WellbeingEvidenceSignal(
        max_non_normal_score=_max_non_normal_score(analysis_event.category_scores),
        direct_risk=direct_risk,
        reason=DIRECT_RISK_REASON if direct_risk else None,
    )


def contains_direct_risk_expression(text: str) -> bool:
    """직접적인 자해/자살 표현이 있는지 확인한다."""

    lowered = text.lower()
    return any(phrase in lowered for phrase in _DIRECT_RISK_PHRASES)


def _max_non_normal_score(category_scores: dict[str, float]) -> float:
    return max(
        (
            float(score)
            for category, score in category_scores.items()
            if category != _NORMAL_CATEGORY
        ),
        default=0.0,
    )
