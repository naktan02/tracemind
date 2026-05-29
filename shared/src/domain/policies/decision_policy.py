"""규칙 기반 로컬 판단 정책."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from shared.src.domain.entities.inference.events import ScoredEvent
from shared.src.domain.entities.inference.result import AssessmentResult
from shared.src.domain.entities.inference.state import (
    BaselineProfile,
    PersonalizationState,
    TimeSeriesState,
)


class DecisionLevel(StrEnum):
    """v1 로컬 판단 레벨."""

    NORMAL = "NORMAL"
    WATCH = "WATCH"
    SUPPORT = "SUPPORT"
    RISK = "RISK"


@dataclass(slots=True)
class RuleBasedDecisionPolicy:
    """score, baseline, persistence를 결합해 최종 판단을 내린다."""

    score_floor: float = 0.55
    default_delta_threshold: float = 0.2
    support_delta_multiplier: float = 1.5
    risk_delta_multiplier: float = 2.0
    watch_streak: int = 1
    support_streak: int = 2
    risk_streak: int = 3
    warmup_max_level: DecisionLevel = DecisionLevel.WATCH
    ignored_categories: frozenset[str] = field(
        default_factory=lambda: frozenset({"normal"})
    )

    def evaluate(
        self,
        *,
        scored_event: ScoredEvent,
        baseline_profile: BaselineProfile,
        personalization_state: PersonalizationState,
        time_series_state: TimeSeriesState,
        assessment_id: str,
    ) -> AssessmentResult:
        best_level = DecisionLevel.NORMAL
        best_category: str | None = None
        best_score = 0.0
        best_delta = 0.0
        best_persistence = 0.0

        categories = sorted(scored_event.category_scores)
        for category in categories:
            if category in self.ignored_categories:
                continue

            score = time_series_state.latest_scores.get(
                category,
                scored_event.category_scores.get(category, 0.0),
            )
            delta = time_series_state.latest_deltas.get(
                category,
                score - baseline_profile.category_means.get(category, 0.0),
            )
            ewma_delta = time_series_state.ewma_deltas.get(category, delta)
            streak = time_series_state.elevated_streaks.get(category, 0)

            threshold = personalization_state.threshold_by_category.get(
                category,
                self.default_delta_threshold,
            )
            level = self._classify_level(
                score=score,
                delta=delta,
                ewma_delta=ewma_delta,
                streak=streak,
                threshold=threshold,
            )

            if self._level_rank(level) > self._level_rank(best_level) or (
                level == best_level and (delta, score) > (best_delta, best_score)
            ):
                best_level = level
                best_category = category
                best_score = score
                best_delta = delta
                best_persistence = float(streak)

        if not baseline_profile.warmup_complete:
            best_level = self._cap_level(best_level, self.warmup_max_level)

        explanation = self._build_explanation(
            category=best_category,
            level=best_level,
            score=best_score,
            delta=best_delta,
            persistence=best_persistence,
            warmup_complete=baseline_profile.warmup_complete,
        )

        return AssessmentResult(
            schema_version="assessment_result.v1",
            assessment_id=assessment_id,
            decision=best_level.value,
            explanation=explanation,
            global_score=best_score if best_category is not None else None,
            personalized_score=best_delta if best_category is not None else None,
            baseline_shift=best_delta if best_category is not None else None,
            persistence=best_persistence if best_category is not None else None,
            confidence=self._confidence_from_level(best_level),
            focus_category=best_category,
        )

    def _classify_level(
        self,
        *,
        score: float,
        delta: float,
        ewma_delta: float,
        streak: int,
        threshold: float,
    ) -> DecisionLevel:
        if score < self.score_floor or delta < threshold:
            return DecisionLevel.NORMAL
        if (
            delta >= threshold * self.risk_delta_multiplier
            and ewma_delta >= threshold * self.support_delta_multiplier
            and streak >= self.risk_streak
        ):
            return DecisionLevel.RISK
        if (
            delta >= threshold * self.support_delta_multiplier
            and ewma_delta >= threshold
            and streak >= self.support_streak
        ):
            return DecisionLevel.SUPPORT
        if streak >= self.watch_streak:
            return DecisionLevel.WATCH
        return DecisionLevel.NORMAL

    @staticmethod
    def _level_rank(level: DecisionLevel) -> int:
        ranks = {
            DecisionLevel.NORMAL: 0,
            DecisionLevel.WATCH: 1,
            DecisionLevel.SUPPORT: 2,
            DecisionLevel.RISK: 3,
        }
        return ranks[level]

    def _cap_level(
        self,
        current: DecisionLevel,
        maximum: DecisionLevel,
    ) -> DecisionLevel:
        return (
            current
            if self._level_rank(current) <= self._level_rank(maximum)
            else maximum
        )

    @staticmethod
    def _confidence_from_level(level: DecisionLevel) -> float:
        mapping = {
            DecisionLevel.NORMAL: 0.5,
            DecisionLevel.WATCH: 0.65,
            DecisionLevel.SUPPORT: 0.8,
            DecisionLevel.RISK: 0.92,
        }
        return mapping[level]

    @staticmethod
    def _build_explanation(
        *,
        category: str | None,
        level: DecisionLevel,
        score: float,
        delta: float,
        persistence: float,
        warmup_complete: bool,
    ) -> str:
        if category is None:
            if warmup_complete:
                return "No category crossed the personalized decision threshold."
            return (
                "Warm-up is not complete and no category crossed the watch threshold."
            )
        warmup_note = (
            ""
            if warmup_complete
            else " Warm-up is incomplete, so escalation is capped."
        )
        return (
            f"{category} reached {level.value} with score={score:.3f}, "
            f"delta={delta:.3f}, persistence={persistence:.0f}.{warmup_note}"
        )
