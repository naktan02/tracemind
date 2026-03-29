"""로컬 판단 orchestration 서비스."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from agent.src.services.inference.time_series_service import TimeSeriesAccumulator
from shared.src.domain.entities.inference.events import ScoredEvent
from shared.src.domain.entities.inference.result import AssessmentResult
from shared.src.domain.entities.inference.state import (
    BaselineProfile,
    PersonalizationState,
    TimeSeriesState,
)
from shared.src.domain.policies.decision_policy import RuleBasedDecisionPolicy


@dataclass(slots=True)
class DecisionEvaluation:
    """최종 판단과 갱신된 시계열 상태를 함께 반환한다."""

    assessment_result: AssessmentResult
    time_series_state: TimeSeriesState


@dataclass(slots=True)
class DecisionService:
    """개인화 상태, 시계열 누적, 정책을 결합해 최종 판단을 만든다."""

    policy_version: str = "bootstrap"
    accumulator: TimeSeriesAccumulator = field(default_factory=TimeSeriesAccumulator)
    policy: RuleBasedDecisionPolicy = field(default_factory=RuleBasedDecisionPolicy)

    def evaluate(
        self,
        *,
        scored_event: ScoredEvent,
        baseline_profile: BaselineProfile,
        personalization_state: PersonalizationState,
        previous_state: TimeSeriesState | None = None,
        assessment_id: str | None = None,
    ) -> DecisionEvaluation:
        time_series_state = self.accumulator.update(
            scored_event=scored_event,
            baseline_profile=baseline_profile,
            personalization_state=personalization_state,
            previous_state=previous_state,
        )
        result = self.policy.evaluate(
            scored_event=scored_event,
            baseline_profile=baseline_profile,
            personalization_state=personalization_state,
            time_series_state=time_series_state,
            assessment_id=assessment_id or str(uuid4()),
        )
        return DecisionEvaluation(
            assessment_result=result,
            time_series_state=time_series_state,
        )
