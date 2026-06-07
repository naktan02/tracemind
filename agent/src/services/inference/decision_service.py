"""로컬 판단 orchestration 서비스."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from agent.src.services.inference.decision_policy import RuleBasedDecisionPolicy
from agent.src.services.inference.result import AssessmentResult
from agent.src.services.inference.state import (
    BaselineProfile,
    TimeSeriesState,
)
from agent.src.services.inference.time_series_service import TimeSeriesAccumulator
from shared.src.domain.entities.inference.events import AnalysisEvent


@dataclass(slots=True)
class DecisionEvaluation:
    """최종 판단과 갱신된 시계열 상태를 함께 반환한다."""

    assessment_result: AssessmentResult
    time_series_state: TimeSeriesState


@dataclass(slots=True)
class DecisionService:
    """시계열 누적과 agent-local rule을 결합해 최종 판단을 만든다."""

    policy_version: str = "bootstrap"
    accumulator: TimeSeriesAccumulator = field(default_factory=TimeSeriesAccumulator)
    policy: RuleBasedDecisionPolicy = field(default_factory=RuleBasedDecisionPolicy)

    def evaluate(
        self,
        *,
        analysis_event: AnalysisEvent,
        baseline_profile: BaselineProfile,
        previous_state: TimeSeriesState | None = None,
        assessment_id: str | None = None,
    ) -> DecisionEvaluation:
        time_series_state = self.accumulator.update(
            analysis_event=analysis_event,
            baseline_profile=baseline_profile,
            previous_state=previous_state,
        )
        result = self.policy.evaluate(
            analysis_event=analysis_event,
            baseline_profile=baseline_profile,
            time_series_state=time_series_state,
            assessment_id=assessment_id or str(uuid4()),
        )
        return DecisionEvaluation(
            assessment_result=result,
            time_series_state=time_series_state,
        )
