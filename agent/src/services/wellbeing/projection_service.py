"""로컬 inference 결과를 wellbeing signal snapshot으로 투영한다."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from agent.src.contracts.wellbeing_signal_contracts import (
    WellbeingSignalConfidence,
    WellbeingSignalLevel,
    WellbeingSignalSummaryPayload,
    WellbeingSignalTrend,
)
from agent.src.infrastructure.repositories.analysis_event_repository import (
    AnalysisEventRepository,
)
from agent.src.infrastructure.repositories.wellbeing_snapshot_repository import (
    WellbeingSnapshotRepository,
)
from agent.src.services.inference.baseline_service import BaselineService
from agent.src.services.inference.decision_service import DecisionService
from shared.src.contracts.personalization_contracts import (
    PersonalizationState,
    PersonalizationWarmupStatus,
)
from shared.src.domain.entities.inference.result import AssessmentResult
from shared.src.domain.entities.inference.state import BaselineProfile, TimeSeriesState


@dataclass(slots=True)
class WellbeingSignalProjectionService:
    """기존 local inference rail을 wellbeing signal 출력으로 번역한다."""

    analysis_event_repository: AnalysisEventRepository
    snapshot_repository: WellbeingSnapshotRepository
    baseline_service: BaselineService = field(default_factory=BaselineService)
    decision_service: DecisionService = field(default_factory=DecisionService)
    lookback_days: int = 30
    _last_refresh_at: datetime | None = field(default=None, init=False)

    def refresh_from_runtime(self) -> None:
        """최근 analysis event를 replay해 wellbeing snapshot을 갱신한다."""

        analysis_events = sorted(
            self.analysis_event_repository.get_recent(days=self.lookback_days),
            key=lambda event: event.occurred_at,
        )
        if not analysis_events:
            return

        previous_state: TimeSeriesState | None = None
        history: list = []

        for analysis_event in analysis_events:
            baseline_profile = self.baseline_service.build_profile(
                history,
                as_of=analysis_event.occurred_at,
            )
            personalization_state = _build_personalization_state(
                baseline_profile=baseline_profile,
                updated_at=analysis_event.occurred_at,
            )
            evaluation = self.decision_service.evaluate(
                analysis_event=analysis_event,
                baseline_profile=baseline_profile,
                personalization_state=personalization_state,
                previous_state=previous_state,
                assessment_id=analysis_event.query_id,
            )
            payload = _translate_to_wellbeing_summary(
                assessment_result=evaluation.assessment_result,
                baseline_profile=baseline_profile,
                computed_at=analysis_event.occurred_at,
            )
            self.snapshot_repository.save_summary(payload)
            previous_state = evaluation.time_series_state
            history.append(analysis_event)

        self._last_refresh_at = datetime.now(tz=timezone.utc)


def _build_personalization_state(
    *,
    baseline_profile: BaselineProfile,
    updated_at: datetime,
) -> PersonalizationState:
    if baseline_profile.event_count == 0:
        warmup_status = PersonalizationWarmupStatus.COLD_START
    elif baseline_profile.warmup_complete:
        warmup_status = PersonalizationWarmupStatus.READY
    else:
        warmup_status = PersonalizationWarmupStatus.WARMING_UP

    return PersonalizationState(
        state_version="personalization_state.v1",
        warmup_status=warmup_status,
        updated_at=updated_at,
    )


def _translate_to_wellbeing_summary(
    *,
    assessment_result: AssessmentResult,
    baseline_profile: BaselineProfile,
    computed_at: datetime,
) -> WellbeingSignalSummaryPayload:
    signal_level, signal_label = _map_signal_level(assessment_result.decision)
    trend = _map_trend(assessment_result)
    confidence = _map_confidence(assessment_result.confidence)
    signal_score = _build_signal_score(
        decision=assessment_result.decision,
        global_score=assessment_result.global_score,
        baseline_shift=assessment_result.baseline_shift,
    )
    low_data = not baseline_profile.warmup_complete
    summary = _build_summary_text(signal_level=signal_level, low_data=low_data)
    action_tip = _build_action_tip(signal_level=signal_level, low_data=low_data)

    return WellbeingSignalSummaryPayload(
        computed_at=computed_at,
        signal_score=signal_score,
        signal_level=signal_level,
        signal_label=signal_label,
        trend=trend,
        summary=summary,
        action_tip=action_tip,
        confidence=confidence,
        low_data=low_data,
    )


def _map_signal_level(decision: str) -> tuple[WellbeingSignalLevel, str]:
    mapping = {
        "NORMAL": (WellbeingSignalLevel.LOW, "안정"),
        "WATCH": (WellbeingSignalLevel.MODERATE, "관찰 필요"),
        "SUPPORT": (WellbeingSignalLevel.HIGH, "주의 필요"),
        "RISK": (WellbeingSignalLevel.VERY_HIGH, "도움 필요"),
    }
    return mapping.get(
        decision,
        (WellbeingSignalLevel.MODERATE, "관찰 필요"),
    )


def _map_confidence(value: float | None) -> WellbeingSignalConfidence:
    if value is None:
        return WellbeingSignalConfidence.LOW
    if value >= 0.85:
        return WellbeingSignalConfidence.HIGH
    if value >= 0.6:
        return WellbeingSignalConfidence.MEDIUM
    return WellbeingSignalConfidence.LOW


def _map_trend(assessment_result: AssessmentResult) -> WellbeingSignalTrend:
    shift = assessment_result.baseline_shift
    persistence = assessment_result.persistence or 0.0
    if shift is None:
        return WellbeingSignalTrend.UNKNOWN
    if abs(shift) < 0.08:
        return WellbeingSignalTrend.STEADY
    if persistence < 1.0 and abs(shift) >= 0.12:
        return WellbeingSignalTrend.VOLATILE
    if shift > 0:
        return WellbeingSignalTrend.RISING
    return WellbeingSignalTrend.FALLING


def _build_signal_score(
    *,
    decision: str,
    global_score: float | None,
    baseline_shift: float | None,
) -> float:
    severity = min(
        max(((global_score or 0.0) + max(baseline_shift or 0.0, 0.0)) / 1.5, 0.0),
        1.0,
    )
    bands = {
        "NORMAL": (0.0, 34.0),
        "WATCH": (35.0, 54.0),
        "SUPPORT": (55.0, 79.0),
        "RISK": (80.0, 100.0),
    }
    lower, upper = bands.get(decision, (35.0, 54.0))
    return round(lower + (upper - lower) * severity, 2)


def _build_summary_text(
    *,
    signal_level: WellbeingSignalLevel,
    low_data: bool,
) -> str:
    if low_data:
        return (
            "최근 데이터가 아직 충분하지 않아 현재 상태를 보수적으로 해석하고 있습니다."
        )
    mapping = {
        WellbeingSignalLevel.LOW: (
            "최근 전체 상태가 비교적 안정적으로 유지되고 있습니다."
        ),
        WellbeingSignalLevel.MODERATE: (
            "최근 전체 상태에 평소와 다른 변화가 보여 조금 더 지켜볼 필요가 있습니다."
        ),
        WellbeingSignalLevel.HIGH: (
            "최근 전체 상태가 평소보다 높게 나타나 주의가 필요합니다."
        ),
        WellbeingSignalLevel.VERY_HIGH: (
            "최근 전체 상태가 높은 수준으로 나타나 빠른 확인과 대화가 필요합니다."
        ),
    }
    return mapping[signal_level]


def _build_action_tip(
    *,
    signal_level: WellbeingSignalLevel,
    low_data: bool,
) -> str:
    if low_data:
        return "조금 더 관찰하면서 짧은 안부 대화를 먼저 시도해 보세요."
    mapping = {
        WellbeingSignalLevel.LOW: (
            "지금처럼 짧은 안부 확인과 안정적인 루틴을 유지해 보세요."
        ),
        WellbeingSignalLevel.MODERATE: (
            "오늘은 짧게 상태를 묻고 저녁에 한 번 더 확인해 보세요."
        ),
        WellbeingSignalLevel.HIGH: (
            "오늘은 사용 시간을 줄이고 바로 짧은 대화를 시도해 보세요."
        ),
        WellbeingSignalLevel.VERY_HIGH: (
            "혼자 두지 말고 바로 대화를 시도하며 가까운 보호자가 함께 확인해 주세요."
        ),
    }
    return mapping[signal_level]
