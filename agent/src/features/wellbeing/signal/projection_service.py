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
from agent.src.features.captured_text.storage.records import CapturedTextRecord
from agent.src.features.captured_text.storage.repository import CapturedTextRepository
from agent.src.features.inference.interpretation.baseline import BaselineService
from agent.src.features.inference.interpretation.decision import DecisionService
from agent.src.features.inference.interpretation.result import AssessmentResult
from agent.src.features.inference.interpretation.state import (
    BaselineProfile,
    TimeSeriesState,
)
from agent.src.features.wellbeing.evidence_signal import (
    WellbeingEvidenceSignal,
    build_wellbeing_evidence_signal,
)
from agent.src.features.wellbeing.storage.wellbeing_snapshot_repository import (
    WellbeingSnapshotRepository,
)
from agent.src.infrastructure.repositories.analysis_event_repository import (
    AnalysisEventRepository,
)
from shared.src.domain.entities.inference.events import AnalysisEvent

WELLBEING_PROJECTION_VERSION = "wellbeing_projection.evidence_signal.v1"


@dataclass(slots=True)
class WellbeingSignalProjectionService:
    """기존 local inference rail을 wellbeing signal 출력으로 번역한다."""

    analysis_event_repository: AnalysisEventRepository
    snapshot_repository: WellbeingSnapshotRepository
    captured_text_repository: CapturedTextRepository | None = None
    baseline_service: BaselineService = field(default_factory=BaselineService)
    decision_service: DecisionService = field(default_factory=DecisionService)
    lookback_days: int = 30
    _last_refresh_at: datetime | None = field(default=None, init=False)

    def refresh_from_runtime(self) -> None:
        """최근 analysis event를 replay해 wellbeing snapshot을 갱신한다."""

        if self._snapshots_are_current():
            return

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
            evaluation = self.decision_service.evaluate(
                analysis_event=analysis_event,
                baseline_profile=baseline_profile,
                previous_state=previous_state,
                assessment_id=analysis_event.query_id,
            )
            evidence_signal = build_wellbeing_evidence_signal(
                analysis_event=analysis_event,
                source_text=self._load_source_text(analysis_event),
            )
            payload = _translate_to_wellbeing_summary(
                assessment_result=evaluation.assessment_result,
                baseline_profile=baseline_profile,
                evidence_signal=evidence_signal,
                computed_at=analysis_event.occurred_at,
            )
            self.snapshot_repository.save_summary(
                payload,
                projection_version=WELLBEING_PROJECTION_VERSION,
            )
            previous_state = evaluation.time_series_state
            history.append(analysis_event)

        self._last_refresh_at = datetime.now(tz=timezone.utc)

    def _snapshots_are_current(self) -> bool:
        latest_analysis_at = self.analysis_event_repository.load_latest_occurred_at()
        if latest_analysis_at is None:
            return True
        latest_snapshot = self.snapshot_repository.load_latest_summary()
        if latest_snapshot is None:
            return False
        if (
            self.snapshot_repository.load_latest_projection_version()
            != WELLBEING_PROJECTION_VERSION
        ):
            return False
        return latest_snapshot.computed_at >= latest_analysis_at

    def _load_source_text(self, analysis_event: AnalysisEvent) -> str:
        if self.captured_text_repository is not None:
            try:
                record = self.captured_text_repository.get(
                    analysis_event.source_event_id or analysis_event.query_id
                )
            except Exception:
                record = None
            if isinstance(record, CapturedTextRecord):
                return record.text
        return analysis_event.translated_text or ""


def _translate_to_wellbeing_summary(
    *,
    assessment_result: AssessmentResult,
    baseline_profile: BaselineProfile,
    evidence_signal: WellbeingEvidenceSignal,
    computed_at: datetime,
) -> WellbeingSignalSummaryPayload:
    signal_level, signal_label = _map_signal_level(
        assessment_result.decision,
        evidence_signal=evidence_signal,
    )
    trend = _map_trend(assessment_result)
    confidence = _map_confidence(assessment_result.confidence)
    signal_score = _build_signal_score(
        decision=assessment_result.decision,
        global_score=assessment_result.global_score,
        baseline_shift=assessment_result.baseline_shift,
        evidence_signal=evidence_signal,
    )
    low_data = not baseline_profile.warmup_complete and not evidence_signal.direct_risk
    summary = _build_summary_text(
        signal_level=signal_level,
        low_data=low_data,
        evidence_signal=evidence_signal,
    )
    action_tip = _build_action_tip(
        signal_level=signal_level,
        low_data=low_data,
        evidence_signal=evidence_signal,
    )

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


def _map_signal_level(
    decision: str,
    *,
    evidence_signal: WellbeingEvidenceSignal,
) -> tuple[WellbeingSignalLevel, str]:
    if evidence_signal.direct_risk:
        return WellbeingSignalLevel.VERY_HIGH, "도움 필요"
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
    evidence_signal: WellbeingEvidenceSignal,
) -> float:
    if evidence_signal.direct_risk:
        return 95.0
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
    evidence_signal: WellbeingEvidenceSignal,
) -> str:
    if evidence_signal.direct_risk:
        return (
            "최근 입력에서 직접적인 자해나 자살 관련 표현이 확인되어 "
            "빠른 확인이 필요합니다."
        )
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
    evidence_signal: WellbeingEvidenceSignal,
) -> str:
    if evidence_signal.direct_risk:
        return (
            "혼자 두지 말고 바로 가까운 보호자나 믿을 수 있는 어른과 "
            "함께 확인해 주세요."
        )
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
