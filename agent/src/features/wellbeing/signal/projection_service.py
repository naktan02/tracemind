"""로컬 inference 결과를 wellbeing signal snapshot으로 투영한다."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from agent.src.contracts.wellbeing_signal_contracts import (
    ParentWellbeingGuidancePayload,
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
    RECENT_DIRECT_RISK_REASON,
    WellbeingEvidenceSignal,
    build_wellbeing_evidence_signal,
    contains_direct_risk_expression,
)
from agent.src.features.wellbeing.storage.wellbeing_snapshot_repository import (
    WellbeingSnapshotRepository,
)
from agent.src.infrastructure.repositories.analysis_event_repository import (
    AnalysisEventRepository,
)
from shared.src.domain.entities.inference.events import AnalysisEvent

WELLBEING_PROJECTION_VERSION = "wellbeing_projection.evidence_signal.v2"
DIRECT_RISK_PERSISTENCE_WINDOW = timedelta(hours=2)


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
        direct_risk_active_until: datetime | None = None

        for analysis_event in analysis_events:
            source_text = self._load_source_text(analysis_event)
            has_direct_risk_text = contains_direct_risk_expression(source_text)
            recent_direct_risk_context_active = (
                direct_risk_active_until is not None
                and analysis_event.occurred_at <= direct_risk_active_until
            )
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
                source_text=source_text,
                recent_direct_risk_context_active=(
                    recent_direct_risk_context_active and not has_direct_risk_text
                ),
            )
            if has_direct_risk_text:
                direct_risk_active_until = (
                    analysis_event.occurred_at + DIRECT_RISK_PERSISTENCE_WINDOW
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
    parent_guidance = _build_parent_guidance(
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
        parent_guidance=parent_guidance,
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
        if evidence_signal.reason == RECENT_DIRECT_RISK_REASON:
            return (
                "최근 직접적인 자해나 자살 관련 표현 이후 상태를 보수적으로 "
                "확인하고 있습니다."
            )
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
        if evidence_signal.reason == RECENT_DIRECT_RISK_REASON:
            return (
                "최근 위험 신호가 바로 낮아졌다고 단정하지 말고, 지금 상태를 "
                "직접 확인해 주세요."
            )
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


def _build_parent_guidance(
    *,
    signal_level: WellbeingSignalLevel,
    low_data: bool,
    evidence_signal: WellbeingEvidenceSignal,
) -> ParentWellbeingGuidancePayload:
    if evidence_signal.direct_risk:
        if evidence_signal.reason == RECENT_DIRECT_RISK_REASON:
            return ParentWellbeingGuidancePayload(
                response_priority=(
                    "최근 위험 신호가 있었으므로 지금 상태를 직접 확인하세요."
                ),
                conversation_starter=(
                    "지금 혼자 있는지, 바로 함께 있어도 되는지부터 차분히 물어보세요."
                ),
                caution_note=(
                    "최근 표현이 줄었다고 바로 안정으로 보지 말고 "
                    "가까운 어른과 함께 확인하세요."
                ),
            )
        return ParentWellbeingGuidancePayload(
            response_priority="혼자 두지 말고 즉시 가까운 보호자가 함께 확인하세요.",
            conversation_starter=(
                "지금 안전한 곳에 있는지, 곁에 있어도 되는지부터 짧게 물어보세요."
            ),
            caution_note=(
                "훈계나 추궁보다 안전 확인을 우선하고 필요하면 "
                "109 또는 1388에 도움을 요청하세요."
            ),
        )
    if low_data:
        return ParentWellbeingGuidancePayload(
            response_priority="판단을 서두르지 말고 짧은 안부 확인부터 시작하세요.",
            conversation_starter=(
                "오늘 하루 중 불편했던 순간이 있었는지 가볍게 물어보세요."
            ),
            caution_note=(
                "데이터가 적으므로 상태를 단정하지 말고 평소 변화와 함께 보세요."
            ),
        )
    mapping = {
        WellbeingSignalLevel.LOW: ParentWellbeingGuidancePayload(
            response_priority="평소처럼 짧은 안부 확인과 안정적인 루틴을 유지하세요.",
            conversation_starter="오늘 괜찮았던 일과 힘들었던 일을 하나씩 물어보세요.",
            caution_note=(
                "문제가 없다고 단정하지 말고 부담 없는 대화 기회를 유지하세요."
            ),
        ),
        WellbeingSignalLevel.MODERATE: ParentWellbeingGuidancePayload(
            response_priority="오늘 한 번은 직접 상태를 묻고 저녁에 다시 확인하세요.",
            conversation_starter=(
                "요즘 평소와 다르게 신경 쓰이는 일이 있는지 물어보세요."
            ),
            caution_note="답을 재촉하지 말고 아이가 말한 감정을 먼저 확인해 주세요.",
        ),
        WellbeingSignalLevel.HIGH: ParentWellbeingGuidancePayload(
            response_priority=(
                "오늘 안에 바로 짧은 대화를 시도하고 혼자 두는 시간을 줄이세요."
            ),
            conversation_starter=(
                "지금 가장 힘든 일이 무엇인지, "
                "같이 줄일 수 있는 부담이 있는지 물어보세요."
            ),
            caution_note=(
                "해결책을 먼저 제시하기보다 안전과 휴식이 필요한지 먼저 확인하세요."
            ),
        ),
        WellbeingSignalLevel.VERY_HIGH: ParentWellbeingGuidancePayload(
            response_priority="빠르게 직접 확인하고 가까운 보호자가 함께 있어 주세요.",
            conversation_starter=(
                "지금 안전한지, 혼자 있기 어렵지는 않은지부터 물어보세요."
            ),
            caution_note=(
                "높은 신호가 반복되면 학교 상담실, 1388, 109 같은 "
                "도움 경로를 함께 열어두세요."
            ),
        ),
    }
    return mapping[signal_level]
