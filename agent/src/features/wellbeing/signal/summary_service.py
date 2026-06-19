"""현재 wellbeing signal summary를 제공하는 service."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from agent.src.contracts.wellbeing_signal_contracts import (
    ParentWellbeingGuidancePayload,
    WellbeingSignalConfidence,
    WellbeingSignalLevel,
    WellbeingSignalSummaryPayload,
    WellbeingSignalTrend,
)
from agent.src.features.wellbeing.signal.projection_service import (
    WellbeingSignalProjectionService,
)
from agent.src.features.wellbeing.storage.wellbeing_snapshot_repository import (
    WellbeingSnapshotRepository,
)


@dataclass(slots=True)
class WellbeingSummaryService:
    """현재 wellbeing signal 한 건을 제공한다.

    snapshot repository를 source of truth로 사용한다. 저장된 값이 없으면
    관측 데이터가 없다는 low-data payload를 반환한다.
    """

    repository: WellbeingSnapshotRepository | None = None
    projection_service: WellbeingSignalProjectionService | None = None
    _mock_payload: WellbeingSignalSummaryPayload | None = field(default=None)

    def get_current_summary(self) -> WellbeingSignalSummaryPayload:
        if self.projection_service is not None:
            self.projection_service.refresh_from_runtime()
        if self.repository is not None:
            latest_summary = self.repository.load_latest_summary()
            if latest_summary is not None:
                return latest_summary
        if self._mock_payload is not None:
            return self._mock_payload
        return WellbeingSignalSummaryPayload(
            computed_at=datetime.now(tz=timezone.utc),
            signal_score=0.0,
            signal_level=WellbeingSignalLevel.LOW,
            signal_label="데이터 없음",
            trend=WellbeingSignalTrend.UNKNOWN,
            summary="아직 분석된 최근 입력이 없어 현재 상태를 판단하지 않습니다.",
            action_tip="입력 데이터가 쌓이면 최근 상태를 다시 확인하세요.",
            parent_guidance=ParentWellbeingGuidancePayload(
                response_priority=(
                    "분석 데이터가 쌓일 때까지 평소 안부 확인을 유지하세요."
                ),
                conversation_starter="오늘 하루가 어땠는지 부담 없이 물어보세요.",
                caution_note="아직 판단 근거가 부족하므로 상태를 단정하지 마세요.",
            ),
            confidence=WellbeingSignalConfidence.LOW,
            low_data=True,
        )
