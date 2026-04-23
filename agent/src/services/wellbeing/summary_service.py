"""현재 wellbeing signal summary를 제공하는 service."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from agent.src.infrastructure.repositories.wellbeing_snapshot_repository import (
    WellbeingSnapshotRepository,
)
from shared.src.contracts.wellbeing_signal_contracts import (
    WellbeingSignalConfidence,
    WellbeingSignalLevel,
    WellbeingSignalSummaryPayload,
    WellbeingSignalTrend,
)


@dataclass(slots=True)
class WellbeingSummaryService:
    """현재 wellbeing signal 한 건을 제공한다.

    현재 단계에서는 snapshot repository를 우선 source of truth로 사용하고,
    저장된 값이 없을 때만 deterministic mock을 fallback으로 반환한다.
    """

    repository: WellbeingSnapshotRepository | None = None
    _mock_payload: WellbeingSignalSummaryPayload | None = field(default=None)

    def get_current_summary(self) -> WellbeingSignalSummaryPayload:
        if self.repository is not None:
            latest_summary = self.repository.load_latest_summary()
            if latest_summary is not None:
                return latest_summary
        if self._mock_payload is not None:
            return self._mock_payload
        return WellbeingSignalSummaryPayload(
            computed_at=datetime.now(tz=timezone.utc),
            signal_score=61.0,
            signal_level=WellbeingSignalLevel.HIGH,
            signal_label="주의 필요",
            trend=WellbeingSignalTrend.RISING,
            summary="최근 전체 상태가 평소보다 높게 관찰되었습니다.",
            action_tip="오늘은 짧게 안부를 묻고 저녁 대화 시간을 확보해 보세요.",
            confidence=WellbeingSignalConfidence.MEDIUM,
            low_data=False,
        )
