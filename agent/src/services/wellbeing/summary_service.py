"""현재 wellbeing signal summary를 제공하는 service."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from shared.src.contracts.wellbeing_signal_contracts import (
    WellbeingSignalConfidence,
    WellbeingSignalLevel,
    WellbeingSignalSummaryPayload,
    WellbeingSignalTrend,
)


@dataclass(slots=True)
class WellbeingSummaryService:
    """현재 wellbeing signal 한 건을 제공한다.

    MVP 1차 구현은 deterministic mock을 반환한다.
    다음 단계에서는 repository/실제 판단 엔진 결과를 이 경계 뒤에 연결한다.
    """

    _mock_payload: WellbeingSignalSummaryPayload | None = field(default=None)

    def get_current_summary(self) -> WellbeingSignalSummaryPayload:
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
