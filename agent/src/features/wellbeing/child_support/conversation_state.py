"""아이용 지원 대화의 agent-local 상태 해석."""

from __future__ import annotations

from dataclasses import dataclass

from agent.src.contracts.child_support_contracts import ChildSupportSafetyLevel
from agent.src.features.wellbeing.storage.child_support_repository import (
    ChildSupportMessageRecord,
)

_OTHER_HARM_INTENTS = {
    "other_harm_ideation",
    "other_harm_method_request",
}


@dataclass(frozen=True, slots=True)
class ChildSupportConversationState:
    """최근 대화에서 이어받을 수 있는 최소 상태.

    이 상태는 UI나 shared contract로 노출하지 않고 agent-local routing에만 쓴다.
    """

    has_recent_parent_handoff: bool = False
    has_recent_urgent: bool = False
    has_recent_other_harm_risk: bool = False
    last_safety_reason: str | None = None
    turns_since_urgent: int | None = None


def derive_child_support_conversation_state(
    records: tuple[ChildSupportMessageRecord, ...],
) -> ChildSupportConversationState:
    """최근 메시지 snapshot에서 안전 routing에 필요한 상태만 추출한다."""

    recent_records = records[-6:]
    recent_reasons = tuple(
        reason for record in recent_records if (reason := _extract_reason(record))
    )
    last_safety_reason = recent_reasons[-1] if recent_reasons else None
    last_urgent_index = _find_last_urgent_index(recent_records)
    return ChildSupportConversationState(
        has_recent_parent_handoff=any(
            record.safety_level == ChildSupportSafetyLevel.PARENT_HANDOFF.value
            for record in recent_records
        ),
        has_recent_urgent=any(
            record.safety_level == ChildSupportSafetyLevel.URGENT.value
            for record in recent_records
        ),
        has_recent_other_harm_risk=any(
            reason in _OTHER_HARM_INTENTS for reason in recent_reasons
        ),
        last_safety_reason=last_safety_reason,
        turns_since_urgent=(
            None
            if last_urgent_index is None
            else len(recent_records) - last_urgent_index - 1
        ),
    )


def _extract_reason(
    record: ChildSupportMessageRecord,
) -> str | None:
    raw_reason = record.metadata.get("assessment_reason")
    return raw_reason if isinstance(raw_reason, str) else None


def _find_last_urgent_index(
    records: tuple[ChildSupportMessageRecord, ...],
) -> int | None:
    for index in range(len(records) - 1, -1, -1):
        if records[index].safety_level == ChildSupportSafetyLevel.URGENT.value:
            return index
    return None
