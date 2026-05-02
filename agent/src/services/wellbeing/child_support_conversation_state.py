"""아이용 지원 대화의 agent-local 상태 해석."""

from __future__ import annotations

from dataclasses import dataclass

from agent.src.infrastructure.repositories.child_support_repository import (
    ChildSupportMessageRecord,
)
from agent.src.services.wellbeing.child_support_safety_intent import (
    ChildSupportSafetyIntent,
)
from shared.src.contracts.child_support_contracts import ChildSupportSafetyLevel

_OTHER_HARM_INTENTS = {
    ChildSupportSafetyIntent.OTHER_HARM_IDEATION,
    ChildSupportSafetyIntent.OTHER_HARM_METHOD_REQUEST,
}


@dataclass(frozen=True, slots=True)
class ChildSupportConversationState:
    """최근 대화에서 이어받을 수 있는 최소 상태.

    이 상태는 UI나 shared contract로 노출하지 않고 agent-local routing에만 쓴다.
    """

    has_recent_parent_handoff: bool = False
    has_recent_urgent: bool = False
    has_recent_other_harm_risk: bool = False
    last_safety_intent: ChildSupportSafetyIntent | None = None
    turns_since_urgent: int | None = None


def derive_child_support_conversation_state(
    records: tuple[ChildSupportMessageRecord, ...],
) -> ChildSupportConversationState:
    """최근 메시지 snapshot에서 안전 routing에 필요한 상태만 추출한다."""

    recent_records = records[-6:]
    recent_intents = tuple(
        intent for record in recent_records if (intent := _extract_intent(record))
    )
    last_safety_intent = recent_intents[-1] if recent_intents else None
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
            intent in _OTHER_HARM_INTENTS for intent in recent_intents
        ),
        last_safety_intent=last_safety_intent,
        turns_since_urgent=(
            None
            if last_urgent_index is None
            else len(recent_records) - last_urgent_index - 1
        ),
    )


def _extract_intent(
    record: ChildSupportMessageRecord,
) -> ChildSupportSafetyIntent | None:
    raw_intent = record.metadata.get("assessment_intent")
    if not isinstance(raw_intent, str):
        return None
    try:
        return ChildSupportSafetyIntent(raw_intent)
    except ValueError:
        return None


def _find_last_urgent_index(
    records: tuple[ChildSupportMessageRecord, ...],
) -> int | None:
    for index in range(len(records) - 1, -1, -1):
        if records[index].safety_level == ChildSupportSafetyLevel.URGENT.value:
            return index
    return None


__all__ = [
    "ChildSupportConversationState",
    "derive_child_support_conversation_state",
]
