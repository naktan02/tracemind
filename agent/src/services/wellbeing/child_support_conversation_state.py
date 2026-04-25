"""아이용 지원 대화의 agent-local 상태 해석."""

from __future__ import annotations

from dataclasses import dataclass

from agent.src.infrastructure.repositories.child_support_repository import (
    ChildSupportMessageRecord,
)
from shared.src.contracts.child_support_contracts import ChildSupportSafetyLevel


@dataclass(frozen=True, slots=True)
class ChildSupportConversationState:
    """최근 대화에서 이어받을 수 있는 최소 상태.

    이 상태는 UI나 shared contract로 노출하지 않고 agent-local routing에만 쓴다.
    """

    has_recent_parent_handoff: bool = False
    has_recent_urgent: bool = False


def derive_child_support_conversation_state(
    records: tuple[ChildSupportMessageRecord, ...],
) -> ChildSupportConversationState:
    """최근 메시지 snapshot에서 안전 routing에 필요한 상태만 추출한다."""

    recent_records = records[-6:]
    return ChildSupportConversationState(
        has_recent_parent_handoff=any(
            record.safety_level == ChildSupportSafetyLevel.PARENT_HANDOFF.value
            for record in recent_records
        ),
        has_recent_urgent=any(
            record.safety_level == ChildSupportSafetyLevel.URGENT.value
            for record in recent_records
        ),
    )


__all__ = [
    "ChildSupportConversationState",
    "derive_child_support_conversation_state",
]
