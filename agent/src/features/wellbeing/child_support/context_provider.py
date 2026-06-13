"""아이용 지원 대화 로컬 컨텍스트 provider."""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.src.contracts.wellbeing_signal_contracts import (
    WellbeingSignalSummaryPayload,
)
from agent.src.features.wellbeing.child_support.conversation_state import (
    ChildSupportConversationState,
    derive_child_support_conversation_state,
)
from agent.src.features.wellbeing.signal.summary_service import WellbeingSummaryService
from agent.src.features.wellbeing.storage.child_support_repository import (
    ChildSupportConversationRepository,
    ChildSupportMessageRecord,
)


@dataclass(frozen=True, slots=True)
class ChildSupportConversationContext:
    """child-support 응답 생성에 필요한 agent-local context."""

    conversation_id: str
    conversation_state: ChildSupportConversationState = field(
        default_factory=ChildSupportConversationState
    )
    wellbeing_summary: WellbeingSignalSummaryPayload | None = None
    wellbeing_summary_is_observed: bool = False
    recent_messages: tuple[ChildSupportMessageRecord, ...] = field(
        default_factory=tuple
    )


@dataclass(slots=True)
class ChildSupportContextProvider:
    """여러 agent-local source를 child-support context로 조합한다."""

    summary_service: WellbeingSummaryService | None = None
    conversation_repository: ChildSupportConversationRepository | None = None
    recent_message_limit: int = 8

    def build(self, conversation_id: str) -> ChildSupportConversationContext:
        """현재 conversation에 맞는 로컬 context를 만든다."""

        recent_messages = tuple(self._load_recent_messages(conversation_id))
        return ChildSupportConversationContext(
            conversation_id=conversation_id,
            conversation_state=derive_child_support_conversation_state(recent_messages),
            wellbeing_summary=self._load_summary(),
            wellbeing_summary_is_observed=self._has_observed_summary(),
            recent_messages=recent_messages,
        )

    def _load_summary(self) -> WellbeingSignalSummaryPayload | None:
        if self.summary_service is None:
            return None
        try:
            return self.summary_service.get_current_summary()
        except Exception:
            return None

    def _has_observed_summary(self) -> bool:
        if self.summary_service is None:
            return False
        repository = getattr(self.summary_service, "repository", None)
        if repository is not None:
            try:
                return repository.load_latest_summary() is not None
            except Exception:
                return False
        return getattr(self.summary_service, "_mock_payload", None) is not None

    def _load_recent_messages(
        self,
        conversation_id: str,
    ) -> list[ChildSupportMessageRecord]:
        if self.conversation_repository is None:
            return []
        try:
            return self.conversation_repository.get_recent_messages(
                conversation_id,
                limit=self.recent_message_limit,
            )
        except Exception:
            return []
