"""아이용 지원 대화 로컬 컨텍스트 provider."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from agent.src.contracts.wellbeing_signal_contracts import (
    WellbeingSignalSummaryPayload,
)
from agent.src.infrastructure.repositories.child_support_repository import (
    ChildSupportConversationRepository,
    ChildSupportMessageRecord,
)
from agent.src.infrastructure.repositories.query_buffer_repository import (
    QueryBufferRecord,
    QueryBufferRepository,
)
from agent.src.services.wellbeing.child_support_conversation_state import (
    ChildSupportConversationState,
    derive_child_support_conversation_state,
)
from agent.src.services.wellbeing.summary_service import WellbeingSummaryService


@dataclass(frozen=True, slots=True)
class ChildSupportRecentQueryContext:
    """LLM prompt에 넣을 수 있는 로컬 query buffer 요약."""

    occurred_at: datetime
    raw_text_excerpt: str
    predicted_label: str | None
    confidence: float | None


@dataclass(frozen=True, slots=True)
class ChildSupportConversationContext:
    """child-support 응답 생성에 필요한 agent-local context."""

    conversation_id: str
    conversation_state: ChildSupportConversationState = field(
        default_factory=ChildSupportConversationState
    )
    wellbeing_summary: WellbeingSignalSummaryPayload | None = None
    wellbeing_summary_is_observed: bool = False
    recent_queries: tuple[ChildSupportRecentQueryContext, ...] = field(
        default_factory=tuple
    )
    recent_messages: tuple[ChildSupportMessageRecord, ...] = field(
        default_factory=tuple
    )


@dataclass(slots=True)
class ChildSupportContextProvider:
    """여러 agent-local source를 child-support context로 조합한다."""

    summary_service: WellbeingSummaryService | None = None
    query_buffer_repository: QueryBufferRepository | None = None
    conversation_repository: ChildSupportConversationRepository | None = None
    recent_query_limit: int = 5
    recent_message_limit: int = 8
    raw_query_excerpt_chars: int = 140

    def build(self, conversation_id: str) -> ChildSupportConversationContext:
        """현재 conversation에 맞는 로컬 context를 만든다."""

        recent_messages = tuple(self._load_recent_messages(conversation_id))
        return ChildSupportConversationContext(
            conversation_id=conversation_id,
            conversation_state=derive_child_support_conversation_state(recent_messages),
            wellbeing_summary=self._load_summary(),
            wellbeing_summary_is_observed=self._has_observed_summary(),
            recent_queries=tuple(self._load_recent_queries()),
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

    def _load_recent_queries(self) -> list[ChildSupportRecentQueryContext]:
        if self.query_buffer_repository is None:
            return []
        try:
            records = self.query_buffer_repository.get_recent(
                limit=self.recent_query_limit
            )
        except Exception:
            return []
        return [self._query_record_to_context(record) for record in records]

    def _query_record_to_context(
        self,
        record: QueryBufferRecord,
    ) -> ChildSupportRecentQueryContext:
        excerpt = " ".join(record.raw_text.split())
        if len(excerpt) > self.raw_query_excerpt_chars:
            excerpt = f"{excerpt[: self.raw_query_excerpt_chars].rstrip()}..."
        return ChildSupportRecentQueryContext(
            occurred_at=record.occurred_at,
            raw_text_excerpt=excerpt,
            predicted_label=record.predicted_label,
            confidence=record.confidence,
        )

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
