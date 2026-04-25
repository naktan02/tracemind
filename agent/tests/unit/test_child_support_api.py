"""Child support API tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from agent.src.api import child_support as child_support_api
from agent.src.api.main import app
from agent.src.infrastructure.repositories.child_support_repository import (
    ChildSupportConversationRepository,
)
from agent.src.services.wellbeing.child_support_service import (
    ChildSupportCoachService,
)
from agent.src.services.wellbeing.summary_service import WellbeingSummaryService
from shared.src.contracts.child_support_contracts import (
    ChildSupportAssistantMode,
    ChildSupportConversationRequestPayload,
    ChildSupportSafetyLevel,
    ChildSupportScopeStatus,
)
from shared.src.contracts.wellbeing_signal_contracts import (
    WellbeingSignalConfidence,
    WellbeingSignalLevel,
    WellbeingSignalSummaryPayload,
    WellbeingSignalTrend,
)


class StubChildSupportLlmProvider:
    assistant_mode = ChildSupportAssistantMode.LOCAL_LLM

    def __init__(self) -> None:
        self.last_prompt = ""

    def generate_reply(self, *, prompt: str) -> str:
        self.last_prompt = prompt
        return (
            "지금 마음이 많이 무거웠겠어요. 먼저 지금 안전한 곳에 있는지 같이 확인해요."
        )


def test_child_support_api_returns_guarded_response() -> None:
    response = child_support_api.create_child_support_message(
        ChildSupportConversationRequestPayload(message="오늘 너무 불안해"),
        service=ChildSupportCoachService(),
    )

    assert response.safety_level == ChildSupportSafetyLevel.CHECK_IN
    assert response.conversation_id
    assert response.reply_text
    assert response.suggested_prompts


def test_child_support_service_uses_high_summary_as_check_in_context() -> None:
    service = ChildSupportCoachService(
        summary_service=WellbeingSummaryService(
            _mock_payload=WellbeingSignalSummaryPayload(
                computed_at=datetime(2026, 4, 25, 10, 30, tzinfo=timezone.utc),
                signal_score=78.0,
                signal_level=WellbeingSignalLevel.HIGH,
                signal_label="주의 필요",
                trend=WellbeingSignalTrend.RISING,
                summary="최근 상태가 평소보다 높습니다.",
                action_tip="짧게 안부를 물어보세요.",
                confidence=WellbeingSignalConfidence.MEDIUM,
                low_data=False,
            )
        )
    )

    response = service.create_response(
        ChildSupportConversationRequestPayload(message="그냥 얘기하고 싶어")
    )

    assert response.safety_level == ChildSupportSafetyLevel.CHECK_IN


def test_child_support_service_flags_parent_handoff_keywords() -> None:
    response = ChildSupportCoachService().create_response(
        ChildSupportConversationRequestPayload(message="친구가 계속 괴롭혀서 무서워")
    )

    assert response.safety_level == ChildSupportSafetyLevel.PARENT_HANDOFF
    assert response.parent_handoff_suggested is True
    assert response.parent_handoff_label is not None


def test_child_support_service_persists_local_conversation(
    tmp_path: Path,
) -> None:
    repository = ChildSupportConversationRepository(
        db_path=tmp_path / "child_support.db"
    )
    service = ChildSupportCoachService(conversation_repository=repository)

    first = service.create_response(
        ChildSupportConversationRequestPayload(message="오늘 마음이 답답해")
    )
    second = service.create_response(
        ChildSupportConversationRequestPayload(
            message="조금 더 말하고 싶어",
            conversation_id=first.conversation_id,
        )
    )

    assert second.conversation_id == first.conversation_id
    assert repository.count_messages(first.conversation_id) == 4


def test_child_support_service_redirects_off_topic_question() -> None:
    response = ChildSupportCoachService().create_response(
        ChildSupportConversationRequestPayload(message="파이썬 for문 알려줘")
    )

    assert response.scope_status == ChildSupportScopeStatus.REDIRECTED
    assert "마음" in response.reply_text


def test_child_support_service_uses_local_llm_provider() -> None:
    provider = StubChildSupportLlmProvider()
    response = ChildSupportCoachService(llm_provider=provider).create_response(
        ChildSupportConversationRequestPayload(message="요즘 계속 우울해")
    )

    assert response.assistant_mode == ChildSupportAssistantMode.LOCAL_LLM
    assert "wellbeing summary" in provider.last_prompt


def test_child_support_service_handles_self_harm_as_counseling_flow() -> None:
    response = ChildSupportCoachService().create_response(
        ChildSupportConversationRequestPayload(message="죽고싶어")
    )

    assert response.safety_level == ChildSupportSafetyLevel.URGENT
    assert response.parent_handoff_suggested is True
    assert "지금 안전한 곳" in response.reply_text


def test_child_support_router_is_registered_on_agent_app() -> None:
    route_paths = {route.path for route in app.routes}

    assert "/api/v1/child-support/messages" in route_paths
