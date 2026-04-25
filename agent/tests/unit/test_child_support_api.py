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
            "지금 마음이 많이 무거웠겠어요. 가장 힘든 느낌이 몸 어디에서 "
            "크게 느껴지는지 하나만 골라볼까요? 그 다음에는 숨을 천천히 "
            "같이 쉬어볼게요."
        )


class ParentHandoffLeakingLlmProvider:
    assistant_mode = ChildSupportAssistantMode.LOCAL_LLM

    def generate_reply(self, *, prompt: str) -> str:
        return (
            "많이 힘들었겠어요. 가족이나 친구와 이야기해보세요. "
            "어른에게 말하는 것도 도움이 돼요. 어떤 부분이 제일 무거웠나요?"
        )


class ClosingViolenceLlmProvider:
    assistant_mode = ChildSupportAssistantMode.LOCAL_LLM

    def generate_reply(self, *, prompt: str) -> str:
        return (
            "친구한테 맞았다니 정말 힘들겠구나. "
            "엄마나 아빠에게 이야기해도 돼. "
            "안전하고 편안한 시간 보내길 바라."
        )


class UnsafeSoothingViolenceLlmProvider:
    assistant_mode = ChildSupportAssistantMode.LOCAL_LLM

    def generate_reply(self, *, prompt: str) -> str:
        return (
            "친구 때문에 힘들어 보이는데, 지금 혼자 조용히 시간을 가지며 "
            "마음을 진정해 보는 건 어떨까? 편안한 음악을 들으며 쉬어보자."
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


def test_child_support_service_keeps_violence_flow_open() -> None:
    response = ChildSupportCoachService().create_response(
        ChildSupportConversationRequestPayload(message="친구한테 맞았어 너무 힘들어")
    )

    assert response.safety_level == ChildSupportSafetyLevel.PARENT_HANDOFF
    assert "다친 곳" in response.reply_text
    assert "말해줄 수 있을까요" in response.reply_text


def test_child_support_service_removes_closing_from_violence_llm() -> None:
    response = ChildSupportCoachService(
        llm_provider=ClosingViolenceLlmProvider()
    ).create_response(
        ChildSupportConversationRequestPayload(message="친구한테 맞았어 너무 힘들어")
    )

    assert response.safety_level == ChildSupportSafetyLevel.PARENT_HANDOFF
    assert response.assistant_mode == ChildSupportAssistantMode.LOCAL_GUARDED
    assert "편안한 시간 보내" not in response.reply_text
    assert "다친 곳" in response.reply_text


def test_child_support_service_rejects_unsafe_soothing_violence_llm() -> None:
    response = ChildSupportCoachService(
        llm_provider=UnsafeSoothingViolenceLlmProvider()
    ).create_response(
        ChildSupportConversationRequestPayload(message="친구한테 맞았어 너무 힘들어")
    )

    assert response.safety_level == ChildSupportSafetyLevel.PARENT_HANDOFF
    assert response.assistant_mode == ChildSupportAssistantMode.LOCAL_GUARDED
    assert "편안한 음악" not in response.reply_text
    assert "혼자 조용히" not in response.reply_text
    assert "안전한 곳" in response.reply_text
    assert "다친 곳" in response.reply_text


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
        ChildSupportConversationRequestPayload(message="요즘 계속 힘들어")
    )

    assert response.safety_level == ChildSupportSafetyLevel.CHECK_IN
    assert response.assistant_mode == ChildSupportAssistantMode.LOCAL_LLM
    assert "wellbeing summary" in provider.last_prompt
    assert "response_strategy: check_in" in provider.last_prompt
    assert "skeleton:" in provider.last_prompt


def test_child_support_service_keeps_general_distress_in_check_in() -> None:
    response = ChildSupportCoachService().create_response(
        ChildSupportConversationRequestPayload(message="나 너무 힘들어")
    )

    assert response.safety_level == ChildSupportSafetyLevel.CHECK_IN
    assert response.parent_handoff_suggested is False
    assert "어른" not in response.reply_text


def test_child_support_service_filters_parent_handoff_from_check_in_llm() -> None:
    response = ChildSupportCoachService(
        llm_provider=ParentHandoffLeakingLlmProvider()
    ).create_response(ChildSupportConversationRequestPayload(message="나 너무 힘들어"))

    assert response.safety_level == ChildSupportSafetyLevel.CHECK_IN
    assert response.assistant_mode == ChildSupportAssistantMode.LOCAL_GUARDED
    assert "가족" not in response.reply_text
    assert "어른" not in response.reply_text
    assert "골라볼까요" in response.reply_text


def test_child_support_service_handles_self_harm_as_counseling_flow() -> None:
    response = ChildSupportCoachService().create_response(
        ChildSupportConversationRequestPayload(message="죽고싶어")
    )

    assert response.safety_level == ChildSupportSafetyLevel.URGENT
    assert response.parent_handoff_suggested is True
    assert "지금 안전한 곳" in response.reply_text


def test_child_support_service_bypasses_llm_for_urgent_flow() -> None:
    provider = StubChildSupportLlmProvider()
    response = ChildSupportCoachService(llm_provider=provider).create_response(
        ChildSupportConversationRequestPayload(message="죽고싶어")
    )

    assert response.safety_level == ChildSupportSafetyLevel.URGENT
    assert response.assistant_mode == ChildSupportAssistantMode.LOCAL_GUARDED
    assert provider.last_prompt == ""


def test_child_support_service_builds_proactive_prompt_for_high_summary() -> None:
    service = ChildSupportCoachService(
        summary_service=WellbeingSummaryService(
            _mock_payload=WellbeingSignalSummaryPayload(
                computed_at=datetime(2026, 4, 25, 10, 30, tzinfo=timezone.utc),
                signal_score=82.0,
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

    prompt = service.build_proactive_prompt()

    assert prompt.should_prompt is True
    assert prompt.safety_level == ChildSupportSafetyLevel.CHECK_IN
    assert prompt.prompt_text is not None


def test_child_support_service_skips_proactive_prompt_for_low_data() -> None:
    service = ChildSupportCoachService(
        summary_service=WellbeingSummaryService(
            _mock_payload=WellbeingSignalSummaryPayload(
                computed_at=datetime(2026, 4, 25, 10, 30, tzinfo=timezone.utc),
                signal_score=0.0,
                signal_level=WellbeingSignalLevel.LOW,
                signal_label="데이터 부족",
                trend=WellbeingSignalTrend.UNKNOWN,
                summary="아직 충분한 데이터가 없습니다.",
                action_tip="조금 더 사용한 뒤 확인하세요.",
                confidence=WellbeingSignalConfidence.LOW,
                low_data=True,
            )
        )
    )

    prompt = service.build_proactive_prompt()

    assert prompt.should_prompt is False
    assert prompt.prompt_text is None


def test_child_support_router_is_registered_on_agent_app() -> None:
    route_paths = {route.path for route in app.routes}

    assert "/api/v1/child-support/messages" in route_paths
    assert "/api/v1/child-support/proactive-prompt" in route_paths
