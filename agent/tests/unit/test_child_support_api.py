"""Child support API tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent.src.api import child_support as child_support_api
from agent.src.api.main import app
from agent.src.contracts.child_support_contracts import (
    ChildSupportAssistantMode,
    ChildSupportConversationRequestPayload,
    ChildSupportSafetyLevel,
    ChildSupportScopeStatus,
)
from agent.src.contracts.wellbeing_signal_contracts import (
    WellbeingSignalConfidence,
    WellbeingSignalLevel,
    WellbeingSignalSummaryPayload,
    WellbeingSignalTrend,
)
from agent.src.features.captured_text.storage.records import CapturedTextRecord
from agent.src.features.captured_text.storage.repository import CapturedTextRepository
from agent.src.features.wellbeing.child_support.context_provider import (
    ChildSupportContextProvider,
)
from agent.src.features.wellbeing.child_support.evidence_summary import (
    ChildSupportEvidenceSummaryBuilder,
)
from agent.src.features.wellbeing.child_support.service import (
    ChildSupportCoachService,
    ChildSupportReplyUnavailable,
)
from agent.src.features.wellbeing.signal.summary_service import WellbeingSummaryService
from agent.src.features.wellbeing.storage.child_support_repository import (
    ChildSupportConversationRepository,
)
from agent.src.infrastructure.repositories.analysis_event_repository import (
    AnalysisEventRepository,
)
from shared.src.domain.entities.inference.events import AnalysisEvent


class StubChildSupportLlmProvider:
    assistant_mode = ChildSupportAssistantMode.LOCAL_LLM

    def __init__(self) -> None:
        self.last_prompt = ""

    def generate_reply(self, *, prompt: str) -> str:
        self.last_prompt = prompt
        return (
            "지금 정말 많이 버거워 보이네요. 말이 잘 안 나와도 괜찮아요. "
            "이 힘듦이 오늘 갑자기 커진 건지, 아니면 오래 쌓여 있다가 "
            "터진 건지만 천천히 이어 말해줘도 돼요. 짧은 한 문장이어도 괜찮아요."
        )


class ContextualChildSupportLlmProvider:
    assistant_mode = ChildSupportAssistantMode.LOCAL_LLM

    def __init__(self) -> None:
        self.last_prompt = ""

    def generate_reply(self, *, prompt: str) -> str:
        self.last_prompt = prompt
        return (
            "AI가 로컬 맥락을 보고 이어서 답합니다. "
            "지금 제일 크게 남은 느낌은 무엇인가요?"
        )


class ValidUrgentSafetyLlmProvider:
    assistant_mode = ChildSupportAssistantMode.LOCAL_LLM

    def generate_reply(self, *, prompt: str) -> str:
        return (
            "그 말을 꺼내준 건 정말 중요한 신호예요. 먼저 지금 안전한 곳에 "
            "있는지 확인하고 싶어요. 가까운 어른에게 같이 있어달라고 "
            "보여줄 문장을 만들 수 있어요. 말이 잘 안 나와도 괜찮아요."
        )


def test_child_support_api_returns_llm_response() -> None:
    response = child_support_api.create_child_support_message(
        ChildSupportConversationRequestPayload(message="오늘 너무 불안해"),
        service=ChildSupportCoachService(llm_provider=StubChildSupportLlmProvider()),
    )

    assert response.safety_level == ChildSupportSafetyLevel.CHECK_IN
    assert response.conversation_id
    assert response.reply_text
    assert response.suggested_prompts == ()


def test_child_support_service_requires_llm_provider() -> None:
    service = ChildSupportCoachService()

    with pytest.raises(ChildSupportReplyUnavailable):
        service.create_response(
            ChildSupportConversationRequestPayload(message="오늘 너무 불안해")
        )


def test_child_support_urgent_path_uses_llm_when_static_fallback_is_disabled() -> None:
    response = ChildSupportCoachService(
        llm_provider=ValidUrgentSafetyLlmProvider(),
    ).create_response(ChildSupportConversationRequestPayload(message="죽고 싶어"))

    assert response.assistant_mode == ChildSupportAssistantMode.LOCAL_LLM
    assert response.safety_level == ChildSupportSafetyLevel.URGENT
    assert "안전한 곳" in response.reply_text
    assert "어른" in response.reply_text


def test_child_support_passes_contextual_violence_question_to_llm() -> None:
    provider = ContextualChildSupportLlmProvider()
    response = ChildSupportCoachService(llm_provider=provider).create_response(
        ChildSupportConversationRequestPayload(
            message="친구들한테 맞아서 힘들어. 나 왜 힘든지 알아?"
        )
    )

    assert response.assistant_mode == ChildSupportAssistantMode.LOCAL_LLM
    assert response.reply_text.startswith("AI가 로컬 맥락을 보고")
    assert "아이의 새 메시지: 친구들한테 맞아서 힘들어" in provider.last_prompt
    assert "가능한 이유" in provider.last_prompt


def test_child_support_service_uses_high_summary_as_check_in_context() -> None:
    service = ChildSupportCoachService(
        llm_provider=StubChildSupportLlmProvider(),
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
        ),
    )

    response = service.create_response(
        ChildSupportConversationRequestPayload(message="그냥 얘기하고 싶어")
    )

    assert response.safety_level == ChildSupportSafetyLevel.CHECK_IN


def test_child_support_service_flags_parent_handoff_keywords() -> None:
    response = ChildSupportCoachService(
        llm_provider=StubChildSupportLlmProvider()
    ).create_response(
        ChildSupportConversationRequestPayload(message="친구가 계속 괴롭혀서 무서워")
    )

    assert response.safety_level == ChildSupportSafetyLevel.PARENT_HANDOFF
    assert response.parent_handoff_suggested is True
    assert response.parent_handoff_label is None


def test_child_support_service_keeps_violence_flow_open() -> None:
    response = ChildSupportCoachService(
        llm_provider=StubChildSupportLlmProvider()
    ).create_response(
        ChildSupportConversationRequestPayload(message="친구한테 맞았어 너무 힘들어")
    )

    assert response.safety_level == ChildSupportSafetyLevel.PARENT_HANDOFF
    assert response.assistant_mode == ChildSupportAssistantMode.LOCAL_LLM


def test_child_support_service_persists_local_conversation(
    tmp_path: Path,
) -> None:
    repository = ChildSupportConversationRepository(
        db_path=tmp_path / "child_support.db"
    )
    service = ChildSupportCoachService(
        conversation_repository=repository,
        llm_provider=StubChildSupportLlmProvider(),
    )

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


def test_child_support_service_uses_violence_context_for_safe_followup(
    tmp_path: Path,
) -> None:
    repository = ChildSupportConversationRepository(
        db_path=tmp_path / "child_support.db"
    )
    service = ChildSupportCoachService(
        conversation_repository=repository,
        llm_provider=StubChildSupportLlmProvider(),
    )

    first = service.create_response(
        ChildSupportConversationRequestPayload(message="친구한테 맞았어 너무 힘들어")
    )
    second = service.create_response(
        ChildSupportConversationRequestPayload(
            message="떨어졌고 집에 왔는데 속상해",
            conversation_id=first.conversation_id,
        )
    )

    assert second.safety_level == ChildSupportSafetyLevel.CHECK_IN
    assert second.parent_handoff_suggested is False
    assert second.assistant_mode == ChildSupportAssistantMode.LOCAL_LLM
    assert second.suggested_prompts == ()


def test_child_support_service_uses_violence_context_for_peer_response_planning(
    tmp_path: Path,
) -> None:
    repository = ChildSupportConversationRepository(
        db_path=tmp_path / "child_support.db"
    )
    service = ChildSupportCoachService(
        conversation_repository=repository,
        llm_provider=StubChildSupportLlmProvider(),
    )

    first = service.create_response(
        ChildSupportConversationRequestPayload(message="친구한테 맞았어 너무 힘들어")
    )
    second = service.create_response(
        ChildSupportConversationRequestPayload(
            message="집으로 왔고 그 친구가 미워 어떻게 하는게 좋을까",
            conversation_id=first.conversation_id,
        )
    )

    assert second.safety_level == ChildSupportSafetyLevel.CHECK_IN
    assert second.assistant_mode == ChildSupportAssistantMode.LOCAL_LLM
    assert second.suggested_prompts == ()


def test_child_support_service_escalates_other_harm_ideation_after_violence(
    tmp_path: Path,
) -> None:
    repository = ChildSupportConversationRepository(
        db_path=tmp_path / "child_support.db"
    )
    service = ChildSupportCoachService(
        conversation_repository=repository,
        llm_provider=StubChildSupportLlmProvider(),
    )

    first = service.create_response(
        ChildSupportConversationRequestPayload(message="친구한테 맞았어 너무 힘들어")
    )
    second = service.create_response(
        ChildSupportConversationRequestPayload(
            message="죽여버리고 싶어",
            conversation_id=first.conversation_id,
        )
    )

    assert second.safety_level == ChildSupportSafetyLevel.URGENT
    assert second.parent_handoff_suggested is True
    assert second.assistant_mode == ChildSupportAssistantMode.LOCAL_LLM
    assert second.suggested_prompts == ()


def test_child_support_service_uses_warm_deescalation_after_other_harm_urgent(
    tmp_path: Path,
) -> None:
    repository = ChildSupportConversationRepository(
        db_path=tmp_path / "child_support.db"
    )
    service = ChildSupportCoachService(
        conversation_repository=repository,
        llm_provider=StubChildSupportLlmProvider(),
    )

    first = service.create_response(
        ChildSupportConversationRequestPayload(message="친구한테 맞았어 너무 힘들어")
    )
    second = service.create_response(
        ChildSupportConversationRequestPayload(
            message="그 친구 죽여버리고 싶어",
            conversation_id=first.conversation_id,
        )
    )
    third = service.create_response(
        ChildSupportConversationRequestPayload(
            message="너무 힘든데...",
            conversation_id=first.conversation_id,
        )
    )

    assert second.safety_level == ChildSupportSafetyLevel.URGENT
    assert third.safety_level == ChildSupportSafetyLevel.CHECK_IN
    assert third.assistant_mode == ChildSupportAssistantMode.LOCAL_LLM
    assert third.suggested_prompts == ()


def test_child_support_service_refuses_other_harm_method_request_after_violence(
    tmp_path: Path,
) -> None:
    repository = ChildSupportConversationRepository(
        db_path=tmp_path / "child_support.db"
    )
    service = ChildSupportCoachService(
        conversation_repository=repository,
        llm_provider=StubChildSupportLlmProvider(),
    )

    first = service.create_response(
        ChildSupportConversationRequestPayload(message="친구한테 맞았어 너무 힘들어")
    )
    second = service.create_response(
        ChildSupportConversationRequestPayload(
            message="내가 걔 죽이려면 어떻게 해야 할까",
            conversation_id=first.conversation_id,
        )
    )

    assert second.safety_level == ChildSupportSafetyLevel.URGENT
    assert second.parent_handoff_suggested is True
    assert second.assistant_mode == ChildSupportAssistantMode.LOCAL_LLM


def test_child_support_service_redirects_off_topic_question() -> None:
    response = ChildSupportCoachService(
        llm_provider=StubChildSupportLlmProvider()
    ).create_response(
        ChildSupportConversationRequestPayload(message="파이썬 for문 알려줘")
    )

    assert response.scope_status == ChildSupportScopeStatus.REDIRECTED
    assert response.assistant_mode == ChildSupportAssistantMode.LOCAL_LLM


def test_child_support_service_uses_local_llm_provider() -> None:
    provider = StubChildSupportLlmProvider()
    response = ChildSupportCoachService(llm_provider=provider).create_response(
        ChildSupportConversationRequestPayload(message="요즘 계속 힘들어")
    )

    assert response.safety_level == ChildSupportSafetyLevel.CHECK_IN
    assert response.assistant_mode == ChildSupportAssistantMode.LOCAL_LLM
    assert "wellbeing summary" in provider.last_prompt
    assert "safety_reason_hint: calming_keyword" in provider.last_prompt
    assert "아이의 새 메시지: 요즘 계속 힘들어" in provider.last_prompt


def test_child_support_llm_prompt_includes_local_evidence_summary(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "agent_local.db"
    analysis_repository = AnalysisEventRepository(db_path=db_path)
    captured_repository = CapturedTextRepository(db_path=db_path)
    occurred_at = datetime(2026, 6, 14, 9, 0, tzinfo=timezone.utc)
    captured_repository.save(
        CapturedTextRecord(
            event_id="captured-evidence-1",
            occurred_at=occurred_at,
            received_at=occurred_at,
            text="요즘 친구 관계 때문에 불안하고 잠을 못 자겠다는 검색을 계속 봤어",
            locale="ko",
            source_type="browser",
            surface_type="search",
        )
    )
    analysis_repository.save(
        AnalysisEvent(
            query_id="captured-evidence-1",
            occurred_at=occurred_at,
            translated_text=None,
            embedding_model_id="test-embedding",
            translation_model_id=None,
            category_scores={"normal": 12.0, "anxiety": 76.0, "depression": 42.0},
        ),
        source_event_id="captured-evidence-1",
        scorer_name="test_scorer",
        model_revision="test-revision",
    )
    provider = StubChildSupportLlmProvider()
    service = ChildSupportCoachService(
        llm_provider=provider,
        context_provider=ChildSupportContextProvider(
            summary_service=_high_summary_service(),
            evidence_summary_builder=ChildSupportEvidenceSummaryBuilder(
                analysis_event_repository=analysis_repository,
                captured_text_repository=captured_repository,
            ),
        ),
    )

    response = service.create_response(
        ChildSupportConversationRequestPayload(message="요즘 계속 힘들어")
    )

    assert response.assistant_mode == ChildSupportAssistantMode.LOCAL_LLM
    assert "최근 캡처/검색 근거" in provider.last_prompt
    assert "반복 주제:" in provider.last_prompt
    assert "불안/공포" in provider.last_prompt
    assert "친구/관계 갈등" in provider.last_prompt
    assert "최근 원문 일부:" in provider.last_prompt


def test_child_support_proactive_prompt_uses_evidence_topic(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "agent_local.db"
    analysis_repository = AnalysisEventRepository(db_path=db_path)
    captured_repository = CapturedTextRepository(db_path=db_path)
    occurred_at = datetime(2026, 6, 14, 9, 0, tzinfo=timezone.utc)
    captured_repository.save(
        CapturedTextRecord(
            event_id="captured-evidence-2",
            occurred_at=occurred_at,
            received_at=occurred_at,
            text="학교 시험 성적 때문에 불안해서 잠이 안 온다는 검색",
            locale="ko",
            source_type="browser",
            surface_type="search",
        )
    )
    analysis_repository.save(
        AnalysisEvent(
            query_id="captured-evidence-2",
            occurred_at=occurred_at,
            translated_text=None,
            embedding_model_id="test-embedding",
            translation_model_id=None,
            category_scores={"normal": 18.0, "anxiety": 81.0},
        ),
        source_event_id="captured-evidence-2",
        scorer_name="test_scorer",
        model_revision="test-revision",
    )
    provider = StubChildSupportLlmProvider()
    service = ChildSupportCoachService(
        llm_provider=provider,
        context_provider=ChildSupportContextProvider(
            summary_service=_high_summary_service(),
            evidence_summary_builder=ChildSupportEvidenceSummaryBuilder(
                analysis_event_repository=analysis_repository,
                captured_text_repository=captured_repository,
            ),
        ),
    )

    prompt = service.build_proactive_prompt()

    assert prompt.should_prompt is True
    assert prompt.prompt_text is not None
    assert "학교/성적 부담" in provider.last_prompt


def test_child_support_proactive_prompt_passes_context_to_llm(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "agent_local.db"
    analysis_repository = AnalysisEventRepository(db_path=db_path)
    captured_repository = CapturedTextRepository(db_path=db_path)
    occurred_at = datetime(2026, 6, 14, 9, 0, tzinfo=timezone.utc)
    captured_repository.save(
        CapturedTextRecord(
            event_id="captured-evidence-3",
            occurred_at=occurred_at,
            received_at=occurred_at,
            text="친구랑 싸우고 불안해서 잠이 안 온다는 검색",
            locale="ko",
            source_type="browser",
            surface_type="search",
        )
    )
    analysis_repository.save(
        AnalysisEvent(
            query_id="captured-evidence-3",
            occurred_at=occurred_at,
            translated_text=None,
            embedding_model_id="test-embedding",
            translation_model_id=None,
            category_scores={"normal": 10.0, "anxiety": 84.0},
        ),
        source_event_id="captured-evidence-3",
        scorer_name="test_scorer",
        model_revision="test-revision",
    )
    provider = StubChildSupportLlmProvider()
    service = ChildSupportCoachService(
        llm_provider=provider,
        context_provider=ChildSupportContextProvider(
            summary_service=_high_summary_service(),
            evidence_summary_builder=ChildSupportEvidenceSummaryBuilder(
                analysis_event_repository=analysis_repository,
                captured_text_repository=captured_repository,
            ),
        ),
    )

    prompt = service.build_proactive_prompt()

    assert prompt.should_prompt is True
    assert prompt.prompt_text is not None
    assert "지금 정말 많이 버거워 보이네요" in prompt.prompt_text
    assert "짧은 한 문장이어도 괜찮아요" in prompt.prompt_text
    assert "최근 wellbeing context notes:" in provider.last_prompt
    assert "반복 주제:" in provider.last_prompt
    assert "친구/관계 갈등" in provider.last_prompt


def test_child_support_proactive_prompt_uses_high_risk_local_evidence(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "agent_local.db"
    analysis_repository = AnalysisEventRepository(db_path=db_path)
    captured_repository = CapturedTextRepository(db_path=db_path)
    occurred_at = datetime(2026, 6, 14, 9, 0, tzinfo=timezone.utc)
    captured_repository.save(
        CapturedTextRecord(
            event_id="captured-evidence-4",
            occurred_at=occurred_at,
            received_at=occurred_at,
            text="죽고 싶어. 자살 생각이 계속 나.",
            locale="ko",
            source_type="browser",
            surface_type="search",
        )
    )
    analysis_repository.save(
        AnalysisEvent(
            query_id="captured-evidence-4",
            occurred_at=occurred_at,
            translated_text="I want to die. I keep thinking about suicide.",
            embedding_model_id="test-embedding",
            translation_model_id=None,
            category_scores={"normal": 0.0, "suicidal": 0.09},
        ),
        source_event_id="captured-evidence-4",
        scorer_name="test_scorer",
        model_revision="test-revision",
    )
    provider = StubChildSupportLlmProvider()
    service = ChildSupportCoachService(
        llm_provider=provider,
        context_provider=ChildSupportContextProvider(
            summary_service=_low_summary_service(),
            evidence_summary_builder=ChildSupportEvidenceSummaryBuilder(
                analysis_event_repository=analysis_repository,
                captured_text_repository=captured_repository,
            ),
        ),
    )

    prompt = service.build_proactive_prompt()

    assert prompt.should_prompt is True
    assert prompt.safety_level == ChildSupportSafetyLevel.PARENT_HANDOFF
    assert prompt.prompt_text is not None
    assert "자해/죽음 관련 표현" in provider.last_prompt


def test_child_support_urgent_llm_prompt_reflects_high_risk_local_evidence(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "agent_local.db"
    analysis_repository = AnalysisEventRepository(db_path=db_path)
    captured_repository = CapturedTextRepository(db_path=db_path)
    occurred_at = datetime(2026, 6, 14, 9, 0, tzinfo=timezone.utc)
    captured_repository.save(
        CapturedTextRecord(
            event_id="captured-evidence-5",
            occurred_at=occurred_at,
            received_at=occurred_at,
            text="죽고 싶어. 자살 생각이 계속 나.",
            locale="ko",
            source_type="browser",
            surface_type="search",
        )
    )
    analysis_repository.save(
        AnalysisEvent(
            query_id="captured-evidence-5",
            occurred_at=occurred_at,
            translated_text="I want to die. I keep thinking about suicide.",
            embedding_model_id="test-embedding",
            translation_model_id=None,
            category_scores={"normal": 0.0, "suicidal": 0.09},
        ),
        source_event_id="captured-evidence-5",
        scorer_name="test_scorer",
        model_revision="test-revision",
    )
    provider = StubChildSupportLlmProvider()
    service = ChildSupportCoachService(
        llm_provider=provider,
        context_provider=ChildSupportContextProvider(
            summary_service=_low_summary_service(),
            evidence_summary_builder=ChildSupportEvidenceSummaryBuilder(
                analysis_event_repository=analysis_repository,
                captured_text_repository=captured_repository,
            ),
        ),
    )

    response = service.create_response(
        ChildSupportConversationRequestPayload(message="죽고 싶어")
    )

    assert response.safety_level == ChildSupportSafetyLevel.URGENT
    assert "safety_level_hint: urgent" in provider.last_prompt
    assert "자해/죽음 관련 표현" in provider.last_prompt
    assert "최근 원문 일부: 죽고 싶어. 자살 생각이 계속 나." in provider.last_prompt


def test_child_support_service_keeps_general_distress_in_check_in() -> None:
    response = ChildSupportCoachService(
        llm_provider=StubChildSupportLlmProvider()
    ).create_response(ChildSupportConversationRequestPayload(message="나 너무 힘들어"))

    assert response.safety_level == ChildSupportSafetyLevel.CHECK_IN
    assert response.parent_handoff_suggested is False


def test_child_support_service_handles_self_harm_as_counseling_flow() -> None:
    response = ChildSupportCoachService(
        llm_provider=StubChildSupportLlmProvider()
    ).create_response(ChildSupportConversationRequestPayload(message="죽고싶어"))

    assert response.safety_level == ChildSupportSafetyLevel.URGENT
    assert response.parent_handoff_suggested is True
    assert response.assistant_mode == ChildSupportAssistantMode.LOCAL_LLM


def test_child_support_service_attempts_llm_for_urgent_flow() -> None:
    provider = StubChildSupportLlmProvider()
    response = ChildSupportCoachService(llm_provider=provider).create_response(
        ChildSupportConversationRequestPayload(message="죽고싶어")
    )

    assert response.safety_level == ChildSupportSafetyLevel.URGENT
    assert response.assistant_mode == ChildSupportAssistantMode.LOCAL_LLM
    assert "safety_level_hint: urgent" in provider.last_prompt


def test_child_support_service_builds_proactive_prompt_for_high_summary() -> None:
    service = ChildSupportCoachService(
        llm_provider=StubChildSupportLlmProvider(),
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
        ),
    )

    prompt = service.build_proactive_prompt()

    assert prompt.should_prompt is True
    assert prompt.safety_level == ChildSupportSafetyLevel.CHECK_IN
    assert prompt.prompt_text is not None


def test_child_support_service_builds_proactive_prompt_from_watch_score() -> None:
    service = ChildSupportCoachService(
        llm_provider=StubChildSupportLlmProvider(),
        summary_service=WellbeingSummaryService(
            _mock_payload=WellbeingSignalSummaryPayload(
                computed_at=datetime(2026, 4, 25, 10, 30, tzinfo=timezone.utc),
                signal_score=35.0,
                signal_level=WellbeingSignalLevel.MODERATE,
                signal_label="관찰 필요",
                trend=WellbeingSignalTrend.RISING,
                summary="최근 상태가 조금 올라갔습니다.",
                action_tip="짧게 상태를 확인해 보세요.",
                confidence=WellbeingSignalConfidence.MEDIUM,
                low_data=False,
            )
        ),
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


def _high_summary_service() -> WellbeingSummaryService:
    return WellbeingSummaryService(
        _mock_payload=WellbeingSignalSummaryPayload(
            computed_at=datetime(2026, 6, 14, 9, 0, tzinfo=timezone.utc),
            signal_score=72.0,
            signal_level=WellbeingSignalLevel.HIGH,
            signal_label="주의 필요",
            trend=WellbeingSignalTrend.RISING,
            summary="최근 상태가 평소보다 높습니다.",
            action_tip="짧게 안부를 물어보세요.",
            confidence=WellbeingSignalConfidence.MEDIUM,
            low_data=False,
        )
    )


def _low_summary_service() -> WellbeingSummaryService:
    return WellbeingSummaryService(
        _mock_payload=WellbeingSignalSummaryPayload(
            computed_at=datetime(2026, 6, 14, 9, 0, tzinfo=timezone.utc),
            signal_score=0.0,
            signal_level=WellbeingSignalLevel.LOW,
            signal_label="안정",
            trend=WellbeingSignalTrend.UNKNOWN,
            summary="최근 상태가 비교적 안정적으로 보입니다.",
            action_tip="짧게 상태를 확인해 보세요.",
            confidence=WellbeingSignalConfidence.LOW,
            low_data=False,
        )
    )
