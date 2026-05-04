"""아이용 지원 대화 응답 서비스."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from agent.src.infrastructure.repositories.child_support_repository import (
    ChildSupportConversationRepository,
    ChildSupportMessageRecord,
)
from agent.src.services.wellbeing.child_support_context_provider import (
    ChildSupportContextProvider,
    ChildSupportConversationContext,
)
from agent.src.services.wellbeing.child_support_llm_provider import (
    ChildSupportLlmError,
    ChildSupportLlmProvider,
)
from agent.src.services.wellbeing.child_support_response_policy import (
    ChildSupportResponsePlan,
    ChildSupportResponsePolicy,
)
from agent.src.services.wellbeing.child_support_safety_intent import (
    ChildSupportSafetyIntent,
)
from agent.src.services.wellbeing.child_support_safety_policy import (
    ChildSupportSafetyAssessment,
    ChildSupportSafetyPolicy,
)
from agent.src.services.wellbeing.summary_service import WellbeingSummaryService
from shared.src.contracts.child_support_contracts import (
    ChildSupportAssistantMode,
    ChildSupportConversationRequestPayload,
    ChildSupportConversationResponsePayload,
    ChildSupportProactivePromptPayload,
    ChildSupportSafetyLevel,
    ChildSupportScopeStatus,
    ChildSupportSuggestionPayload,
)
from shared.src.contracts.wellbeing_signal_contracts import WellbeingSignalLevel

DISCLOSURE_NOTICE = (
    "TraceMind는 진단이나 상담을 대신하지 않습니다. 위험하거나 혼자 감당하기 "
    "어려운 상황이면 지금 바로 보호자나 믿을 수 있는 어른에게 알려 주세요."
)


@dataclass(slots=True)
class ChildSupportCoachService:
    """로컬 agent가 소유하는 아이용 지원 대화 service.

    LLM provider를 붙여도 provider는 이 서비스 뒤쪽 adapter로만 들어온다.
    UI는 shared payload만 소비하고, raw 대화/검색 맥락은 agent 로컬에 남긴다.
    """

    summary_service: WellbeingSummaryService | None = None
    conversation_repository: ChildSupportConversationRepository | None = None
    context_provider: ChildSupportContextProvider | None = None
    llm_provider: ChildSupportLlmProvider | None = None
    safety_policy: ChildSupportSafetyPolicy = field(
        default_factory=ChildSupportSafetyPolicy
    )
    response_policy: ChildSupportResponsePolicy = field(
        default_factory=ChildSupportResponsePolicy
    )
    fallback_assistant_mode: ChildSupportAssistantMode = (
        ChildSupportAssistantMode.LOCAL_GUARDED
    )

    def __post_init__(self) -> None:
        if self.context_provider is None:
            self.context_provider = ChildSupportContextProvider(
                summary_service=self.summary_service,
                conversation_repository=self.conversation_repository,
            )

    def create_response(
        self,
        request: ChildSupportConversationRequestPayload,
    ) -> ChildSupportConversationResponsePayload:
        """단일 child-support turn 응답을 만든다."""

        conversation_id = request.conversation_id or _new_conversation_id()
        context = self.context_provider.build(conversation_id)
        assessment = self.safety_policy.assess(
            message=request.message,
            context=context,
        )
        child_message_id = _new_message_id("child")
        self._save_message(
            ChildSupportMessageRecord(
                message_id=child_message_id,
                conversation_id=conversation_id,
                role="child",
                text=request.message,
                created_at=datetime.now(tz=timezone.utc),
                safety_level=assessment.safety_level.value,
                scope_status=assessment.scope_status.value,
            )
        )

        reply_text, assistant_mode = self._build_reply(
            message=request.message,
            context=context,
            assessment=assessment,
        )
        assistant_message_id = _new_message_id("assistant")
        created_at = datetime.now(tz=timezone.utc)
        self._save_message(
            ChildSupportMessageRecord(
                message_id=assistant_message_id,
                conversation_id=conversation_id,
                role="assistant",
                text=reply_text,
                created_at=created_at,
                safety_level=assessment.safety_level.value,
                assistant_mode=assistant_mode.value,
                scope_status=assessment.scope_status.value,
                metadata={
                    "assessment_intent": assessment.intent.value,
                    "assessment_reason": assessment.reason,
                    "immediate_danger": assessment.immediate_danger,
                },
            )
        )

        return ChildSupportConversationResponsePayload(
            conversation_id=conversation_id,
            message_id=assistant_message_id,
            created_at=created_at,
            assistant_mode=assistant_mode,
            safety_level=assessment.safety_level,
            scope_status=assessment.scope_status,
            reply_text=reply_text,
            suggested_prompts=_build_suggestions(assessment),
            parent_handoff_suggested=assessment.parent_handoff_suggested,
            parent_handoff_label=_build_parent_handoff_label(assessment),
            disclosure_notice=DISCLOSURE_NOTICE,
        )

    def build_proactive_prompt(self) -> ChildSupportProactivePromptPayload:
        """아이 화면 진입 시 먼저 말을 걸지 여부를 계산한다.

        데이터가 없거나 위험도가 낮으면 아무 문구도 반환하지 않는다.
        """

        context = self.context_provider.build(_new_conversation_id())
        summary = context.wellbeing_summary
        if (
            summary is None
            or summary.low_data
            or not context.wellbeing_summary_is_observed
        ):
            return ChildSupportProactivePromptPayload(should_prompt=False)
        if summary.signal_level not in {
            WellbeingSignalLevel.HIGH,
            WellbeingSignalLevel.VERY_HIGH,
        }:
            return ChildSupportProactivePromptPayload(should_prompt=False)
        safety_level = (
            ChildSupportSafetyLevel.PARENT_HANDOFF
            if summary.signal_level == WellbeingSignalLevel.VERY_HIGH
            else ChildSupportSafetyLevel.CHECK_IN
        )
        prompt_text = (
            "오늘 마음 신호가 평소보다 높게 보여요. 바로 해결책을 말하기보다, "
            "지금 제일 크게 느껴지는 감정 하나만 같이 확인해볼까요?"
        )
        return ChildSupportProactivePromptPayload(
            should_prompt=True,
            safety_level=safety_level,
            prompt_text=prompt_text,
            suggested_prompts=_build_suggestions(
                ChildSupportSafetyAssessment(
                    safety_level=safety_level,
                    scope_status=ChildSupportScopeStatus.IN_SCOPE,
                    intent=ChildSupportSafetyIntent.HIGH_WELLBEING_SUMMARY,
                    reason="proactive_high_wellbeing_summary",
                )
            ),
        )

    def _build_reply(
        self,
        *,
        message: str,
        context: ChildSupportConversationContext,
        assessment: ChildSupportSafetyAssessment,
    ) -> tuple[str, ChildSupportAssistantMode]:
        plan = self.response_policy.build_plan(
            message=message,
            assessment=assessment,
        )
        if assessment.scope_status == ChildSupportScopeStatus.REDIRECTED:
            return plan.fallback_text, self.fallback_assistant_mode
        if assessment.immediate_danger:
            return plan.fallback_text, self.fallback_assistant_mode
        if not plan.allow_llm_rewrite:
            return plan.fallback_text, self.fallback_assistant_mode
        if self.llm_provider is None:
            return plan.fallback_text, self.fallback_assistant_mode

        prompt = _build_llm_prompt(
            message=message,
            context=context,
            assessment=assessment,
            plan=plan,
        )
        try:
            reply = self.llm_provider.generate_reply(prompt=prompt)
        except (ChildSupportLlmError, OSError, RuntimeError):
            return plan.fallback_text, self.fallback_assistant_mode
        processed_reply = _postprocess_llm_reply(reply, assessment)
        if not plan.accepts(processed_reply):
            return plan.fallback_text, self.fallback_assistant_mode
        return (
            processed_reply,
            self.llm_provider.assistant_mode,
        )

    def _save_message(self, record: ChildSupportMessageRecord) -> None:
        if self.conversation_repository is None:
            return
        self.conversation_repository.save_message(record)


def _build_llm_prompt(
    *,
    message: str,
    context: ChildSupportConversationContext,
    assessment: ChildSupportSafetyAssessment,
    plan: ChildSupportResponsePlan,
) -> str:
    summary = context.wellbeing_summary
    summary_block = "현재 wellbeing summary: 없음"
    if summary is not None:
        summary_block = (
            "현재 wellbeing summary:\n"
            f"- level: {summary.signal_level.value}\n"
            f"- label: {summary.signal_label}\n"
            f"- score: {summary.signal_score:.1f}\n"
            f"- trend: {summary.trend.value}\n"
            f"- summary: {summary.summary}\n"
            f"- action_tip: {summary.action_tip}"
        )

    query_lines = [
        (
            f"- {item.occurred_at.isoformat()} | "
            f"query='{item.raw_text_excerpt}' | "
            f"label={item.predicted_label} | confidence={item.confidence}"
        )
        for item in context.recent_queries
    ]
    query_block = (
        "최근 로컬 query buffer 요약:\n" + "\n".join(query_lines)
        if query_lines
        else "최근 로컬 query buffer 요약: 없음"
    )

    message_lines = [
        f"- {record.role}: {record.text}" for record in context.recent_messages[-6:]
    ]
    history_block = (
        "현재 대화 최근 히스토리:\n" + "\n".join(message_lines)
        if message_lines
        else "현재 대화 최근 히스토리: 없음"
    )
    moves_block = "\n".join(f"- {move.value}" for move in plan.moves)
    blocked_terms_block = (
        ", ".join(plan.blocked_terms) if plan.blocked_terms else "없음"
    )

    return (
        "너는 TraceMind의 아이용 마음 도움 로컬 상담 코치다.\n"
        "원칙:\n"
        "- 한국어로 3~6문장만 답한다.\n"
        "- 아이에게 말할 때는 일관된 다정한 해요체만 쓴다.\n"
        "- 반말, 과한 높임말, '좋겠어', '어떨까', '하신'처럼 어체가 섞이는 "
        "표현을 쓰지 않는다.\n"
        "- 진단, 치료, 법률 조언처럼 말하지 않는다.\n"
        "- 마음 상태, 학교/가족/친구 관계, 안전한 다음 행동 범위 안에서만 답한다.\n"
        "- 일반적인 '힘들어', '답답해', '우울해' 단계에서는 바로 해결책을 "
        "제시하지 말고 감정 확인 질문 1개와 짧은 안정 행동 1개만 제안한다.\n"
        "- 별개의 지식 질문, 코딩, 숙제 정답, 투자, 레시피는 답하지 말고 "
        "마음 도움 범위로 부드럽게 돌린다.\n"
        "- 자해 방법, 은폐 방법, 구체적 실행 방법은 절대 제공하지 않는다.\n"
        "- safety_level이 supportive 또는 check_in이면 가족, 친구, 부모, "
        "어른에게 말하라는 제안을 먼저 꺼내지 않는다.\n"
        "- safety_level이 parent_handoff 또는 urgent일 때만 보호자/믿을 수 "
        "있는 어른에게 알리는 제안을 한다.\n"
        "- safety_level이 parent_handoff이면 대화를 끝내지 말고, 지금 안전한지 "
        "또는 다친 곳이 있는지 확인하는 질문 1개로 마무리한다.\n"
        "- '안전하고 편안한 시간 보내', '언제든 말해줘', '행복하길 바라'처럼 "
        "대화를 종료하는 표현을 쓰지 않는다.\n"
        "- 보호자에게 말하라는 제안은 강압적으로 쓰지 말고, 아이가 보여줄 "
        "수 있는 한 문장을 같이 만드는 방식으로 제안한다.\n\n"
        f"safety_level: {assessment.safety_level.value}\n"
        f"safety_intent: {assessment.intent.value}\n"
        f"scope_status: {assessment.scope_status.value}\n"
        f"immediate_danger: {assessment.immediate_danger}\n"
        f"assessment_reason: {assessment.reason}\n\n"
        f"response_plan: {plan.name.value}\n"
        f"response_tone: {plan.tone.value}\n"
        "required_moves:\n"
        f"{moves_block}\n"
        f"blocked_terms: {blocked_terms_block}\n"
        "아래 fallback_reference를 그대로 베끼지 말고, required_moves의 순서와 "
        "의미를 지켜 자연스럽게 답한다.\n"
        f"fallback_reference:\n{plan.fallback_text}\n\n"
        f"{summary_block}\n\n"
        f"{query_block}\n\n"
        f"{history_block}\n\n"
        f"아이의 새 메시지: {message}\n\n"
        "아이에게 바로 보여줄 답변만 작성해라."
    )


def _postprocess_llm_reply(
    reply: str,
    assessment: ChildSupportSafetyAssessment,
) -> str:
    normalized = " ".join(reply.split())
    normalized = _remove_conversation_closing_sentences(normalized)
    if not assessment.allows_parent_handoff_language:
        normalized = _remove_parent_handoff_sentences(normalized)
    if len(normalized) > 900:
        normalized = f"{normalized[:900].rstrip()}..."
    if assessment.safety_level == ChildSupportSafetyLevel.URGENT:
        required = "지금 안전한 곳에 있는지 먼저 확인해요."
        if required not in normalized:
            normalized = f"{required} {normalized}"
    return normalized


def _remove_conversation_closing_sentences(reply: str) -> str:
    closing_keywords = (
        "안전하고 편안한 시간",
        "편안한 시간 보내",
        "언제든지 말씀해줘",
        "언제든 말해줘",
        "언제든 이야기해",
        "행복하길",
        "좋은 하루",
    )
    return _drop_sentences_with_keywords(reply, closing_keywords)


def _remove_parent_handoff_sentences(reply: str) -> str:
    blocked_keywords = (
        "가족",
        "부모",
        "엄마",
        "아빠",
        "친구",
        "어른",
        "선생님",
        "상담사",
    )
    return _drop_sentences_with_keywords(reply, blocked_keywords)


def _drop_sentences_with_keywords(reply: str, keywords: tuple[str, ...]) -> str:
    sentences = [
        sentence.strip()
        for sentence in reply.replace("?", "?.").replace("!", "!.").split(".")
        if sentence.strip()
    ]
    kept_sentences = [
        sentence
        for sentence in sentences
        if not any(keyword in sentence for keyword in keywords)
    ]
    return " ".join(kept_sentences).strip()


def _build_suggestions(
    assessment: ChildSupportSafetyAssessment,
) -> tuple[ChildSupportSuggestionPayload, ...]:
    if assessment.intent in {
        ChildSupportSafetyIntent.OTHER_HARM_IDEATION,
        ChildSupportSafetyIntent.OTHER_HARM_METHOD_REQUEST,
    }:
        return (
            ChildSupportSuggestionPayload(
                id="show-adult-harm-risk",
                label="어른에게 보여줄 문장",
                prompt=(
                    "누군가를 해칠까 봐 걱정된다는 말을 어른에게 보여줄 "
                    "문장으로 정리해줘."
                ),
            ),
            ChildSupportSuggestionPayload(
                id="keep-distance-now",
                label="지금 거리 두기",
                prompt="그 친구에게 가지 않기 위해 지금 바로 할 행동을 하나만 골라줘.",
            ),
        )
    if assessment.intent == ChildSupportSafetyIntent.POST_URGENT_DEESCALATION:
        return (
            ChildSupportSuggestionPayload(
                id="continue-after-anger",
                label="그냥 이어 말하기",
                prompt="지금 제일 버거운 부분부터 한 문장으로 이어서 말해볼게.",
            ),
            ChildSupportSuggestionPayload(
                id="hold-with-me",
                label="같이 버티기",
                prompt="지금 복수 생각에서 조금 떨어질 수 있게 짧게 붙잡아줘.",
            ),
        )
    if assessment.intent == ChildSupportSafetyIntent.PEER_RESPONSE_PLANNING:
        return (
            ChildSupportSuggestionPayload(
                id="peer-boundary-line",
                label="상대에게 할 말 정리",
                prompt=(
                    "그 친구에게 바로 보내지 않고, 먼저 내가 하고 싶은 말을 "
                    "짧게 정리해줘."
                ),
            ),
            ChildSupportSuggestionPayload(
                id="safe-distance",
                label="안전한 거리 두기",
                prompt="그 친구와 당분간 어떻게 안전하게 거리를 둘지 같이 정리해줘.",
            ),
        )
    if assessment.intent == ChildSupportSafetyIntent.POST_HANDOFF_EMOTIONAL_FOLLOWUP:
        return (
            ChildSupportSuggestionPayload(
                id="name-post-incident-feeling",
                label="감정 하나 고르기",
                prompt="방금 일 때문에 남은 감정을 쉬운 보기로 골라줘.",
            ),
            ChildSupportSuggestionPayload(
                id="sort-post-incident",
                label="방금 일 정리",
                prompt="방금 있었던 일을 마음이 덜 복잡하게 짧게 정리해줘.",
            ),
        )
    if assessment.scope_status == ChildSupportScopeStatus.REDIRECTED:
        return (
            ChildSupportSuggestionPayload(
                id="bring-back-feeling",
                label="감정으로 다시 말하기",
                prompt="방금 질문 때문에 어떤 감정이 생겼는지 같이 정리해줘.",
            ),
            ChildSupportSuggestionPayload(
                id="small-worry",
                label="걱정 하나만 말하기",
                prompt="지금 제일 작은 걱정 하나부터 말해볼게.",
            ),
        )
    if assessment.safety_level == ChildSupportSafetyLevel.URGENT:
        return (
            ChildSupportSuggestionPayload(
                id="check-safety",
                label="지금 안전한지 확인",
                prompt="지금 내가 안전한 곳에 있는지 확인할 질문을 해줘.",
            ),
            ChildSupportSuggestionPayload(
                id="tell-adult-now",
                label="어른에게 보낼 문장",
                prompt="믿을 수 있는 어른에게 보여줄 짧은 문장을 만들어줘.",
            ),
        )
    if assessment.safety_level == ChildSupportSafetyLevel.PARENT_HANDOFF:
        return (
            ChildSupportSuggestionPayload(
                id="parent-message",
                label="부모님께 말할 문장",
                prompt="부모님께 보여줄 수 있게 지금 일을 짧은 문장으로 정리해줘.",
            ),
            ChildSupportSuggestionPayload(
                id="safe-next-step",
                label="다음 안전 행동",
                prompt="지금 당장 할 수 있는 안전한 다음 행동을 골라줘.",
            ),
        )
    return (
        ChildSupportSuggestionPayload(
            id="breathe",
            label="숨 쉬기 도와줘",
            prompt="지금 30초 동안 따라 할 수 있는 숨 쉬기 방법을 알려줘.",
        ),
        ChildSupportSuggestionPayload(
            id="name-feeling",
            label="감정 이름 붙이기",
            prompt="내가 느끼는 감정을 고를 수 있게 쉬운 보기로 알려줘.",
        ),
        ChildSupportSuggestionPayload(
            id="small-step",
            label="작은 행동 하나",
            prompt="지금 바로 할 수 있는 아주 작은 행동 하나만 골라줘.",
        ),
    )


def _build_parent_handoff_label(
    assessment: ChildSupportSafetyAssessment,
) -> str | None:
    if assessment.safety_level == ChildSupportSafetyLevel.URGENT:
        if assessment.intent in {
            ChildSupportSafetyIntent.OTHER_HARM_IDEATION,
            ChildSupportSafetyIntent.OTHER_HARM_METHOD_REQUEST,
        }:
            return "지금 바로 어른에게 알리고 그 친구와 거리 두기"
        if assessment.immediate_danger:
            return "지금 바로 보호자나 믿을 수 있는 어른에게 알리기"
        return "지금 안전한지 확인하고 어른에게 보여줄 문장 만들기"
    if assessment.safety_level == ChildSupportSafetyLevel.PARENT_HANDOFF:
        return "부모님께 보여줄 짧은 문장 만들기"
    return None


def _new_conversation_id() -> str:
    return f"child-conversation-{uuid.uuid4().hex}"


def _new_message_id(role: str) -> str:
    return f"child-support-{role}-{uuid.uuid4().hex}"
