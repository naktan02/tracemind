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
from agent.src.services.wellbeing.child_support_safety_policy import (
    ChildSupportSafetyAssessment,
    ChildSupportSafetyPolicy,
)
from agent.src.services.wellbeing.summary_service import WellbeingSummaryService
from shared.src.contracts.child_support_contracts import (
    ChildSupportAssistantMode,
    ChildSupportConversationRequestPayload,
    ChildSupportConversationResponsePayload,
    ChildSupportSafetyLevel,
    ChildSupportScopeStatus,
    ChildSupportSuggestionPayload,
)

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

    def _build_reply(
        self,
        *,
        message: str,
        context: ChildSupportConversationContext,
        assessment: ChildSupportSafetyAssessment,
    ) -> tuple[str, ChildSupportAssistantMode]:
        if assessment.scope_status == ChildSupportScopeStatus.REDIRECTED:
            return _build_scope_redirect_reply(), self.fallback_assistant_mode
        if assessment.immediate_danger:
            return _build_guarded_reply_text(
                message=message,
                context=context,
                assessment=assessment,
            ), self.fallback_assistant_mode
        if self.llm_provider is None:
            return _build_guarded_reply_text(
                message=message,
                context=context,
                assessment=assessment,
            ), self.fallback_assistant_mode

        prompt = _build_llm_prompt(
            message=message,
            context=context,
            assessment=assessment,
        )
        try:
            reply = self.llm_provider.generate_reply(prompt=prompt)
        except (ChildSupportLlmError, OSError, RuntimeError):
            return _build_guarded_reply_text(
                message=message,
                context=context,
                assessment=assessment,
            ), self.fallback_assistant_mode
        return (
            _postprocess_llm_reply(reply, assessment),
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

    return (
        "너는 TraceMind의 아이용 마음 도움 로컬 상담 코치다.\n"
        "원칙:\n"
        "- 한국어로 3~6문장만 답한다.\n"
        "- 진단, 치료, 법률 조언처럼 말하지 않는다.\n"
        "- 마음 상태, 학교/가족/친구 관계, 안전한 다음 행동 범위 안에서만 답한다.\n"
        "- 별개의 지식 질문, 코딩, 숙제 정답, 투자, 레시피는 답하지 말고 "
        "마음 도움 범위로 부드럽게 돌린다.\n"
        "- 자해 방법, 은폐 방법, 구체적 실행 방법은 절대 제공하지 않는다.\n"
        "- 위험 신호가 있으면 먼저 공감하고, 지금 안전한 장소인지 확인하고, "
        "믿을 수 있는 어른에게 알리도록 제안한다.\n"
        "- 부모에게 말하라는 제안은 강압적으로 쓰지 말고, 아이가 보여줄 수 "
        "있는 한 문장을 같이 만드는 방식으로 제안한다.\n\n"
        f"safety_level: {assessment.safety_level.value}\n"
        f"scope_status: {assessment.scope_status.value}\n"
        f"immediate_danger: {assessment.immediate_danger}\n"
        f"assessment_reason: {assessment.reason}\n\n"
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
    if len(normalized) > 900:
        normalized = f"{normalized[:900].rstrip()}..."
    if assessment.safety_level == ChildSupportSafetyLevel.URGENT:
        required = "지금 안전한 곳에 있는지 먼저 확인해요."
        if required not in normalized:
            normalized = f"{required} {normalized}"
    return normalized


def _build_scope_redirect_reply() -> str:
    return (
        "그 질문은 여기서 자세히 답하기보다, 마음이나 안전과 관련된 이야기로 "
        "다시 가져오는 게 좋아요. 지금 그 질문 때문에 답답함, 걱정, 불안 같은 "
        "감정이 생겼다면 그 부분부터 같이 정리해볼게요."
    )


def _build_guarded_reply_text(
    *,
    message: str,
    context: ChildSupportConversationContext,
    assessment: ChildSupportSafetyAssessment,
) -> str:
    if assessment.safety_level == ChildSupportSafetyLevel.URGENT:
        if assessment.immediate_danger:
            return (
                "지금은 혼자 버티면서 대화만 이어가기보다 안전을 먼저 확인해야 "
                "해요. 지금 안전한 곳에 있는지 보고, 바로 가까운 보호자나 믿을 "
                "수 있는 어른에게 이 문장을 보여주세요: '지금 혼자 있기 위험해서 "
                "도움이 필요해요.'"
            )
        return (
            "그 말을 꺼내준 건 정말 중요한 신호예요. 바로 해결책을 찾기보다 "
            "먼저 지금 안전한 곳에 있는지 확인하고 싶어요. 혼자 있다면 문을 "
            "열어둘 수 있는 곳이나 사람이 있는 곳으로 이동하고, 믿을 수 있는 "
            "어른에게 '지금 너무 위험한 생각이 들어서 같이 있어줬으면 좋겠어'라고 "
            "보여줄 문장을 같이 만들 수 있어요."
        )
    if assessment.safety_level == ChildSupportSafetyLevel.PARENT_HANDOFF:
        return (
            "말해줘서 고마워요. 이건 혼자 해결하려고 버티기보다 어른이 같이 "
            "확인해 주는 편이 안전해 보여요. 지금 느낀 일을 한 문장으로 적고, "
            "부모님이나 믿을 수 있는 어른에게 보여줄 준비를 해볼게요."
        )
    if assessment.safety_level == ChildSupportSafetyLevel.CHECK_IN:
        return (
            "지금 마음이 꽤 크게 흔들린 것 같아요. 먼저 숨을 천천히 세 번만 "
            "같이 쉬어볼게요. 그 다음에는 무엇이 제일 크게 느껴졌는지 하나만 "
            "골라서 말해줘도 괜찮아요."
        )

    summary = context.wellbeing_summary
    context_text = (
        f"현재 상태 카드에는 '{summary.signal_label}'로 표시되어 있어요. "
        if summary is not None
        else ""
    )
    trimmed = " ".join(message.split())
    return (
        f"{context_text}말해줘서 고마워요. 지금은 정답을 바로 찾기보다, 방금 말한 "
        f"'{trimmed[:42]}'에서 어떤 감정이 가장 컸는지부터 천천히 나눠볼게요."
    )


def _build_suggestions(
    assessment: ChildSupportSafetyAssessment,
) -> tuple[ChildSupportSuggestionPayload, ...]:
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


__all__ = ["ChildSupportCoachService"]
