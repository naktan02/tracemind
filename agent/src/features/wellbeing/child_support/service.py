"""아이용 지원 대화 응답 서비스."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from agent.src.contracts.child_support_contracts import (
    ChildSupportAssistantMode,
    ChildSupportConversationRequestPayload,
    ChildSupportConversationResponsePayload,
    ChildSupportProactivePromptClaimRequestPayload,
    ChildSupportProactivePromptPayload,
    ChildSupportSafetyLevel,
    ChildSupportScopeStatus,
)
from agent.src.contracts.wellbeing_signal_contracts import WellbeingSignalLevel
from agent.src.features.wellbeing.child_support.context_provider import (
    ChildSupportContextProvider,
    ChildSupportConversationContext,
)
from agent.src.features.wellbeing.child_support.llm_prompt import (
    build_child_support_conversation_prompt,
    build_child_support_proactive_prompt,
)
from agent.src.features.wellbeing.child_support.llm_provider import (
    ChildSupportLlmError,
    ChildSupportLlmProvider,
)
from agent.src.features.wellbeing.signal.summary_service import WellbeingSummaryService
from agent.src.features.wellbeing.storage.child_support_repository import (
    ChildSupportConversationRepository,
    ChildSupportMessageRecord,
    ChildSupportProactivePromptClaimRecord,
)

DISCLOSURE_NOTICE = (
    "TraceMind는 진단이나 상담을 대신하지 않습니다. 위험하거나 혼자 감당하기 "
    "어려운 상황이면 지금 바로 보호자나 믿을 수 있는 어른에게 알려 주세요."
)
PROACTIVE_PROMPT_SCORE_THRESHOLD = 70.0
PROACTIVE_HIGH_RISK_EVIDENCE_TOPIC = "자해/죽음 관련 표현"
NO_STATIC_FALLBACK_MESSAGE = "AI 응답을 만들지 못했습니다. LLM 설정을 확인하세요."
URGENT_RISK = "urgent_risk"
GENERAL_SUPPORT = "general_support"
_URGENT_RISK_PHRASES = (
    "죽고 싶",
    "죽고싶",
    "자살",
    "자해",
    "죽여",
    "죽일",
    "죽이",
    "살해",
    "kill myself",
    "suicide",
    "self harm",
    "kill him",
    "kill her",
    "kill them",
    "murder",
)


class ChildSupportReplyUnavailable(RuntimeError):
    """정해진 fallback 없이 LLM 응답을 만들 수 없을 때 발생한다."""


@dataclass(frozen=True, slots=True)
class ChildSupportMessageSafety:
    """대화 중 즉시 위험 표현만 표시하는 최소 safety hint."""

    safety_level: ChildSupportSafetyLevel = ChildSupportSafetyLevel.SUPPORTIVE
    scope_status: ChildSupportScopeStatus = ChildSupportScopeStatus.IN_SCOPE
    reason: str = GENERAL_SUPPORT

    @property
    def parent_handoff_suggested(self) -> bool:
        return self.safety_level == ChildSupportSafetyLevel.URGENT


@dataclass(frozen=True, slots=True)
class ChildSupportProactivePromptCandidate:
    """side effect 없는 proactive prompt 후보."""

    prompt_id: str
    context: ChildSupportConversationContext
    safety_level: ChildSupportSafetyLevel


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
        assessment = _assess_message_safety(request.message)
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
            suggested_prompts=(),
            parent_handoff_suggested=assessment.parent_handoff_suggested,
            parent_handoff_label=None,
            disclosure_notice=DISCLOSURE_NOTICE,
        )

    def build_proactive_prompt(self) -> ChildSupportProactivePromptPayload:
        """아이 화면 진입 시 먼저 말을 걸지 여부를 side effect 없이 계산한다.

        데이터가 없거나 위험도가 낮으면 아무 문구도 반환하지 않는다.
        """

        candidate = self._build_proactive_candidate()
        if candidate is None:
            return ChildSupportProactivePromptPayload(should_prompt=False)
        existing_claim = self._load_proactive_claim(candidate.prompt_id)
        return ChildSupportProactivePromptPayload(
            should_prompt=True,
            prompt_id=candidate.prompt_id,
            conversation_id=(
                None if existing_claim is None else existing_claim.conversation_id
            ),
            safety_level=candidate.safety_level,
            prompt_text=None if existing_claim is None else existing_claim.prompt_text,
            suggested_prompts=(),
        )

    def claim_proactive_prompt(
        self,
        request: ChildSupportProactivePromptClaimRequestPayload,
    ) -> ChildSupportProactivePromptPayload:
        """선제 발화를 실제 표시 직전에 conversation으로 claim한다."""

        existing_claim = self._load_proactive_claim(request.prompt_id)
        if existing_claim is not None:
            return ChildSupportProactivePromptPayload(
                should_prompt=True,
                prompt_id=request.prompt_id,
                conversation_id=existing_claim.conversation_id,
                safety_level=(
                    ChildSupportSafetyLevel(existing_claim.safety_level)
                    if existing_claim.safety_level is not None
                    else None
                ),
                prompt_text=existing_claim.prompt_text,
                suggested_prompts=(),
            )

        candidate = self._build_proactive_candidate()
        if candidate is None or candidate.prompt_id != request.prompt_id:
            return ChildSupportProactivePromptPayload(should_prompt=False)
        prompt_text = self._build_proactive_reply(
            context=candidate.context,
            safety_level=candidate.safety_level,
        )
        if prompt_text is None:
            return ChildSupportProactivePromptPayload(should_prompt=False)

        conversation_id = _new_conversation_id()
        created_at = datetime.now(tz=timezone.utc)
        message_id = _new_message_id("assistant")
        message = ChildSupportMessageRecord(
            message_id=message_id,
            conversation_id=conversation_id,
            role="assistant",
            text=prompt_text,
            created_at=created_at,
            safety_level=candidate.safety_level.value,
            assistant_mode=self.llm_provider.assistant_mode.value
            if self.llm_provider is not None
            else None,
            scope_status=ChildSupportScopeStatus.IN_SCOPE.value,
            metadata={
                "source": "proactive_prompt",
                "prompt_id": candidate.prompt_id,
            },
        )
        claim = self._claim_proactive_prompt(
            message=message,
            claim=ChildSupportProactivePromptClaimRecord(
                prompt_id=candidate.prompt_id,
                conversation_id=conversation_id,
                message_id=message_id,
                claimed_at=created_at,
                prompt_text=prompt_text,
                safety_level=candidate.safety_level.value,
                metadata={
                    "source": "proactive_prompt",
                },
            ),
        )
        safety_level = (
            ChildSupportSafetyLevel(claim.safety_level)
            if claim.safety_level is not None
            else candidate.safety_level
        )
        return ChildSupportProactivePromptPayload(
            should_prompt=True,
            prompt_id=claim.prompt_id,
            conversation_id=claim.conversation_id,
            safety_level=safety_level,
            prompt_text=claim.prompt_text,
            suggested_prompts=(),
        )

    def _claim_proactive_prompt(
        self,
        *,
        message: ChildSupportMessageRecord,
        claim: ChildSupportProactivePromptClaimRecord,
    ) -> ChildSupportProactivePromptClaimRecord:
        if self.conversation_repository is None:
            self._save_message(message)
            return claim
        return self.conversation_repository.claim_proactive_prompt(
            message=message,
            claim=claim,
        )

    def _build_reply(
        self,
        *,
        message: str,
        context: ChildSupportConversationContext,
        assessment: ChildSupportMessageSafety,
    ) -> tuple[str, ChildSupportAssistantMode]:
        if self.llm_provider is None:
            raise ChildSupportReplyUnavailable(NO_STATIC_FALLBACK_MESSAGE)

        prompt = build_child_support_conversation_prompt(
            message=message,
            context=context,
            assessment=assessment,
        )
        try:
            reply = self.llm_provider.generate_reply(prompt=prompt)
        except (ChildSupportLlmError, OSError, RuntimeError):
            raise ChildSupportReplyUnavailable(NO_STATIC_FALLBACK_MESSAGE)
        processed_reply = _normalize_llm_reply(reply, max_chars=900)
        if not processed_reply:
            raise ChildSupportReplyUnavailable(NO_STATIC_FALLBACK_MESSAGE)
        return (
            processed_reply,
            self.llm_provider.assistant_mode,
        )

    def _save_message(self, record: ChildSupportMessageRecord) -> None:
        if self.conversation_repository is None:
            return
        self.conversation_repository.save_message(record)

    def _load_proactive_claim(
        self,
        prompt_id: str,
    ) -> ChildSupportProactivePromptClaimRecord | None:
        if self.conversation_repository is None:
            return None
        return self.conversation_repository.get_proactive_prompt_claim(prompt_id)

    def _assess(
        self,
        *,
        message: str,
        context: ChildSupportConversationContext,
    ) -> ChildSupportMessageSafety:
        return _assess_message_safety(message)

    def _build_proactive_reply(
        self,
        *,
        context: ChildSupportConversationContext,
        safety_level: ChildSupportSafetyLevel,
    ) -> str | None:
        if self.llm_provider is None:
            return None
        prompt = build_child_support_proactive_prompt(
            context=context,
            safety_level=safety_level,
        )
        try:
            reply = self.llm_provider.generate_reply(prompt=prompt)
        except (ChildSupportLlmError, OSError, RuntimeError):
            return None
        processed = _normalize_llm_reply(reply, max_chars=420)
        return processed or None

    def _build_proactive_candidate(
        self,
    ) -> ChildSupportProactivePromptCandidate | None:
        context = self.context_provider.build(_new_conversation_id())
        summary = context.wellbeing_summary
        if (
            summary is None
            or summary.low_data
            or not context.wellbeing_summary_is_observed
        ):
            return None
        has_high_risk_evidence = _has_high_risk_evidence(
            context.wellbeing_context_notes
        )
        if (
            summary.signal_score < PROACTIVE_PROMPT_SCORE_THRESHOLD
            and not has_high_risk_evidence
        ):
            return None
        safety_level = (
            ChildSupportSafetyLevel.PARENT_HANDOFF
            if summary.signal_level == WellbeingSignalLevel.VERY_HIGH
            or has_high_risk_evidence
            else ChildSupportSafetyLevel.CHECK_IN
        )
        return ChildSupportProactivePromptCandidate(
            prompt_id=_build_proactive_prompt_id(
                context=context,
                safety_level=safety_level,
            ),
            context=context,
            safety_level=safety_level,
        )


def _first_note_with_prefix(notes: tuple[str, ...], prefix: str) -> str | None:
    for note in notes:
        if note.startswith(prefix):
            return note
    return None


def _has_high_risk_evidence(notes: tuple[str, ...]) -> bool:
    topic_note = _first_note_with_prefix(notes, "반복 주제:")
    return topic_note is not None and PROACTIVE_HIGH_RISK_EVIDENCE_TOPIC in topic_note


def _normalize_llm_reply(reply: str, *, max_chars: int) -> str:
    normalized = " ".join(reply.split())
    if len(normalized) > max_chars:
        normalized = f"{normalized[:max_chars].rstrip()}..."
    return normalized


def _build_proactive_prompt_id(
    *,
    context: ChildSupportConversationContext,
    safety_level: ChildSupportSafetyLevel,
) -> str:
    summary = context.wellbeing_summary
    stable_parts = {
        "safety_level": safety_level.value,
        "summary_computed_at": None
        if summary is None
        else summary.computed_at.isoformat(),
        "summary_score": None if summary is None else round(summary.signal_score, 2),
        "summary_level": None if summary is None else summary.signal_level.value,
        "notes": list(context.wellbeing_context_notes),
    }
    digest = hashlib.sha256(
        json.dumps(stable_parts, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"child-proactive-{digest}"


def _assess_message_safety(message: str) -> ChildSupportMessageSafety:
    normalized = message.lower()
    if not _has_any(normalized, _URGENT_RISK_PHRASES):
        return ChildSupportMessageSafety()
    return ChildSupportMessageSafety(
        safety_level=ChildSupportSafetyLevel.URGENT,
        reason=URGENT_RISK,
    )


def _has_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _new_conversation_id() -> str:
    return f"child-conversation-{uuid.uuid4().hex}"


def _new_message_id(role: str) -> str:
    return f"child-support-{role}-{uuid.uuid4().hex}"
