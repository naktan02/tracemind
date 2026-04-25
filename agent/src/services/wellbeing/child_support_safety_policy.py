"""아이용 지원 대화 안전 라우팅 policy."""

from __future__ import annotations

from dataclasses import dataclass

from agent.src.services.wellbeing.child_support_context_provider import (
    ChildSupportConversationContext,
)
from shared.src.contracts.child_support_contracts import (
    ChildSupportSafetyLevel,
    ChildSupportScopeStatus,
)
from shared.src.contracts.wellbeing_signal_contracts import WellbeingSignalLevel

_SELF_HARM_KEYWORDS = (
    "죽고",
    "자살",
    "해치",
    "사라지고",
    "끝내고",
    "못 살",
    "죽어버",
    "kill myself",
    "suicide",
    "self harm",
)
_IMMEDIATE_DANGER_KEYWORDS = (
    "지금 할",
    "오늘 할",
    "방법",
    "계획",
    "칼",
    "약",
    "옥상",
    "줄",
    "피",
    "how to",
    "plan",
    "tonight",
)
_PARENT_HANDOFF_KEYWORDS = (
    "무서",
    "협박",
    "괴롭",
    "따돌림",
    "폭력",
    "때렸",
    "맞았",
    "도와줘",
    "help",
    "scared",
    "bully",
    "violence",
)
_CALMING_KEYWORDS = (
    "불안",
    "떨려",
    "숨",
    "화나",
    "울고",
    "답답",
    "힘들",
    "무기력",
    "외로",
    "panic",
    "anxious",
    "angry",
)
_SUPPORT_DOMAIN_KEYWORDS = (
    "마음",
    "기분",
    "감정",
    "친구",
    "부모",
    "엄마",
    "아빠",
    "학교",
    "선생",
    "상담",
    "걱정",
    "스트레스",
    "잠",
    "꿈",
    "혼자",
    "무서",
    "슬퍼",
    "불안",
    "우울",
    "힘들",
    "도와",
)
_OFF_TOPIC_KEYWORDS = (
    "파이썬",
    "코딩",
    "수학 문제",
    "레시피",
    "주식",
    "코인",
    "게임 공략",
    "날씨",
    "뉴스",
    "번역",
    "숙제 답",
    "과제 답",
    "python",
    "javascript",
    "recipe",
    "stock",
    "bitcoin",
)


@dataclass(frozen=True, slots=True)
class ChildSupportSafetyAssessment:
    """child-support 응답 생성 전에 확정하는 안전 판단."""

    safety_level: ChildSupportSafetyLevel
    scope_status: ChildSupportScopeStatus
    immediate_danger: bool = False
    reason: str = "supportive"

    @property
    def parent_handoff_suggested(self) -> bool:
        return self.safety_level in {
            ChildSupportSafetyLevel.PARENT_HANDOFF,
            ChildSupportSafetyLevel.URGENT,
        }

    @property
    def allows_parent_handoff_language(self) -> bool:
        """응답에서 보호자/어른 handoff 문구를 직접 써도 되는 단계인지."""

        return self.parent_handoff_suggested


class ChildSupportSafetyPolicy:
    """아이 메시지와 로컬 요약을 바탕으로 안전 단계를 결정한다."""

    def assess(
        self,
        *,
        message: str,
        context: ChildSupportConversationContext,
    ) -> ChildSupportSafetyAssessment:
        normalized = message.lower()
        if _is_off_topic(normalized):
            return ChildSupportSafetyAssessment(
                safety_level=ChildSupportSafetyLevel.SUPPORTIVE,
                scope_status=ChildSupportScopeStatus.REDIRECTED,
                reason="off_topic",
            )

        if any(keyword in normalized for keyword in _SELF_HARM_KEYWORDS):
            return ChildSupportSafetyAssessment(
                safety_level=ChildSupportSafetyLevel.URGENT,
                scope_status=ChildSupportScopeStatus.IN_SCOPE,
                immediate_danger=any(
                    keyword in normalized for keyword in _IMMEDIATE_DANGER_KEYWORDS
                ),
                reason="self_harm_signal",
            )

        if any(keyword in normalized for keyword in _PARENT_HANDOFF_KEYWORDS):
            return ChildSupportSafetyAssessment(
                safety_level=ChildSupportSafetyLevel.PARENT_HANDOFF,
                scope_status=ChildSupportScopeStatus.IN_SCOPE,
                reason="parent_handoff_keyword",
            )

        if any(keyword in normalized for keyword in _CALMING_KEYWORDS):
            return ChildSupportSafetyAssessment(
                safety_level=ChildSupportSafetyLevel.CHECK_IN,
                scope_status=ChildSupportScopeStatus.IN_SCOPE,
                reason="calming_keyword",
            )

        summary = context.wellbeing_summary
        if summary is not None and summary.signal_level in {
            WellbeingSignalLevel.HIGH,
            WellbeingSignalLevel.VERY_HIGH,
        }:
            return ChildSupportSafetyAssessment(
                safety_level=ChildSupportSafetyLevel.CHECK_IN,
                scope_status=ChildSupportScopeStatus.IN_SCOPE,
                reason="high_wellbeing_summary",
            )

        return ChildSupportSafetyAssessment(
            safety_level=ChildSupportSafetyLevel.SUPPORTIVE,
            scope_status=ChildSupportScopeStatus.IN_SCOPE,
            reason="supportive",
        )


def _is_off_topic(normalized_message: str) -> bool:
    has_support_keyword = any(
        keyword in normalized_message for keyword in _SUPPORT_DOMAIN_KEYWORDS
    )
    if has_support_keyword:
        return False
    return any(keyword in normalized_message for keyword in _OFF_TOPIC_KEYWORDS)


__all__ = [
    "ChildSupportSafetyAssessment",
    "ChildSupportSafetyPolicy",
]
