"""아이용 지원 대화 안전 라우팅 policy."""

from __future__ import annotations

from dataclasses import dataclass

from agent.src.contracts.child_support_contracts import (
    ChildSupportSafetyLevel,
    ChildSupportScopeStatus,
)
from agent.src.features.wellbeing.child_support.context_provider import (
    ChildSupportConversationContext,
)

OFF_TOPIC = "off_topic"
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
_IMMEDIATE_DANGER_PHRASES = (
    "지금 할",
    "오늘 할",
    "어떻게",
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
    "속상",
    "억울",
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
    intent: str = GENERAL_SUPPORT
    immediate_danger: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.reason:
            object.__setattr__(self, "reason", self.intent)

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
                intent=OFF_TOPIC,
            )

        if _has_any(normalized, _URGENT_RISK_PHRASES):
            return ChildSupportSafetyAssessment(
                safety_level=ChildSupportSafetyLevel.URGENT,
                scope_status=ChildSupportScopeStatus.IN_SCOPE,
                intent=URGENT_RISK,
                immediate_danger=_has_any(normalized, _IMMEDIATE_DANGER_PHRASES),
            )

        return ChildSupportSafetyAssessment(
            safety_level=ChildSupportSafetyLevel.SUPPORTIVE,
            scope_status=ChildSupportScopeStatus.IN_SCOPE,
            intent=GENERAL_SUPPORT,
        )


def _is_off_topic(normalized_message: str) -> bool:
    has_support_keyword = _has_any(normalized_message, _SUPPORT_DOMAIN_KEYWORDS)
    if has_support_keyword:
        return False
    return _has_any(normalized_message, _OFF_TOPIC_KEYWORDS)


def _has_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)
