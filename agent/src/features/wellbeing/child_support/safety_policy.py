"""아이용 지원 대화 안전 라우팅 policy."""

from __future__ import annotations

from dataclasses import dataclass

from agent.src.contracts.child_support_contracts import (
    ChildSupportSafetyLevel,
    ChildSupportScopeStatus,
)
from agent.src.contracts.wellbeing_signal_contracts import WellbeingSignalLevel
from agent.src.features.wellbeing.child_support.context_provider import (
    ChildSupportConversationContext,
)

OFF_TOPIC = "off_topic"
SELF_HARM_SIGNAL = "self_harm_signal"
OTHER_HARM_IDEATION = "other_harm_ideation"
OTHER_HARM_METHOD_REQUEST = "other_harm_method_request"
POST_URGENT_DEESCALATION = "post_urgent_deescalation"
PARENT_HANDOFF_KEYWORD = "parent_handoff_keyword"
POST_HANDOFF_EMOTIONAL_FOLLOWUP = "post_handoff_emotional_followup"
PEER_RESPONSE_PLANNING = "peer_response_planning"
CALMING_KEYWORD = "calming_keyword"
HIGH_WELLBEING_SUMMARY = "high_wellbeing_summary"
SUPPORTIVE = "supportive"

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
_OTHER_HARM_KEYWORDS = (
    "죽여",
    "죽일",
    "죽이",
    "살해",
    "해치고",
    "해치려",
    "패버",
    "때려버",
    "kill him",
    "kill her",
    "kill them",
    "murder",
)
_IMMEDIATE_DANGER_KEYWORDS = (
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
    "속상",
    "억울",
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
    intent: str = SUPPORTIVE
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

        if any(keyword in normalized for keyword in _OTHER_HARM_KEYWORDS):
            immediate_danger = any(
                keyword in normalized for keyword in _IMMEDIATE_DANGER_KEYWORDS
            )
            return ChildSupportSafetyAssessment(
                safety_level=ChildSupportSafetyLevel.URGENT,
                scope_status=ChildSupportScopeStatus.IN_SCOPE,
                intent=(
                    OTHER_HARM_METHOD_REQUEST
                    if immediate_danger
                    else OTHER_HARM_IDEATION
                ),
                immediate_danger=immediate_danger,
            )

        if any(keyword in normalized for keyword in _SELF_HARM_KEYWORDS):
            return ChildSupportSafetyAssessment(
                safety_level=ChildSupportSafetyLevel.URGENT,
                scope_status=ChildSupportScopeStatus.IN_SCOPE,
                intent=SELF_HARM_SIGNAL,
                immediate_danger=any(
                    keyword in normalized for keyword in _IMMEDIATE_DANGER_KEYWORDS
                ),
            )

        if (
            context.conversation_state.has_recent_other_harm_risk
            and _is_post_urgent_distress_followup(normalized)
        ):
            return ChildSupportSafetyAssessment(
                safety_level=ChildSupportSafetyLevel.CHECK_IN,
                scope_status=ChildSupportScopeStatus.IN_SCOPE,
                intent=POST_URGENT_DEESCALATION,
            )

        if any(keyword in normalized for keyword in _PARENT_HANDOFF_KEYWORDS):
            return ChildSupportSafetyAssessment(
                safety_level=ChildSupportSafetyLevel.PARENT_HANDOFF,
                scope_status=ChildSupportScopeStatus.IN_SCOPE,
                intent=PARENT_HANDOFF_KEYWORD,
            )

        if context.conversation_state.has_recent_parent_handoff:
            if _is_peer_response_question(normalized):
                return ChildSupportSafetyAssessment(
                    safety_level=ChildSupportSafetyLevel.CHECK_IN,
                    scope_status=ChildSupportScopeStatus.IN_SCOPE,
                    intent=PEER_RESPONSE_PLANNING,
                )
            if _is_post_handoff_followup(normalized):
                return ChildSupportSafetyAssessment(
                    safety_level=ChildSupportSafetyLevel.CHECK_IN,
                    scope_status=ChildSupportScopeStatus.IN_SCOPE,
                    intent=POST_HANDOFF_EMOTIONAL_FOLLOWUP,
                )

        if any(keyword in normalized for keyword in _CALMING_KEYWORDS):
            return ChildSupportSafetyAssessment(
                safety_level=ChildSupportSafetyLevel.CHECK_IN,
                scope_status=ChildSupportScopeStatus.IN_SCOPE,
                intent=CALMING_KEYWORD,
            )

        summary = context.wellbeing_summary
        if summary is not None and summary.signal_level in {
            WellbeingSignalLevel.HIGH,
            WellbeingSignalLevel.VERY_HIGH,
        }:
            return ChildSupportSafetyAssessment(
                safety_level=ChildSupportSafetyLevel.CHECK_IN,
                scope_status=ChildSupportScopeStatus.IN_SCOPE,
                intent=HIGH_WELLBEING_SUMMARY,
            )

        return ChildSupportSafetyAssessment(
            safety_level=ChildSupportSafetyLevel.SUPPORTIVE,
            scope_status=ChildSupportScopeStatus.IN_SCOPE,
            intent=SUPPORTIVE,
        )


def _is_off_topic(normalized_message: str) -> bool:
    has_support_keyword = any(
        keyword in normalized_message for keyword in _SUPPORT_DOMAIN_KEYWORDS
    )
    if has_support_keyword:
        return False
    return any(keyword in normalized_message for keyword in _OFF_TOPIC_KEYWORDS)


def _is_peer_response_question(normalized_message: str) -> bool:
    """폭력 사건 뒤 상대에게 어떻게 대응할지 묻는 후속 질문인지 본다."""

    response_keywords = (
        "어떻게",
        "어쩌",
        "대응",
        "말해야",
        "말할",
        "해야",
        "하면",
        "걔한테",
        "그 친구",
        "사과",
        "거리",
        "복수",
        "따져",
    )
    return any(keyword in normalized_message for keyword in response_keywords)


def _is_post_handoff_followup(normalized_message: str) -> bool:
    """폭력 사건 뒤 안전 확인 또는 감정 정리로 넘어갈 수 있는지 본다."""

    safe_update_keywords = (
        "떨어졌",
        "떨어져",
        "집",
        "집에",
        "안전",
        "피했",
        "나왔",
    )
    emotional_keywords = (
        "속상",
        "억울",
        "화나",
        "무서",
        "슬퍼",
        "힘들",
        "답답",
        "울",
    )
    return any(
        keyword in normalized_message
        for keyword in safe_update_keywords + emotional_keywords
    )


def _is_post_urgent_distress_followup(normalized_message: str) -> bool:
    """위해 충동 직후 이어지는 힘듦/무너짐 표현인지 본다."""

    distress_keywords = (
        "힘들",
        "힘든",
        "버거",
        "괴로",
        "무너",
        "모르겠",
        "답답",
        "속상",
        "억울",
        "화가",
        "화나",
        "눈물",
        "울",
        "너무",
    )
    return any(keyword in normalized_message for keyword in distress_keywords)
