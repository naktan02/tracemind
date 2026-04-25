"""아이용 지원 대화 응답 전략 policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from agent.src.services.wellbeing.child_support_safety_intent import (
    ChildSupportSafetyIntent,
)
from agent.src.services.wellbeing.child_support_safety_policy import (
    ChildSupportSafetyAssessment,
)
from shared.src.contracts.child_support_contracts import (
    ChildSupportSafetyLevel,
    ChildSupportScopeStatus,
)


class ChildSupportResponseStrategyName(StrEnum):
    """child-support 응답을 어떤 구조로 만들지 나타내는 전략 이름."""

    SCOPE_REDIRECT = "scope_redirect"
    SUPPORTIVE_REFLECTION = "supportive_reflection"
    CHECK_IN = "check_in"
    POST_URGENT_DEESCALATION = "post_urgent_deescalation"
    POST_INCIDENT_EMOTIONAL_FOLLOWUP = "post_incident_emotional_followup"
    PEER_RESPONSE_PLANNING = "peer_response_planning"
    SAFETY_CHECK = "safety_check"
    HARM_TO_OTHERS_SAFETY = "harm_to_others_safety"
    URGENT_SAFETY = "urgent_safety"


@dataclass(frozen=True, slots=True)
class ChildSupportResponseStrategy:
    """LLM에 넘기거나 fallback으로 쓸 응답 skeleton."""

    name: ChildSupportResponseStrategyName
    fallback_text: str
    required_terms: tuple[str, ...] = ()
    required_any_term_groups: tuple[tuple[str, ...], ...] = ()
    blocked_terms: tuple[str, ...] = ()
    allow_llm_rewrite: bool = True

    def accepts(self, reply: str) -> bool:
        """LLM 응답이 이 strategy를 지키는지 확인한다."""

        if not reply.strip():
            return False
        if any(term in reply for term in self.blocked_terms):
            return False
        if not all(term in reply for term in self.required_terms):
            return False
        return all(
            any(term in reply for term in term_group)
            for term_group in self.required_any_term_groups
        )


_COMMON_BLOCKED_TERMS = (
    "좋겠어",
    "어떨까",
    "해보자",
    "쉬어보자",
    "거야",
    "하신",
    "도착하신",
    "언제든지 말해줘",
    "언제든 말해줘",
    "행복하길",
    "좋은 하루",
    "뭘 고르라고",
    "고르라고",
    "받아줄게요",
    "내가 바로 해결책",
    "바로 해결책",
    "굳이 정확히 설명",
)


class ChildSupportResponsePolicy:
    """safety assessment를 실제 응답 전략으로 바꾼다."""

    def build_strategy(
        self,
        *,
        message: str,
        assessment: ChildSupportSafetyAssessment,
    ) -> ChildSupportResponseStrategy:
        """현재 safety 단계에 맞는 응답 skeleton을 만든다."""

        if assessment.scope_status == ChildSupportScopeStatus.REDIRECTED:
            return ChildSupportResponseStrategy(
                name=ChildSupportResponseStrategyName.SCOPE_REDIRECT,
                fallback_text=(
                    "그 질문은 여기서 자세히 답하기보다, 마음이나 안전과 관련된 "
                    "이야기로 다시 가져오는 게 좋아요. 지금 그 질문 때문에 답답함, "
                    "걱정, 불안 같은 감정이 생겼다면 그 부분부터 같이 정리해볼게요."
                ),
                blocked_terms=_COMMON_BLOCKED_TERMS,
                allow_llm_rewrite=False,
            )

        if assessment.safety_level == ChildSupportSafetyLevel.URGENT:
            if assessment.intent in {
                ChildSupportSafetyIntent.OTHER_HARM_IDEATION,
                ChildSupportSafetyIntent.OTHER_HARM_METHOD_REQUEST,
            }:
                if (
                    assessment.intent
                    == ChildSupportSafetyIntent.OTHER_HARM_METHOD_REQUEST
                ):
                    fallback_text = (
                        "그 친구를 해치는 방법은 알려줄 수 없어요. 그 행동은 하면 "
                        "안 되고, 지금은 그 친구에게 가까이 가지 않는 게 먼저예요. "
                        "손에 위험한 물건이 있으면 멀리 두고, 집에 있는 어른이나 "
                        "믿을 수 있는 어른에게 '누군가를 해치고 싶은 생각이 너무 "
                        "커져서 지금 혼자 있으면 위험해요'라고 바로 보여주세요."
                    )
                else:
                    fallback_text = (
                        "그만큼 화가 크다는 뜻일 수 있지만, 그 친구를 해치거나 "
                        "찾아가는 행동은 하면 안 돼요. 지금은 그 친구와 떨어진 "
                        "상태를 유지하고, 손에 위험한 물건이 있으면 멀리 두세요. "
                        "가까운 어른에게 '너무 화가 커서 누군가를 해칠까 봐 "
                        "걱정돼요'라고 보여주고, 지금 혼자 있지 않는 게 필요해요."
                    )
                return ChildSupportResponseStrategy(
                    name=ChildSupportResponseStrategyName.HARM_TO_OTHERS_SAFETY,
                    fallback_text=fallback_text,
                    required_any_term_groups=(
                        ("해치는 방법은 알려줄 수 없어요", "해치거나"),
                        ("하면 안", "가까이 가지 않는", "떨어진"),
                        ("어른", "혼자 있지"),
                    ),
                    blocked_terms=(
                        *_COMMON_BLOCKED_TERMS,
                        "칼로",
                        "목을",
                        "찌르",
                        "때리면",
                    ),
                    allow_llm_rewrite=False,
                )
            if assessment.immediate_danger:
                return ChildSupportResponseStrategy(
                    name=ChildSupportResponseStrategyName.URGENT_SAFETY,
                    fallback_text=(
                        "지금은 혼자 버티면서 대화만 이어가기보다 안전을 먼저 "
                        "확인해야 해요. 지금 안전한 곳에 있는지 보고, 바로 가까운 "
                        "보호자나 믿을 수 있는 어른에게 이 문장을 보여주세요: "
                        "'지금 혼자 있기 위험해서 도움이 필요해요.'"
                    ),
                    required_terms=("안전", "어른"),
                    blocked_terms=_COMMON_BLOCKED_TERMS,
                    allow_llm_rewrite=False,
                )
            return ChildSupportResponseStrategy(
                name=ChildSupportResponseStrategyName.URGENT_SAFETY,
                fallback_text=(
                    "그 말을 꺼내준 건 정말 중요한 신호예요. 먼저 지금 안전한 "
                    "곳에 있는지 확인하고 싶어요. 혼자 있다면 사람이 있는 곳으로 "
                    "이동하고, 믿을 수 있는 어른에게 '지금 너무 위험한 생각이 들어서 "
                    "같이 있어줬으면 좋겠어요'라고 보여줄 문장을 같이 만들 수 있어요."
                ),
                required_terms=("안전", "어른"),
                blocked_terms=_COMMON_BLOCKED_TERMS,
                allow_llm_rewrite=False,
            )

        if assessment.safety_level == ChildSupportSafetyLevel.PARENT_HANDOFF:
            return ChildSupportResponseStrategy(
                name=ChildSupportResponseStrategyName.SAFETY_CHECK,
                fallback_text=(
                    "친구한테 맞았다는 말을 해줘서 고마워요. 그건 그냥 넘길 "
                    "일이 아니라, 먼저 안전과 몸 상태를 확인해야 하는 일이에요. "
                    "지금은 그 친구와 떨어진 안전한 곳에 있나요? 다친 곳이나 아픈 "
                    "곳이 있으면 어디가 아픈지 하나만 말해줄 수 있을까요?"
                ),
                required_any_term_groups=(
                    ("안전", "안전한 곳"),
                    ("다친 곳", "아픈 곳", "몸 상태"),
                    ("말해줄", "알려줄", "있나요"),
                ),
                blocked_terms=(
                    *_COMMON_BLOCKED_TERMS,
                    "음악",
                    "좋아하는 활동",
                    "혼자 조용히",
                    "편안한 시간",
                    "언제든 말해줘",
                    "행복하길",
                ),
                allow_llm_rewrite=True,
            )

        if assessment.intent == ChildSupportSafetyIntent.POST_URGENT_DEESCALATION:
            return ChildSupportResponseStrategy(
                name=ChildSupportResponseStrategyName.POST_URGENT_DEESCALATION,
                fallback_text=(
                    "응, 정말 많이 힘들었겠다. 방금까지 화가 너무 컸고, 이제 "
                    "그 뒤에 속상함까지 밀려오는 것 같아요. 그런 생각이 스친다고 "
                    "네가 나쁜 아이인 건 아니에요. 다만 그 친구를 해치는 쪽으로는 "
                    "가지 않게, 지금은 여기서 같이 멈춰볼게요. 말이 잘 안 나와도 "
                    "괜찮아요. 지금 제일 버거운 말 하나만 천천히 이어줘도 돼요."
                ),
                required_any_term_groups=(
                    ("많이 힘들", "힘들었겠다", "버거운"),
                    ("나쁜 아이", "그런 생각", "속상함"),
                    ("해치는 쪽", "멈춰"),
                    ("말이 잘 안 나와도", "이어줘도 돼요"),
                ),
                blocked_terms=(
                    *_COMMON_BLOCKED_TERMS,
                    "골라볼까요",
                    "어디에 가까운지",
                    "가고 싶은 마음",
                    "상태를 유지",
                    "위험한 물건",
                    "말부터 받아",
                ),
                allow_llm_rewrite=False,
            )

        if assessment.intent == ChildSupportSafetyIntent.PEER_RESPONSE_PLANNING:
            return ChildSupportResponseStrategy(
                name=ChildSupportResponseStrategyName.PEER_RESPONSE_PLANNING,
                fallback_text=(
                    "복수하고 싶을 만큼 억울하고 화가 났구나. 그런 마음이 올라오는 "
                    "건 이상한 일이 아니지만, 혼자 찾아가거나 되갚는 행동은 너를 "
                    "더 위험하게 만들 수 있어요. 지금은 바로 행동하기보다 네 마음과 "
                    "안전을 먼저 붙잡을게요. 상대에게 할 말은 '때린 행동은 싫었고, "
                    "다시는 그러지 않았으면 해요'처럼 천천히 정리할 수 있어요. "
                    "당장은 그 친구에게 바로 연락하거나 찾아가지 않는 쪽으로 같이 "
                    "멈춰볼게요."
                ),
                required_any_term_groups=(
                    ("복수", "억울", "화"),
                    ("혼자", "위험", "안전"),
                    ("할 말", "정리"),
                    ("찾아가지", "멈춰"),
                ),
                blocked_terms=(
                    *_COMMON_BLOCKED_TERMS,
                    "골라볼까요",
                    "때려",
                    "복수해",
                    "혼내",
                    "아픈 곳",
                    "다친 곳",
                    "몸의 어디",
                    "몸 어디",
                ),
                allow_llm_rewrite=True,
            )

        if (
            assessment.intent
            == ChildSupportSafetyIntent.POST_HANDOFF_EMOTIONAL_FOLLOWUP
        ):
            return ChildSupportResponseStrategy(
                name=ChildSupportResponseStrategyName.POST_INCIDENT_EMOTIONAL_FOLLOWUP,
                fallback_text=(
                    "그 친구와 떨어져서 집에 왔다면 정말 많이 흔들렸겠어요. 맞은 "
                    "일은 그냥 넘길 일이 아니고, 마음이 이렇게 흔들리는 것도 "
                    "이상한 일이 아니에요. 지금 떠오르는 장면이나 말 중 제일 "
                    "마음에 걸리는 것부터 한 문장으로 이어 말해줘도 괜찮아요. "
                    "여기서는 복수보다 네 마음이 더 망가지지 않게 같이 붙잡을게요."
                ),
                required_any_term_groups=(
                    ("흔들", "마음"),
                    ("한 문장", "이어"),
                    ("괜찮아요", "붙잡"),
                    ("안전한 곳", "집", "떨어져"),
                ),
                blocked_terms=(
                    *_COMMON_BLOCKED_TERMS,
                    "아픈 곳",
                    "다친 곳",
                    "몸 상태",
                    "몸의 어디",
                    "몸 어디",
                    "몸의 어떤 부분",
                    "골라볼까요",
                    "어디에 가까운지",
                ),
                allow_llm_rewrite=True,
            )

        if assessment.safety_level == ChildSupportSafetyLevel.CHECK_IN:
            return ChildSupportResponseStrategy(
                name=ChildSupportResponseStrategyName.CHECK_IN,
                fallback_text=(
                    "지금 정말 많이 버거워 보이네요. 말이 잘 안 나와도 괜찮아요. "
                    "이 힘듦이 오늘 갑자기 커진 건지, 아니면 오래 쌓여 있다가 "
                    "터진 건지만 천천히 이어 말해줘도 돼요. 짧은 한 문장이어도 "
                    "괜찮아요."
                ),
                required_any_term_groups=(
                    ("버거", "힘듦", "마음"),
                    ("말이 잘 안 나와도", "한 문장", "이어"),
                    ("갑자기", "쌓여", "천천히"),
                ),
                blocked_terms=(
                    *_COMMON_BLOCKED_TERMS,
                    "골라볼까요",
                    "어디에 가까운지",
                    "가족",
                    "부모",
                    "어른",
                    "선생님",
                    "상담사",
                    "몸의 어디",
                    "몸 어디",
                ),
                allow_llm_rewrite=True,
            )

        trimmed = " ".join(message.split())
        return ChildSupportResponseStrategy(
            name=ChildSupportResponseStrategyName.SUPPORTIVE_REFLECTION,
            fallback_text=(
                "말해줘서 고마워요. 지금은 정답을 바로 찾기보다, 방금 말한 "
                f"'{trimmed[:42]}'에서 어떤 감정이 가장 컸는지부터 천천히 "
                "나눠볼게요."
            ),
            required_any_term_groups=(
                ("마음", "감정", "느낌"),
                ("천천히", "나눠", "말해"),
            ),
            blocked_terms=(
                *_COMMON_BLOCKED_TERMS,
                "가족",
                "부모",
                "어른",
                "선생님",
                "상담사",
            ),
            allow_llm_rewrite=True,
        )


__all__ = [
    "ChildSupportResponsePolicy",
    "ChildSupportResponseStrategy",
    "ChildSupportResponseStrategyName",
]
