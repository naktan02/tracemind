"""아이용 지원 대화 응답 전략 policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

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
    SAFETY_CHECK = "safety_check"
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
                allow_llm_rewrite=False,
            )

        if assessment.safety_level == ChildSupportSafetyLevel.URGENT:
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
                    allow_llm_rewrite=False,
                )
            return ChildSupportResponseStrategy(
                name=ChildSupportResponseStrategyName.URGENT_SAFETY,
                fallback_text=(
                    "그 말을 꺼내준 건 정말 중요한 신호예요. 먼저 지금 안전한 "
                    "곳에 있는지 확인하고 싶어요. 혼자 있다면 사람이 있는 곳으로 "
                    "이동하고, 믿을 수 있는 어른에게 '지금 너무 위험한 생각이 들어서 "
                    "같이 있어줬으면 좋겠어'라고 보여줄 문장을 같이 만들 수 있어요."
                ),
                required_terms=("안전", "어른"),
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
                    "음악",
                    "좋아하는 활동",
                    "혼자 조용히",
                    "편안한 시간",
                    "언제든 말해줘",
                    "행복하길",
                ),
                allow_llm_rewrite=True,
            )

        if assessment.safety_level == ChildSupportSafetyLevel.CHECK_IN:
            return ChildSupportResponseStrategy(
                name=ChildSupportResponseStrategyName.CHECK_IN,
                fallback_text=(
                    "지금 마음이 많이 무거운 상태로 느껴져요. 바로 해결책을 찾기보다, "
                    "가장 힘든 느낌이 몸의 어디에서 크게 느껴지는지 하나만 "
                    "골라볼까요? 그 다음에는 숨을 천천히 한 번 같이 쉬어볼게요."
                ),
                required_any_term_groups=(
                    ("마음", "감정", "느낌", "느껴"),
                    ("골라", "말해", "알려"),
                    ("숨", "호흡", "천천히"),
                ),
                blocked_terms=("가족", "부모", "어른", "선생님", "상담사"),
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
            blocked_terms=("가족", "부모", "어른", "선생님", "상담사"),
            allow_llm_rewrite=True,
        )


__all__ = [
    "ChildSupportResponsePolicy",
    "ChildSupportResponseStrategy",
    "ChildSupportResponseStrategyName",
]
