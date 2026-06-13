"""아이용 지원 대화 응답 전략 policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from agent.src.contracts.child_support_contracts import (
    ChildSupportSafetyLevel,
    ChildSupportScopeStatus,
)
from agent.src.features.wellbeing.child_support.safety_intent import (
    ChildSupportSafetyIntent,
)
from agent.src.features.wellbeing.child_support.safety_policy import (
    ChildSupportSafetyAssessment,
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


class ChildSupportResponseMove(StrEnum):
    """응답이 수행해야 하는 상담 동작 단위."""

    ACKNOWLEDGE_DISCLOSURE = "acknowledge_disclosure"
    VALIDATE_FEELING = "validate_feeling"
    NORMALIZE_REACTION = "normalize_reaction"
    SEPARATE_THOUGHT_FROM_IDENTITY = "separate_thought_from_identity"
    SET_HARM_BOUNDARY = "set_harm_boundary"
    REFUSE_HARM_METHOD = "refuse_harm_method"
    KEEP_PHYSICAL_DISTANCE = "keep_physical_distance"
    ADULT_HANDOFF = "adult_handoff"
    CHECK_IMMEDIATE_SAFETY = "check_immediate_safety"
    CHECK_INJURY = "check_injury"
    INVITE_OPEN_NARRATIVE = "invite_open_narrative"
    PLAN_SAFE_BOUNDARY = "plan_safe_boundary"
    REDIRECT_SCOPE = "redirect_scope"
    HOLD_WITH_CHILD = "hold_with_child"


class ChildSupportResponseTone(StrEnum):
    """응답 생성에 사용할 말투 policy."""

    WARM_CHILD_SUPPORT = "warm_child_support"
    CALM_SAFETY_BOUNDARY = "calm_safety_boundary"


@dataclass(frozen=True, slots=True)
class ChildSupportResponsePlan:
    """LLM에 넘기거나 fallback으로 쓸 응답 plan."""

    name: ChildSupportResponseStrategyName
    moves: tuple[ChildSupportResponseMove, ...]
    fallback_text: str
    tone: ChildSupportResponseTone = ChildSupportResponseTone.WARM_CHILD_SUPPORT
    required_terms: tuple[str, ...] = ()
    required_any_term_groups: tuple[tuple[str, ...], ...] = ()
    blocked_terms: tuple[str, ...] = ()
    allow_llm_rewrite: bool = True

    def accepts(self, reply: str) -> bool:
        """LLM 응답이 이 plan을 지키는지 확인한다."""

        if not reply.strip():
            return False
        if any(term in reply for term in self.blocked_terms):
            return False
        if not all(term in reply for term in self.required_terms):
            return False
        if not all(
            any(term in reply for term in term_group)
            for term_group in self.required_any_term_groups
        ):
            return False
        return all(_move_is_satisfied(reply, move) for move in self.moves)

    @property
    def forbidden_terms(self) -> tuple[str, ...]:
        """기존 호출부 호환용 alias.

        새 호출부는 canonical 필드인 `blocked_terms`를 사용한다. 유지 이유와
        제거 조건은 `docs/contracts/legacy_contract_ledger.md`에 기록한다.
        """

        return self.blocked_terms


# 기존 외부 호출부가 이전 이름을 import해도 같은 plan 타입을 보게 둔다.
ChildSupportResponseStrategy = ChildSupportResponsePlan


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

_MOVE_REQUIRED_ANY_TERM_GROUPS: dict[
    ChildSupportResponseMove,
    tuple[tuple[str, ...], ...],
] = {
    ChildSupportResponseMove.ACKNOWLEDGE_DISCLOSURE: (
        ("말해줘서", "꺼내줘서", "꺼내준", "이야기해줘서", "힘들었겠다"),
    ),
    ChildSupportResponseMove.VALIDATE_FEELING: (
        ("힘들", "화", "억울", "속상", "무서", "버거", "흔들"),
    ),
    ChildSupportResponseMove.NORMALIZE_REACTION: (
        ("이상한 일이 아니", "그럴 수", "나쁜 아이", "나쁜 마음"),
    ),
    ChildSupportResponseMove.SEPARATE_THOUGHT_FROM_IDENTITY: (
        ("나쁜 아이", "나쁜 마음", "그런 생각이 스친다고"),
    ),
    ChildSupportResponseMove.SET_HARM_BOUNDARY: (
        ("해치는 쪽", "해치거나", "하면 안", "멈춰"),
    ),
    ChildSupportResponseMove.REFUSE_HARM_METHOD: (
        ("방법은 알려줄 수 없", "알려줄 수 없"),
    ),
    ChildSupportResponseMove.KEEP_PHYSICAL_DISTANCE: (
        ("가까이 가지", "떨어진", "찾아가지", "거리"),
    ),
    ChildSupportResponseMove.ADULT_HANDOFF: (("어른", "보호자", "혼자 있지"),),
    ChildSupportResponseMove.CHECK_IMMEDIATE_SAFETY: (("안전한 곳", "안전"),),
    ChildSupportResponseMove.CHECK_INJURY: (("다친 곳", "아픈 곳", "몸 상태"),),
    ChildSupportResponseMove.INVITE_OPEN_NARRATIVE: (
        ("한 문장", "이어", "말이 잘 안 나와도", "천천히"),
    ),
    ChildSupportResponseMove.PLAN_SAFE_BOUNDARY: (
        ("할 말", "정리", "찾아가지", "멈춰", "연락하거나"),
    ),
    ChildSupportResponseMove.REDIRECT_SCOPE: (("마음", "안전", "다시 가져오는"),),
    ChildSupportResponseMove.HOLD_WITH_CHILD: (
        ("같이", "붙잡", "멈춰볼게요", "이어줘도 돼요"),
    ),
}


def _move_is_satisfied(reply: str, move: ChildSupportResponseMove) -> bool:
    term_groups = _MOVE_REQUIRED_ANY_TERM_GROUPS.get(move, ())
    return all(any(term in reply for term in term_group) for term_group in term_groups)


class ChildSupportResponsePolicy:
    """safety assessment를 실제 응답 전략으로 바꾼다."""

    def build_strategy(
        self,
        *,
        message: str,
        assessment: ChildSupportSafetyAssessment,
    ) -> ChildSupportResponsePlan:
        """기존 호출부 호환용 method alias."""

        return self.build_plan(message=message, assessment=assessment)

    def build_plan(
        self,
        *,
        message: str,
        assessment: ChildSupportSafetyAssessment,
    ) -> ChildSupportResponsePlan:
        """현재 safety 단계에 맞는 응답 plan을 만든다."""

        if assessment.scope_status == ChildSupportScopeStatus.REDIRECTED:
            return _SCOPE_REDIRECT_PLAN

        if assessment.safety_level == ChildSupportSafetyLevel.URGENT:
            if assessment.intent == ChildSupportSafetyIntent.OTHER_HARM_METHOD_REQUEST:
                return _OTHER_HARM_METHOD_REQUEST_PLAN
            if assessment.intent == ChildSupportSafetyIntent.OTHER_HARM_IDEATION:
                return _OTHER_HARM_IDEATION_PLAN
            if assessment.immediate_danger:
                return _URGENT_IMMEDIATE_DANGER_PLAN
            return _URGENT_SAFETY_PLAN

        if assessment.safety_level == ChildSupportSafetyLevel.PARENT_HANDOFF:
            return _PARENT_HANDOFF_PLAN

        if assessment.intent in _INTENT_PLAN_TABLE:
            return _INTENT_PLAN_TABLE[assessment.intent]

        if assessment.safety_level == ChildSupportSafetyLevel.CHECK_IN:
            return _CHECK_IN_PLAN

        return _build_supportive_reflection_plan(message)


_HARM_TO_OTHERS_REQUIRED_TERM_GROUPS = (
    ("해치는 방법은 알려줄 수 없어요", "해치거나"),
    ("하면 안", "가까이 가지 않는", "떨어진"),
    ("어른", "혼자 있지"),
)
_HARM_TO_OTHERS_BLOCKED_TERMS = (
    *_COMMON_BLOCKED_TERMS,
    "칼로",
    "목을",
    "찌르",
    "때리면",
)


def _build_harm_to_others_plan(
    *,
    moves: tuple[ChildSupportResponseMove, ...],
    fallback_text: str,
) -> ChildSupportResponsePlan:
    return ChildSupportResponsePlan(
        name=ChildSupportResponseStrategyName.HARM_TO_OTHERS_SAFETY,
        moves=moves,
        fallback_text=fallback_text,
        tone=ChildSupportResponseTone.CALM_SAFETY_BOUNDARY,
        required_any_term_groups=_HARM_TO_OTHERS_REQUIRED_TERM_GROUPS,
        blocked_terms=_HARM_TO_OTHERS_BLOCKED_TERMS,
        allow_llm_rewrite=False,
    )


_SCOPE_REDIRECT_PLAN = ChildSupportResponsePlan(
    name=ChildSupportResponseStrategyName.SCOPE_REDIRECT,
    moves=(ChildSupportResponseMove.REDIRECT_SCOPE,),
    fallback_text=(
        "그 질문은 여기서 자세히 답하기보다, 마음이나 안전과 관련된 "
        "이야기로 다시 가져오는 게 좋아요. 지금 그 질문 때문에 답답함, "
        "걱정, 불안 같은 감정이 생겼다면 그 부분부터 같이 정리해볼게요."
    ),
    blocked_terms=_COMMON_BLOCKED_TERMS,
    allow_llm_rewrite=False,
)

_OTHER_HARM_METHOD_REQUEST_PLAN = _build_harm_to_others_plan(
    moves=(
        ChildSupportResponseMove.REFUSE_HARM_METHOD,
        ChildSupportResponseMove.SET_HARM_BOUNDARY,
        ChildSupportResponseMove.KEEP_PHYSICAL_DISTANCE,
        ChildSupportResponseMove.ADULT_HANDOFF,
    ),
    fallback_text=(
        "그 친구를 해치는 방법은 알려줄 수 없어요. 그 행동은 하면 "
        "안 되고, 지금은 그 친구에게 가까이 가지 않는 게 먼저예요. "
        "손에 위험한 물건이 있으면 멀리 두고, 집에 있는 어른이나 "
        "믿을 수 있는 어른에게 '누군가를 해치고 싶은 생각이 너무 "
        "커져서 지금 혼자 있으면 위험해요'라고 바로 보여주세요."
    ),
)

_OTHER_HARM_IDEATION_PLAN = _build_harm_to_others_plan(
    moves=(
        ChildSupportResponseMove.VALIDATE_FEELING,
        ChildSupportResponseMove.SET_HARM_BOUNDARY,
        ChildSupportResponseMove.KEEP_PHYSICAL_DISTANCE,
        ChildSupportResponseMove.ADULT_HANDOFF,
    ),
    fallback_text=(
        "그만큼 화가 크다는 뜻일 수 있지만, 그 친구를 해치거나 "
        "찾아가는 행동은 하면 안 돼요. 지금은 그 친구와 떨어진 "
        "상태를 유지하고, 손에 위험한 물건이 있으면 멀리 두세요. "
        "가까운 어른에게 '너무 화가 커서 누군가를 해칠까 봐 "
        "걱정돼요'라고 보여주고, 지금 혼자 있지 않는 게 필요해요."
    ),
)

_URGENT_IMMEDIATE_DANGER_PLAN = ChildSupportResponsePlan(
    name=ChildSupportResponseStrategyName.URGENT_SAFETY,
    moves=(
        ChildSupportResponseMove.CHECK_IMMEDIATE_SAFETY,
        ChildSupportResponseMove.ADULT_HANDOFF,
    ),
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

_URGENT_SAFETY_PLAN = ChildSupportResponsePlan(
    name=ChildSupportResponseStrategyName.URGENT_SAFETY,
    moves=(
        ChildSupportResponseMove.ACKNOWLEDGE_DISCLOSURE,
        ChildSupportResponseMove.CHECK_IMMEDIATE_SAFETY,
        ChildSupportResponseMove.ADULT_HANDOFF,
        ChildSupportResponseMove.HOLD_WITH_CHILD,
    ),
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

_PARENT_HANDOFF_PLAN = ChildSupportResponsePlan(
    name=ChildSupportResponseStrategyName.SAFETY_CHECK,
    moves=(
        ChildSupportResponseMove.ACKNOWLEDGE_DISCLOSURE,
        ChildSupportResponseMove.CHECK_IMMEDIATE_SAFETY,
        ChildSupportResponseMove.CHECK_INJURY,
    ),
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

_POST_URGENT_DEESCALATION_PLAN = ChildSupportResponsePlan(
    name=ChildSupportResponseStrategyName.POST_URGENT_DEESCALATION,
    moves=(
        ChildSupportResponseMove.VALIDATE_FEELING,
        ChildSupportResponseMove.NORMALIZE_REACTION,
        ChildSupportResponseMove.SEPARATE_THOUGHT_FROM_IDENTITY,
        ChildSupportResponseMove.SET_HARM_BOUNDARY,
        ChildSupportResponseMove.HOLD_WITH_CHILD,
        ChildSupportResponseMove.INVITE_OPEN_NARRATIVE,
    ),
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

_PEER_RESPONSE_PLANNING_PLAN = ChildSupportResponsePlan(
    name=ChildSupportResponseStrategyName.PEER_RESPONSE_PLANNING,
    moves=(
        ChildSupportResponseMove.VALIDATE_FEELING,
        ChildSupportResponseMove.NORMALIZE_REACTION,
        ChildSupportResponseMove.HOLD_WITH_CHILD,
        ChildSupportResponseMove.PLAN_SAFE_BOUNDARY,
        ChildSupportResponseMove.KEEP_PHYSICAL_DISTANCE,
    ),
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

_POST_INCIDENT_EMOTIONAL_FOLLOWUP_PLAN = ChildSupportResponsePlan(
    name=ChildSupportResponseStrategyName.POST_INCIDENT_EMOTIONAL_FOLLOWUP,
    moves=(
        ChildSupportResponseMove.VALIDATE_FEELING,
        ChildSupportResponseMove.NORMALIZE_REACTION,
        ChildSupportResponseMove.INVITE_OPEN_NARRATIVE,
        ChildSupportResponseMove.HOLD_WITH_CHILD,
    ),
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

_CHECK_IN_PLAN = ChildSupportResponsePlan(
    name=ChildSupportResponseStrategyName.CHECK_IN,
    moves=(
        ChildSupportResponseMove.VALIDATE_FEELING,
        ChildSupportResponseMove.INVITE_OPEN_NARRATIVE,
    ),
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

_INTENT_PLAN_TABLE = {
    ChildSupportSafetyIntent.POST_URGENT_DEESCALATION: _POST_URGENT_DEESCALATION_PLAN,
    ChildSupportSafetyIntent.PEER_RESPONSE_PLANNING: _PEER_RESPONSE_PLANNING_PLAN,
    ChildSupportSafetyIntent.POST_HANDOFF_EMOTIONAL_FOLLOWUP: (
        _POST_INCIDENT_EMOTIONAL_FOLLOWUP_PLAN
    ),
}


def _build_supportive_reflection_plan(message: str) -> ChildSupportResponsePlan:
    trimmed = " ".join(message.split())
    return ChildSupportResponsePlan(
        name=ChildSupportResponseStrategyName.SUPPORTIVE_REFLECTION,
        moves=(
            ChildSupportResponseMove.ACKNOWLEDGE_DISCLOSURE,
            ChildSupportResponseMove.INVITE_OPEN_NARRATIVE,
        ),
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
