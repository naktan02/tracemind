"""아이용 지원 대화 LLM prompt 생성."""

from __future__ import annotations

from typing import Protocol

from agent.src.contracts.child_support_contracts import ChildSupportSafetyLevel
from agent.src.features.wellbeing.child_support.context_provider import (
    ChildSupportConversationContext,
)
from agent.src.features.wellbeing.child_support.llm_provider import (
    ChildSupportLlmMessage,
)


class ChildSupportSafetyHint(Protocol):
    safety_level: ChildSupportSafetyLevel
    reason: str


def build_child_support_conversation_prompt(
    *,
    message: str,
    context: ChildSupportConversationContext,
    assessment: ChildSupportSafetyHint,
) -> str:
    """아이 메시지에 바로 답하기 위한 LLM prompt를 만든다."""

    return _render_messages(
        build_child_support_conversation_messages(
            message=message,
            context=context,
            assessment=assessment,
        )
    )


def build_child_support_conversation_messages(
    *,
    message: str,
    context: ChildSupportConversationContext,
    assessment: ChildSupportSafetyHint,
) -> tuple[ChildSupportLlmMessage, ...]:
    """아이 메시지 응답용 구조화 LLM message를 만든다."""

    return (
        ChildSupportLlmMessage(
            role="system",
            content=_build_conversation_system_prompt(),
        ),
        ChildSupportLlmMessage(
            role="user",
            content=_build_conversation_context_block(
                message=message,
                context=context,
                assessment=assessment,
            ),
        ),
        *_build_recent_conversation_messages(context),
        ChildSupportLlmMessage(
            role="user",
            content=(
                f"아이의 새 메시지: {message}\n\n"
                "아이에게 바로 보여줄 답변만 작성해라. 첫 문장은 아이가 방금 "
                "묻거나 요청한 내용에 대한 직접 답이어야 한다."
            ),
        ),
    )


def _build_conversation_system_prompt() -> str:
    return (
        "너는 TraceMind의 아이용 마음 도움 로컬 AI 코치다.\n"
        "아래 로컬 context를 바탕으로 아이의 새 메시지에 자연스럽게 답한다.\n"
        "지원 범위:\n"
        "- 이 AI는 마음 도움, 감정, 관계, 안전, 현재 상황을 다루는 대화만 한다.\n"
        "- 공부, 코딩, 일반 지식, 취미, 검색 대신 풀어주기처럼 마음 도움과 직접 "
        "관련 없는 질문은 지원 범위 밖이다.\n"
        "- 지원 범위 밖 질문에는 절대 내용을 풀이하거나 정답을 주지 않는다.\n"
        "- 지원 범위 밖 질문이면 그 질문을 하게 된 마음, 지금 기분, 현재 상황으로 "
        "부드럽게 되돌린다.\n"
        "원칙:\n"
        "- 한국어 해요체로 3~6문장만 답한다.\n"
        "- 한 답변 안에서 반말과 존댓말을 섞지 않는다.\n"
        "- 아이를 '당신'이라고 부르지 않는다. 2인칭 호칭은 되도록 생략하고, "
        "필요할 때만 자연스럽게 '네' 정도로 쓴다.\n"
        "- 첫 문장은 아이의 마지막 말이 묻거나 요청한 내용에 대한 직접 답이어야 "
        "한다.\n"
        "- 첫 문장을 안부 인사, 하루 안부, 잘 지내길 바란다는 말, 일반적인 "
        "상담 시작 문장으로 쓰지 않는다.\n"
        "- 로컬 context의 캡처/검색 원문은 아이 상태를 이해하기 위해 참고한다.\n"
        "- 답변에서 원문을 언급해도 되지만, 아이가 감시당한다고 느끼지 않게 "
        "필요한 부분만 짧게 연결한다.\n"
        "- 아이가 설명이나 판단을 요청하면 곧바로 짧게 답한 뒤, 필요한 경우에만 "
        "이어 말할 질문을 덧붙인다.\n"
        "- 아이가 자신의 상태나 상황을 묻는다면 질문으로 되돌리지 말고, 최근 "
        "대화와 로컬 context를 바탕으로 질문에 맞게 먼저 답한다.\n"
        "- 아이가 최근에 무슨 일을 당했는지, 겪었는지, 알고 있는지 묻는다면 "
        "로컬 context에서 확인되는 범위를 먼저 말하고, 확인되지 않은 부분은 "
        "모른다고 분리해서 말한다.\n"
        "- 이번 턴의 response_policy가 질문을 금지하면 마지막 문장을 질문으로 "
        "끝내지 않는다.\n"
        "- 답변은 근거 목록이나 요약문 형식이 아니라 자연스러운 대화여야 한다.\n"
        "- 필요할 때만 어떤 흐름을 보고 그렇게 이해했는지 짧게 덧붙인다.\n"
        "- 로컬 context에서 확인되는 최근 흐름과 대화 내용을 바탕으로 답하되, "
        "확인되지 않은 정보는 안다고 말하지 않는다.\n"
        "- 이미 아이가 말한 내용과 같은 질문을 반복하지 않는다.\n"
        "- 원인을 설명할 때는 가능성으로 말하고 단정하지 않는다.\n"
        "- 출력은 순수 답변 문장만 사용하고 장식 문자는 쓰지 않는다.\n"
        "- 진단, 치료, 법률 조언처럼 말하지 않는다.\n"
        "- 자해/타해 방법, 은폐 방법, 구체적 실행 방법은 절대 제공하지 않는다.\n"
        "- 즉시 위험이 보이면 지금 안전한지 먼저 확인한다.\n"
        "- 기본적으로는 아이가 이어 말할 수 있게 열어 두되, 모든 답변을 질문으로 "
        "끝낼 필요는 없다."
    )


def _build_conversation_context_block(
    *,
    message: str,
    context: ChildSupportConversationContext,
    assessment: ChildSupportSafetyHint,
) -> str:
    message_lines = [
        f"- {record.role}: {record.text}" for record in context.recent_messages[-6:]
    ]
    history_block = (
        "현재 대화 최근 히스토리:\n" + "\n".join(message_lines)
        if message_lines
        else "현재 대화 최근 히스토리: 없음"
    )
    return (
        "로컬 context:\n"
        f"{_build_message_intent_block(message, assessment)}\n\n"
        f"{_build_counselor_context_block(context)}\n\n"
        f"safety_level_hint: {assessment.safety_level.value}\n"
        f"safety_reason_hint: {assessment.reason}\n\n"
        f"{_build_summary_block(context)}\n\n"
        f"{_build_context_notes_block(context.wellbeing_context_notes)}\n\n"
        f"{history_block}"
    )


def _build_recent_conversation_messages(
    context: ChildSupportConversationContext,
) -> tuple[ChildSupportLlmMessage, ...]:
    messages: list[ChildSupportLlmMessage] = []
    for record in context.recent_messages[-6:]:
        if record.role == "assistant":
            messages.append(
                ChildSupportLlmMessage(role="assistant", content=record.text)
            )
        elif record.role == "child":
            messages.append(ChildSupportLlmMessage(role="user", content=record.text))
    return tuple(messages)


def build_child_support_conversation_repair_prompt(
    *,
    original_prompt: str,
    previous_reply: str,
) -> str:
    """질문 회피형 응답을 같은 LLM에 다시 쓰게 하는 prompt를 만든다."""

    return (
        f"{original_prompt}\n\n"
        "이전 답변은 아이의 질문에 직접 답하지 않고 다시 질문으로 돌려서 부적절했다.\n"
        f"이전 답변: {previous_reply}\n\n"
        "다시 작성 규칙:\n"
        "- 아이가 물은 내용에 첫 문장부터 직접 답한다.\n"
        "- 아이가 이미 말한 내용을 다시 자세히 말해 달라고 요구하지 않는다.\n"
        "- 현재 상태나 감정을 묻는 말에는 최근 대화와 로컬 context를 바탕으로 "
        "가능한 이해를 먼저 말한다.\n"
        "- response_policy가 질문을 금지하면 마지막 문장을 질문으로 끝내지 않는다.\n"
        "- 한 답변 안에서 말투를 바꾸지 않는다.\n"
        "- 순수 답변만 작성한다."
    )


def build_child_support_conversation_style_repair_prompt(
    *,
    original_prompt: str,
    previous_reply: str,
) -> str:
    """상담 대화 말투가 어색한 응답을 같은 LLM에 다시 쓰게 하는 prompt."""

    return (
        f"{original_prompt}\n\n"
        "이전 답변은 내용 방향은 사용할 수 있지만 말투가 어색했다.\n"
        f"이전 답변: {previous_reply}\n\n"
        "다시 작성 규칙:\n"
        "- 내용의 핵심 판단과 안전 방향은 유지한다.\n"
        "- 아이를 '당신'이라고 부르지 않는다.\n"
        "- '말씀', '도와드릴게요', '도와드릴 수 있어요' 같은 기관 상담 말투를 "
        "쓰지 않는다.\n"
        "- 한 답변 안에서 반말과 존댓말을 섞지 않는다.\n"
        "- 모든 문장 끝은 '~요', '~예요', '~이에요', '~습니다', '~까요' 같은 "
        "해요체나 부드러운 존댓말로 통일한다.\n"
        "- '~야', '~같아', '~봐', '~줘', '~어떨까'처럼 반말로 끝나는 문장은 "
        "하나도 쓰지 않는다.\n"
        "- 짧고 자연스러운 해요체로 쓴다.\n"
        "- 순수 답변만 작성한다."
    )


def is_child_support_self_state_question(message: str) -> bool:
    """아이가 자기 상태/감정/상황에 대한 판단을 요청하는지 확인한다."""

    normalized = " ".join(message.lower().split())
    compact = "".join(normalized.split())
    if not normalized:
        return False
    self_markers = ("내", "나", "저", "제")
    state_markers = ("기분", "감정", "상태", "상황", "마음", "힘든지", "우울")
    ask_markers = (
        "어떤",
        "어떨",
        "어때",
        "같",
        "보여",
        "생각",
        "알아",
        "모르",
        "말해",
        "알려",
        "판단",
    )
    if (
        any(marker in compact for marker in self_markers)
        and any(marker in compact for marker in state_markers)
        and any(marker in compact for marker in ask_markers)
    ):
        return True
    recent_incident_markers = (
        "최근에무슨일",
        "최근무슨일",
        "무슨일을당",
        "무슨일당",
        "무슨일이있",
        "어떤일을당",
        "어떤일당",
        "어떤일이있",
        "일을당했",
        "당했는지",
        "겪었는지",
        "있었는지",
    )
    incident_ask_markers = (
        "알고",
        "알아",
        "기억",
        "봤",
        "보여",
        "말해",
        "알려",
        "생각",
    )
    if (
        any(marker in compact for marker in self_markers)
        and any(marker in compact for marker in recent_incident_markers)
        and any(marker in compact for marker in incident_ask_markers)
    ):
        return True
    if any(marker in compact for marker in self_markers) and any(
        phrase in compact for phrase in ("어떤것같", "어떤거같", "어떻게보")
    ):
        return True
    if "질문하지말고" in compact and (
        any(marker in compact for marker in state_markers) or "나에대해" in compact
    ):
        return True
    return "왜힘든지" in compact and any(
        marker in compact for marker in ("알아", "생각", "말해")
    )


def is_child_support_answer_first_request(message: str) -> bool:
    """아이가 질문 회피 없이 먼저 답을 요구하는 턴인지 확인한다."""

    normalized = " ".join(message.lower().split())
    compact = "".join(normalized.split())
    if not normalized:
        return False
    if is_child_support_self_state_question(message):
        return True
    no_question_markers = (
        "질문하지말고",
        "묻지말고",
        "되묻지말고",
        "물어보지말고",
    )
    direct_answer_markers = (
        "먼저답",
        "답부터",
        "네생각",
        "너의생각",
        "너가볼때",
        "네가볼때",
        "어떻게보",
        "말해줘",
        "알려줘",
    )
    return (
        any(marker in compact for marker in no_question_markers)
        or any(marker in compact for marker in direct_answer_markers)
        and any(marker in compact for marker in ("내", "나", "저", "제", "지금"))
    )


def build_child_support_proactive_prompt(
    *,
    context: ChildSupportConversationContext,
    safety_level: ChildSupportSafetyLevel,
) -> str:
    """아이에게 먼저 말을 걸기 위한 LLM prompt를 만든다."""

    return (
        "너는 TraceMind의 아이용 마음 도움 로컬 AI 코치다.\n"
        "아래 로컬 context를 바탕으로 아이에게 먼저 건네는 첫 메시지를 작성한다.\n"
        "원칙:\n"
        "- 한국어 해요체로 2~4문장만 답한다.\n"
        "- 아이를 '당신'이라고 부르지 않는다. 2인칭 호칭은 되도록 생략한다.\n"
        "- 첫 문장은 로컬 context로 보이는 아이의 현재 상태에 자연스럽게 말을 "
        "거는 문장이어야 한다.\n"
        "- 첫 문장을 안부 인사, 하루 안부, 잘 지내길 바란다는 말, 일반적인 "
        "상담 시작 문장으로 쓰지 않는다.\n"
        "- 로컬 context의 캡처/검색 원문은 아이 상태를 이해하기 위해 참고한다.\n"
        "- 상담사의 현재 이해를 먼저 보고 첫 문장을 만든다.\n"
        "- 먼저 말을 걸 때는 원문을 길게 나열하지 말고 반복 주제나 짧은 표현만 "
        "자연스럽게 언급한다.\n"
        "- context로 확인되지 않은 안정 상태나 안심 결론을 먼저 말하지 않는다.\n"
        "- context에 직접 위험 표현이 있으면 평온하거나 잘 지내는 것처럼 말하지 "
        "않는다.\n"
        "- 왜 말을 거는지는 상태 설명처럼 자연스럽게 드러내고, 근거 목록처럼 "
        "나열하지 않는다.\n"
        "- 출력은 순수 답변 문장만 사용하고 장식 문자는 쓰지 않는다.\n"
        "- 진단처럼 말하지 않는다.\n"
        "- 자해/타해 방법, 은폐 방법, 구체적 실행 방법은 절대 제공하지 않는다.\n"
        "- 해결책을 길게 말하지 말고 아이가 이어 말할 수 있는 질문 1개로 끝낸다.\n\n"
        f"{_build_counselor_context_block(context)}\n\n"
        f"safety_level_hint: {safety_level.value}\n\n"
        f"{_build_summary_block(context)}\n\n"
        f"{_build_context_notes_block(context.wellbeing_context_notes)}\n\n"
        "아이에게 바로 보여줄 첫 메시지만 작성해라."
    )


def _build_summary_block(context: ChildSupportConversationContext) -> str:
    summary = context.wellbeing_summary
    if summary is None:
        return "현재 wellbeing summary: 없음"
    return (
        "현재 wellbeing summary:\n"
        f"- level: {summary.signal_level.value}\n"
        f"- score: {summary.signal_score:.1f}\n"
        f"- trend: {summary.trend.value}\n"
        f"- summary: {summary.summary}\n"
        f"- action_tip: {summary.action_tip}"
    )


def _build_context_notes_block(notes: tuple[str, ...]) -> str:
    if not notes:
        return "최근 wellbeing context notes: 없음"
    return "최근 wellbeing context notes:\n" + "\n".join(f"- {note}" for note in notes)


def _build_counselor_context_block(context: ChildSupportConversationContext) -> str:
    return "\n".join(context.counselor_context.to_prompt_lines())


def _build_message_intent_block(
    message: str,
    assessment: ChildSupportSafetyHint,
) -> str:
    if is_child_support_self_state_question(message):
        return (
            "message_intent_hint: self_state_reflection\n"
            "turn_goal: reflect_known_state\n"
            "response_policy: answer_first_no_followup_question\n"
            "intent_rule: 아이가 자기 상태나 감정에 대한 판단을 요청했다. "
            "또는 최근에 무슨 일을 겪었는지 로컬 context가 아는 범위를 물었다. "
            "답변은 질문이 아니라 현재 이해와 확인되는 최근 일을 먼저 말하고, "
            "마지막 문장을 질문으로 끝내지 않는다."
        )
    if is_child_support_answer_first_request(message):
        return (
            "message_intent_hint: direct_answer_request\n"
            "turn_goal: answer_direct_request\n"
            "response_policy: answer_first_no_followup_question\n"
            "intent_rule: 아이가 되묻기보다 먼저 답을 원한다. 답변은 요청한 "
            "내용에 직접 답하고, 마지막 문장을 질문으로 끝내지 않는다."
        )
    if assessment.safety_level == ChildSupportSafetyLevel.URGENT:
        return (
            "message_intent_hint: urgent_support\n"
            "turn_goal: safety_check_and_emotional_support\n"
            "response_policy: supportive_open_followup\n"
            "intent_rule: 즉시 위험 표현이 있다. 방법을 제공하지 말고, 감정은 "
            "짧게 받아 준 뒤 지금 안전과 가까운 도움 연결을 우선한다."
        )
    return (
        "message_intent_hint: general_support\n"
        "turn_goal: supportive_listening\n"
        "response_policy: supportive_open_followup"
    )


def _render_messages(messages: tuple[ChildSupportLlmMessage, ...]) -> str:
    return "\n\n".join(f"{message.role}:\n{message.content}" for message in messages)
