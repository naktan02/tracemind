"""아이용 지원 대화 LLM prompt 생성."""

from __future__ import annotations

from typing import Protocol

from agent.src.contracts.child_support_contracts import ChildSupportSafetyLevel
from agent.src.features.wellbeing.child_support.context_provider import (
    ChildSupportConversationContext,
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

    message_lines = [
        f"- {record.role}: {record.text}" for record in context.recent_messages[-6:]
    ]
    history_block = (
        "현재 대화 최근 히스토리:\n" + "\n".join(message_lines)
        if message_lines
        else "현재 대화 최근 히스토리: 없음"
    )
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
        "- 로컬 context의 캡처/검색 원문은 아이 상태를 이해하기 위해 참고한다.\n"
        "- 답변에서 원문을 언급해도 되지만, 아이가 감시당한다고 느끼지 않게 "
        "필요한 부분만 짧게 연결한다.\n"
        "- 아이가 설명이나 판단을 요청하면 곧바로 짧게 답한 뒤, 필요한 경우에만 "
        "이어 말할 질문을 덧붙인다.\n"
        "- 로컬 context에서 확인되는 최근 흐름과 대화 내용을 근거로 답하되, "
        "확인되지 않은 정보는 안다고 말하지 않는다.\n"
        "- 원인을 설명할 때는 가능성으로 말하고 단정하지 않는다.\n"
        "- 출력은 순수 답변 문장만 사용하고 장식 문자는 쓰지 않는다.\n"
        "- 진단, 치료, 법률 조언처럼 말하지 않는다.\n"
        "- 자해/타해 방법, 은폐 방법, 구체적 실행 방법은 절대 제공하지 않는다.\n"
        "- 즉시 위험이 보이면 지금 안전한지 먼저 확인한다.\n"
        "- 대화를 닫는 말보다 아이가 이어 말할 수 있는 질문 1개로 끝낸다.\n\n"
        f"safety_level_hint: {assessment.safety_level.value}\n"
        f"safety_reason_hint: {assessment.reason}\n\n"
        f"{_build_summary_block(context)}\n\n"
        f"{_build_context_notes_block(context.wellbeing_context_notes)}\n\n"
        f"{history_block}\n\n"
        f"아이의 새 메시지: {message}\n\n"
        "아이에게 바로 보여줄 답변만 작성해라."
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
        "- 넓은 안부 질문으로 시작하지 않는다.\n"
        "- 로컬 context의 캡처/검색 원문은 아이 상태를 이해하기 위해 참고한다.\n"
        "- 먼저 말을 걸 때는 원문을 길게 나열하지 말고 반복 주제나 짧은 표현만 "
        "자연스럽게 언급한다.\n"
        "- context로 확인되지 않은 안정 상태나 안심 결론을 먼저 말하지 않는다.\n"
        "- 왜 말을 거는지 최근 위험 신호나 반복 주제와 연결해 짧게 밝힌다.\n"
        "- 출력은 순수 답변 문장만 사용하고 장식 문자는 쓰지 않는다.\n"
        "- 진단처럼 말하지 않는다.\n"
        "- 자해/타해 방법, 은폐 방법, 구체적 실행 방법은 절대 제공하지 않는다.\n"
        "- 해결책을 길게 말하지 말고 아이가 이어 말할 수 있는 질문 1개로 끝낸다.\n\n"
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
