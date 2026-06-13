"""아이용 지원 대화 LLM prompt 생성."""

from __future__ import annotations

from agent.src.contracts.child_support_contracts import ChildSupportSafetyLevel
from agent.src.features.wellbeing.child_support.context_provider import (
    ChildSupportConversationContext,
)
from agent.src.features.wellbeing.child_support.safety_policy import (
    ChildSupportSafetyAssessment,
)


def build_child_support_conversation_prompt(
    *,
    message: str,
    context: ChildSupportConversationContext,
    assessment: ChildSupportSafetyAssessment,
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
        "원칙:\n"
        "- 한국어 해요체로 3~6문장만 답한다.\n"
        "- 로컬 context의 캡처/검색 원문은 아이 상태를 이해하기 위해 참고한다.\n"
        "- 답변에서 원문을 언급해도 되지만, 아이가 감시당한다고 느끼지 않게 "
        "필요한 부분만 짧게 연결한다.\n"
        "- 아이가 왜 힘든지 묻는다면, 상황과 context를 바탕으로 가능한 이유를 "
        "단정하지 않고 설명한다.\n"
        "- 진단, 치료, 법률 조언처럼 말하지 않는다.\n"
        "- 자해/타해 방법, 은폐 방법, 구체적 실행 방법은 절대 제공하지 않는다.\n"
        "- 즉시 위험이 보이면 지금 안전한지 먼저 확인한다.\n"
        "- 대화를 닫는 말보다 아이가 이어 말할 수 있는 질문 1개로 끝낸다.\n\n"
        f"safety_level_hint: {assessment.safety_level.value}\n"
        f"safety_reason_hint: {assessment.reason}\n"
        f"immediate_danger_hint: {assessment.immediate_danger}\n\n"
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
        "- 로컬 context의 캡처/검색 원문은 아이 상태를 이해하기 위해 참고한다.\n"
        "- 먼저 말을 걸 때는 원문을 길게 나열하지 말고 반복 주제나 짧은 표현만 "
        "자연스럽게 언급한다.\n"
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
