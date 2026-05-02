"""아이용 지원 대화 agent-local safety intent."""

from __future__ import annotations

from enum import StrEnum


class ChildSupportSafetyIntent(StrEnum):
    """agent 내부 routing에 쓰는 typed intent.

    shared contract의 화면 노출용 safety_level보다 세밀한 내부 분기다.
    """

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


__all__ = ["ChildSupportSafetyIntent"]
