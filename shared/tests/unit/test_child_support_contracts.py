"""Child support conversation contract tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from shared.src.contracts.child_support_contracts import (
    ChildSupportAssistantMode,
    ChildSupportConversationRequestPayload,
    ChildSupportConversationResponsePayload,
    ChildSupportSafetyLevel,
    ChildSupportSuggestionPayload,
)


def test_child_support_request_accepts_minimal_message() -> None:
    payload = ChildSupportConversationRequestPayload(message="지금 마음이 답답해")

    assert payload.message == "지금 마음이 답답해"
    assert payload.conversation_id is None


def test_child_support_request_rejects_empty_message() -> None:
    with pytest.raises(ValidationError):
        ChildSupportConversationRequestPayload(message="")


def test_child_support_response_accepts_canonical_fields() -> None:
    payload = ChildSupportConversationResponsePayload(
        conversation_id="conversation-1",
        message_id="message-1",
        created_at=datetime(2026, 4, 25, 10, 30, tzinfo=timezone.utc),
        assistant_mode=ChildSupportAssistantMode.LOCAL_GUARDED,
        safety_level=ChildSupportSafetyLevel.CHECK_IN,
        reply_text="먼저 숨을 천천히 쉬어볼게요.",
        suggested_prompts=(
            ChildSupportSuggestionPayload(
                id="breathe",
                label="숨 쉬기",
                prompt="숨 쉬기 도와줘",
            ),
        ),
        parent_handoff_suggested=False,
        disclosure_notice="진단이나 상담을 대신하지 않습니다.",
    )

    assert payload.schema_version == "child_support_response.v1"
    assert payload.conversation_id == "conversation-1"
    assert payload.assistant_mode == ChildSupportAssistantMode.LOCAL_GUARDED
    assert payload.safety_level == ChildSupportSafetyLevel.CHECK_IN
    assert payload.suggested_prompts[0].id == "breathe"
