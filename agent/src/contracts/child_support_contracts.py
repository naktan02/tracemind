"""아이용 지원 대화 API 계약.

이 계약은 family extension이 로컬 agent의 지원 대화 surface를 호출할 때 쓰는
최소 payload다. 실제 LLM provider 선택, prompt, safety policy 실행 방식은
agent runtime이 소유하고 UI는 응답을 표시만 한다.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

CHILD_SUPPORT_RESPONSE_V1 = "child_support_response.v1"
CHILD_SUPPORT_PROACTIVE_PROMPT_V1 = "child_support_proactive_prompt.v1"
CHILD_SUPPORT_PROACTIVE_PROMPT_CLAIM_V1 = "child_support_proactive_prompt_claim.v1"

ChildSupportResponseSchemaVersion: TypeAlias = Literal["child_support_response.v1"]
ChildSupportProactivePromptSchemaVersion: TypeAlias = Literal[
    "child_support_proactive_prompt.v1"
]
ChildSupportProactivePromptClaimSchemaVersion: TypeAlias = Literal[
    "child_support_proactive_prompt_claim.v1"
]


class ChildSupportAssistantMode(StrEnum):
    """아이용 지원 대화 응답을 만든 backend mode."""

    LOCAL_GUARDED = "local_guarded"
    LOCAL_LLM = "local_llm"
    LLM = "llm"


class ChildSupportSafetyLevel(StrEnum):
    """아이 화면에 노출 가능한 지원 응답의 safety 단계."""

    SUPPORTIVE = "supportive"
    CHECK_IN = "check_in"
    PARENT_HANDOFF = "parent_handoff"
    URGENT = "urgent"


class ChildSupportScopeStatus(StrEnum):
    """아이용 마음 도움 대화가 지원 범위 안에 있는지의 상태."""

    IN_SCOPE = "in_scope"
    REDIRECTED = "redirected"


class ChildSupportSuggestionPayload(BaseModel):
    """아이용 대화 입력 suggestion."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    prompt: str = Field(min_length=1)


class ChildSupportConversationRequestPayload(BaseModel):
    """아이용 지원 대화 단일 turn 요청."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=1000)
    conversation_id: str | None = Field(default=None, min_length=1)


class ChildSupportConversationResponsePayload(BaseModel):
    """아이용 지원 대화 단일 turn 응답."""

    model_config = ConfigDict(extra="forbid")

    schema_version: ChildSupportResponseSchemaVersion = CHILD_SUPPORT_RESPONSE_V1
    conversation_id: str = Field(min_length=1)
    message_id: str = Field(min_length=1)
    created_at: datetime
    assistant_mode: ChildSupportAssistantMode
    safety_level: ChildSupportSafetyLevel
    scope_status: ChildSupportScopeStatus = ChildSupportScopeStatus.IN_SCOPE
    reply_text: str = Field(min_length=1)
    suggested_prompts: tuple[ChildSupportSuggestionPayload, ...] = Field(
        default_factory=tuple
    )
    parent_handoff_suggested: bool = False
    parent_handoff_label: str | None = None
    disclosure_notice: str = Field(min_length=1)


class ChildSupportProactivePromptPayload(BaseModel):
    """아이 화면 진입 시 agent가 먼저 건넬 수 있는 말."""

    model_config = ConfigDict(extra="forbid")

    schema_version: ChildSupportProactivePromptSchemaVersion = (
        CHILD_SUPPORT_PROACTIVE_PROMPT_V1
    )
    should_prompt: bool
    prompt_id: str | None = Field(default=None, min_length=1)
    conversation_id: str | None = Field(default=None, min_length=1)
    safety_level: ChildSupportSafetyLevel | None = None
    prompt_text: str | None = None
    suggested_prompts: tuple[ChildSupportSuggestionPayload, ...] = Field(
        default_factory=tuple
    )


class ChildSupportProactivePromptClaimRequestPayload(BaseModel):
    """선제 발화를 실제 표시 직전에 대화로 claim하는 요청."""

    model_config = ConfigDict(extra="forbid")

    schema_version: ChildSupportProactivePromptClaimSchemaVersion = (
        CHILD_SUPPORT_PROACTIVE_PROMPT_CLAIM_V1
    )
    prompt_id: str = Field(min_length=1)
