"""아이용 지원 대화 API."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from agent.src.api.dependencies import ChildSupportCoachServiceDep
from agent.src.contracts.child_support_contracts import (
    ChildSupportConversationRequestPayload,
    ChildSupportConversationResponsePayload,
    ChildSupportProactivePromptClaimRequestPayload,
    ChildSupportProactivePromptPayload,
)
from agent.src.features.wellbeing.child_support.service import (
    ChildSupportReplyUnavailable,
)

router = APIRouter(prefix="/api/v1/child-support", tags=["child-support"])


@router.post(
    "/messages",
    response_model=ChildSupportConversationResponsePayload,
)
def create_child_support_message(
    request: ChildSupportConversationRequestPayload,
    service: ChildSupportCoachServiceDep,
) -> ChildSupportConversationResponsePayload:
    """아이용 지원 대화 단일 turn 응답을 만든다."""

    try:
        return service.create_response(request)
    except ChildSupportReplyUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.get(
    "/proactive-prompt",
    response_model=ChildSupportProactivePromptPayload,
)
def get_child_support_proactive_prompt(
    service: ChildSupportCoachServiceDep,
) -> ChildSupportProactivePromptPayload:
    """아이 화면 진입 시 먼저 건넬 말이 필요한지 반환한다."""

    return service.build_proactive_prompt()


@router.post(
    "/proactive-prompt/claim",
    response_model=ChildSupportProactivePromptPayload,
)
def claim_child_support_proactive_prompt(
    request: ChildSupportProactivePromptClaimRequestPayload,
    service: ChildSupportCoachServiceDep,
) -> ChildSupportProactivePromptPayload:
    """선제 발화를 실제 표시 직전에 대화로 claim한다."""

    return service.claim_proactive_prompt(request)
