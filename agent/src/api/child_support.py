"""아이용 지원 대화 API."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from agent.src.contracts.child_support_contracts import (
    ChildSupportConversationRequestPayload,
    ChildSupportConversationResponsePayload,
    ChildSupportProactivePromptPayload,
)
from agent.src.features.wellbeing.child_support.service import (
    ChildSupportCoachService,
)

router = APIRouter(prefix="/api/v1/child-support", tags=["child-support"])


def get_child_support_coach_service(request: Request) -> ChildSupportCoachService:
    """app.state에서 ChildSupportCoachService를 읽는다."""

    service = getattr(request.app.state, "child_support_coach_service", None)
    if service is None:
        raise RuntimeError(
            "ChildSupportCoachService가 app.state에 설정되지 않았습니다. "
            "앱 생성 시 app.state.child_support_coach_service를 설정하세요."
        )
    return service


ChildSupportCoachServiceDep = Annotated[
    ChildSupportCoachService,
    Depends(get_child_support_coach_service),
]


@router.post(
    "/messages",
    response_model=ChildSupportConversationResponsePayload,
)
def create_child_support_message(
    request: ChildSupportConversationRequestPayload,
    service: ChildSupportCoachServiceDep,
) -> ChildSupportConversationResponsePayload:
    """아이용 지원 대화 단일 turn 응답을 만든다."""

    return service.create_response(request)


@router.get(
    "/proactive-prompt",
    response_model=ChildSupportProactivePromptPayload,
)
def get_child_support_proactive_prompt(
    service: ChildSupportCoachServiceDep,
) -> ChildSupportProactivePromptPayload:
    """아이 화면 진입 시 먼저 건넬 말이 필요한지 반환한다."""

    return service.build_proactive_prompt()
