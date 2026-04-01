"""로컬 agent 상태 확인 라우터."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def healthcheck() -> dict[str, str]:
    """로컬 개발용 최소 상태 확인 엔드포인트를 노출한다."""
    return {"status": "ok", "service": "agent"}
