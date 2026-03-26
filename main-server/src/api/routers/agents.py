"""FL agent 등록 라우터 자리표시자."""

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/agents", tags=["agents"])
