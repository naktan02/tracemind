"""요약 동기화 라우터 자리표시자."""

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/sync", tags=["sync"])
