"""입력 수집 라우터 자리표시자."""

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/ingest", tags=["ingest"])
