"""중앙 서버 API 진입점."""

from fastapi import FastAPI

from main_server.src.api.health import router as health_router
from main_server.src.api.prototypes import router as prototypes_router

app = FastAPI(title="TraceMind Main Server", version="0.1.0")
app.include_router(health_router)
app.include_router(prototypes_router)
