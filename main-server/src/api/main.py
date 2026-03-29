"""중앙 서버 API 진입점."""

from fastapi import FastAPI

from src.api.routers.agents import router as agents_router
from src.api.routers.fl_rounds import router as fl_rounds_router
from src.api.routers.health import router as health_router
from src.api.routers.prototypes import router as prototypes_router

app = FastAPI(title="TraceMind Main Server", version="0.1.0")
app.include_router(health_router)
app.include_router(prototypes_router)
app.include_router(agents_router)
app.include_router(fl_rounds_router)
