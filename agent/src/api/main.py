"""Agent API 진입점."""

from fastapi import FastAPI

from agent.src.api.routers.health import router as health_router
from agent.src.api.routers.sync import router as sync_router

app = FastAPI(title="TraceMind Agent", version="0.1.0")
app.include_router(health_router)
app.include_router(sync_router)
