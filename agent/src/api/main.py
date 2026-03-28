"""Agent API 진입점."""

from fastapi import FastAPI

from agent.src.api.routers.assessment import router as assessment_router
from agent.src.api.routers.health import router as health_router
from agent.src.api.routers.ingest import router as ingest_router
from agent.src.api.routers.sync import router as sync_router
from agent.src.api.routers.training import router as training_router

app = FastAPI(title="TraceMind Agent", version="0.1.0")
app.include_router(health_router)
app.include_router(ingest_router)
app.include_router(assessment_router)
app.include_router(sync_router)
app.include_router(training_router)
