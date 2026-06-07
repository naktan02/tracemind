"""Captured text API and service tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from agent.src.api import captured_text as captured_text_api
from agent.src.api.main import app, create_app
from agent.src.infrastructure.repositories.captured_text_repository import (
    CapturedTextRepository,
    captured_text_record_from_payload,
)
from agent.src.services.inference.pipeline_service import InferencePipelineResult
from agent.src.services.ingest.captured_text_ingest_service import (
    CapturedTextIngestService,
)
from agent.src.services.ingest.captured_text_lifecycle_service import (
    CapturedTextLifecycleConfig,
    CapturedTextLifecycleService,
)
from shared.src.contracts.captured_text_contracts import (
    CapturedTextBatchIngestRequestPayload,
    CapturedTextDebugJobConfigRequestPayload,
    CapturedTextDebugJobRunRequestPayload,
    CapturedTextEventPayload,
    CapturedTextSourceType,
    CapturedTextSurfaceType,
)
from shared.src.domain.entities.inference.events import AnalysisEvent


@dataclass(slots=True)
class StubEventRepository:
    count_value: int = 0

    def count(self) -> int:
        return self.count_value


@dataclass(slots=True)
class StubPipelineService:
    """captured text service test용 pipeline stub."""

    processed_texts: list[str]
    processed_source_types: list[str]
    event_repository: StubEventRepository

    def process(self, event):
        self.processed_texts.append(event.text)
        self.processed_source_types.append(event.source_type)
        self.event_repository.count_value += 1
        return InferencePipelineResult(
            analysis_event=AnalysisEvent(
                query_id=event.query_id,
                occurred_at=event.occurred_at,
                translated_text=None,
                embedding_model_id="test",
                translation_model_id=None,
                category_scores={"risk": 0.9, "neutral": 0.1},
            ),
            base_embedding=[0.1],
            was_translated=False,
        )


def _event(event_id: str = "event_1") -> CapturedTextEventPayload:
    return CapturedTextEventPayload(
        event_id=event_id,
        occurred_at=datetime(2026, 6, 6, 1, 0, tzinfo=timezone.utc),
        text="오늘 너무 불안해",
        locale="ko",
        source_type=CapturedTextSourceType.SEARCH,
        surface_type=CapturedTextSurfaceType.SEARCH_BOX,
        page_url="https://example.test/search",
        collector_version="collector@1",
    )


def _pipeline() -> StubPipelineService:
    return StubPipelineService(
        processed_texts=[],
        processed_source_types=[],
        event_repository=StubEventRepository(),
    )


def test_captured_text_service_saves_raw_event_and_processes_query_event(
    tmp_path: Path,
) -> None:
    pipeline = _pipeline()
    repository = CapturedTextRepository(db_path=tmp_path / "captured_text.db")
    service = CapturedTextIngestService(
        pipeline_service=pipeline,
        captured_text_repository=repository,
    )

    response = service.process(_event())

    assert response.event_id == "event_1"
    assert response.query_id == "event_1"
    assert response.top_category == "risk"
    assert response.top_score == 0.9
    assert pipeline.processed_texts == ["오늘 너무 불안해"]
    assert pipeline.processed_source_types == ["search:search_box"]
    stored = repository.get("event_1")
    assert stored is not None
    assert stored.source_type == "search"
    assert stored.surface_type == "search_box"


def test_captured_text_service_applies_lifecycle_after_ingest(
    tmp_path: Path,
) -> None:
    repository = CapturedTextRepository(db_path=tmp_path / "captured_text.db")
    old_event = _event("old_event").model_copy(
        update={"occurred_at": datetime.now(tz=timezone.utc) - timedelta(days=4)}
    )
    repository.save(captured_text_record_from_payload(old_event))
    service = CapturedTextIngestService(
        pipeline_service=_pipeline(),
        captured_text_repository=repository,
        lifecycle_service=CapturedTextLifecycleService(
            config=CapturedTextLifecycleConfig(retention_days=3, max_records=500)
        ),
    )

    service.process(_event("new_event"))

    assert repository.get("old_event") is None
    assert repository.get("new_event") is not None


def test_captured_text_api_returns_service_response(tmp_path: Path) -> None:
    pipeline = _pipeline()
    repository = CapturedTextRepository(db_path=tmp_path / "captured_text.db")
    service = CapturedTextIngestService(
        pipeline_service=pipeline,
        captured_text_repository=repository,
    )

    response = captured_text_api.ingest_captured_text_event(
        _event(),
        service=service,
    )

    assert response.top_category == "risk"
    assert repository.count() == 1


def test_captured_text_batch_api_processes_events(tmp_path: Path) -> None:
    pipeline = _pipeline()
    repository = CapturedTextRepository(db_path=tmp_path / "captured_text.db")
    service = CapturedTextIngestService(
        pipeline_service=pipeline,
        captured_text_repository=repository,
    )

    response = captured_text_api.ingest_captured_text_batch(
        CapturedTextBatchIngestRequestPayload(
            events=[_event("event_1"), _event("event_2")]
        ),
        service=service,
    )

    assert response.processed == 2
    assert [item.event_id for item in response.results] == ["event_1", "event_2"]
    assert repository.count() == 2


def test_captured_text_router_is_registered_on_agent_app() -> None:
    route_paths = {route.path for route in app.routes}

    assert "/api/v1/captured-text/events" in route_paths
    assert "/api/v1/captured-text/batch" in route_paths
    assert "/api/v1/captured-text/status" in route_paths
    assert "/api/v1/captured-text/debug-job/status" in route_paths
    assert "/api/v1/captured-text/debug-job/config" in route_paths
    assert "/api/v1/captured-text/debug-job/run-view-generation" in route_paths


def test_captured_text_endpoint_reports_missing_pipeline_without_traceback() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/api/v1/captured-text/events",
        json=_event().model_dump(mode="json"),
    )

    assert response.status_code == 503
    assert "pipeline_service" in response.json()["detail"]


def test_captured_text_endpoint_uses_app_state_dependencies(tmp_path: Path) -> None:
    pipeline = _pipeline()
    repository = CapturedTextRepository(db_path=tmp_path / "captured_text.db")
    client = TestClient(
        create_app(
            pipeline_service=pipeline,  # type: ignore[arg-type]
            captured_text_repository=repository,
        )
    )

    response = client.post(
        "/api/v1/captured-text/events",
        json=_event().model_dump(mode="json"),
    )

    assert response.status_code == 201
    assert response.json()["query_id"] == "event_1"
    assert repository.count() == 1
    assert pipeline.processed_source_types == ["search:search_box"]


def test_captured_text_status_reports_view_generation_counts(tmp_path: Path) -> None:
    pipeline = _pipeline()
    repository = CapturedTextRepository(db_path=tmp_path / "captured_text.db")
    client = TestClient(
        create_app(
            pipeline_service=pipeline,  # type: ignore[arg-type]
            captured_text_repository=repository,
        )
    )
    response = client.post(
        "/api/v1/captured-text/events",
        json=_event().model_dump(mode="json"),
    )
    assert response.status_code == 201

    status_response = client.get("/api/v1/captured-text/status")

    assert status_response.status_code == 200
    payload = status_response.json()
    assert payload["captured_text_event_count"] == 1
    assert payload["stored_event_count"] == 1
    assert payload["view_generation_status_counts"] == {"pending": 1}


def test_captured_text_debug_job_status_reports_pipeline_state(
    tmp_path: Path,
) -> None:
    repository = CapturedTextRepository(db_path=tmp_path / "captured_text.db")
    repository.save(captured_text_record_from_payload(_event("event_1")))
    client = TestClient(create_app(captured_text_repository=repository))

    response = client.get("/api/v1/captured-text/debug-job/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["view_generation_enabled"] is False
    assert payload["view_generation_running"] is False
    assert payload["weak_text_provider_name"] == "identity"
    assert payload["strong_text_provider_name"] == "identity"
    assert payload["weak_text_identity_fallback"] is True
    assert payload["strong_text_identity_fallback"] is True
    assert payload["captured_text_event_count"] == 1
    assert payload["generated_view_count"] == 0
    assert payload["view_generation_status_counts"] == {"pending": 1}


def test_captured_text_debug_job_run_generates_views(
    tmp_path: Path,
) -> None:
    repository = CapturedTextRepository(db_path=tmp_path / "captured_text.db")
    repository.save(captured_text_record_from_payload(_event("event_1")))
    client = TestClient(create_app(captured_text_repository=repository))

    response = client.post(
        "/api/v1/captured-text/debug-job/run-view-generation",
        json=CapturedTextDebugJobRunRequestPayload(limit=10).model_dump(mode="json"),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_count"] == 1
    assert payload["generated_count"] == 1
    assert payload["generated_view_count"] == 1
    assert repository.get_generated_view("event_1") is not None


def test_captured_text_debug_job_config_toggles_state(
    tmp_path: Path,
) -> None:
    repository = CapturedTextRepository(db_path=tmp_path / "captured_text.db")
    client = TestClient(create_app(captured_text_repository=repository))

    response = client.post(
        "/api/v1/captured-text/debug-job/config",
        json=CapturedTextDebugJobConfigRequestPayload(
            view_generation_enabled=True,
            view_generation_interval_seconds=30,
            view_generation_batch_size=10,
        ).model_dump(mode="json"),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["view_generation_enabled"] is True
    assert payload["view_generation_interval_seconds"] == 30
    assert payload["view_generation_batch_size"] == 10

    off_response = client.post(
        "/api/v1/captured-text/debug-job/config",
        json=CapturedTextDebugJobConfigRequestPayload(
            view_generation_enabled=False,
            view_generation_interval_seconds=30,
            view_generation_batch_size=10,
        ).model_dump(mode="json"),
    )

    assert off_response.status_code == 200
    assert off_response.json()["view_generation_enabled"] is False
