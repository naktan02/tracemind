"""Captured text API and service tests."""

from __future__ import annotations

import concurrent.futures
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from agent.src.api import captured_text as captured_text_api
from agent.src.api.main import app, create_app
from agent.src.contracts.captured_text_contracts import (
    CapturedTextBatchIngestRequestPayload,
    CapturedTextDebugJobConfigRequestPayload,
    CapturedTextDebugJobRunRequestPayload,
    CapturedTextEventPayload,
    CapturedTextSourceType,
    CapturedTextSurfaceType,
)
from agent.src.infrastructure.repositories.captured_text_repository import (
    CapturedTextRepository,
    captured_text_record_from_payload,
)
from agent.src.services.ingest.captured_text_ingest_service import (
    CapturedTextIngestService,
)
from agent.src.services.ingest.captured_text_lifecycle_service import (
    CapturedTextLifecycleConfig,
    CapturedTextLifecycleService,
)
from agent.src.services.ingest.captured_text_view_generation_service import (
    CapturedTextViewGenerationService,
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


def test_captured_text_service_saves_raw_event(
    tmp_path: Path,
) -> None:
    repository = CapturedTextRepository(db_path=tmp_path / "captured_text.db")
    service = CapturedTextIngestService(
        captured_text_repository=repository,
    )

    response = service.process(_event())

    assert response.event_id == "event_1"
    assert response.query_id == "event_1"
    assert response.top_category is None
    assert response.top_score is None
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
        captured_text_repository=repository,
        lifecycle_service=CapturedTextLifecycleService(
            config=CapturedTextLifecycleConfig(retention_days=3, max_records=500)
        ),
    )

    service.process(_event("new_event"))

    assert repository.get("old_event") is None
    assert repository.get("new_event") is not None


def test_captured_text_api_returns_service_response(tmp_path: Path) -> None:
    repository = CapturedTextRepository(db_path=tmp_path / "captured_text.db")
    service = CapturedTextIngestService(
        captured_text_repository=repository,
    )

    response = captured_text_api.ingest_captured_text_event(
        _event(),
        service=service,
    )

    assert response.top_category is None
    assert repository.count() == 1


def test_captured_text_batch_api_processes_events(tmp_path: Path) -> None:
    repository = CapturedTextRepository(db_path=tmp_path / "captured_text.db")
    service = CapturedTextIngestService(
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
    assert [item.top_category for item in response.results] == [None, None]
    assert repository.count() == 2


def test_captured_text_router_is_registered_on_agent_app() -> None:
    route_paths = {route.path for route in app.routes}

    assert "/api/v1/captured-text/events" in route_paths
    assert "/api/v1/captured-text/batch" in route_paths
    assert "/api/v1/captured-text/status" in route_paths
    assert "/api/v1/captured-text/debug-job/status" in route_paths
    assert "/api/v1/captured-text/debug-job/config" in route_paths
    assert "/api/v1/captured-text/debug-job/run-view-generation" in route_paths


def test_captured_text_endpoint_stores_without_pipeline(tmp_path: Path) -> None:
    repository = CapturedTextRepository(db_path=tmp_path / "captured_text.db")
    client = TestClient(create_app(auto_configure_pipeline=False))
    client.app.state.captured_text_repository = repository

    response = client.post(
        "/api/v1/captured-text/events",
        json=_event().model_dump(mode="json"),
    )

    assert response.status_code == 201
    assert repository.get("event_1") is not None


def test_captured_text_endpoint_uses_app_state_dependencies(tmp_path: Path) -> None:
    repository = CapturedTextRepository(db_path=tmp_path / "captured_text.db")
    client = TestClient(
        create_app(
            captured_text_repository=repository,
            auto_configure_pipeline=False,
        )
    )

    response = client.post(
        "/api/v1/captured-text/events",
        json=_event().model_dump(mode="json"),
    )

    assert response.status_code == 201
    assert response.json()["query_id"] == "event_1"
    assert response.json()["top_category"] is None
    assert repository.count() == 1


def test_captured_text_status_reports_view_generation_counts(tmp_path: Path) -> None:
    repository = CapturedTextRepository(db_path=tmp_path / "captured_text.db")
    client = TestClient(
        create_app(
            captured_text_repository=repository,
            auto_configure_pipeline=False,
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
    assert payload["view_generation_status_counts"] == {"pending": 1}
    assert payload["analysis_status_counts"] == {}


def test_captured_text_debug_job_status_reports_pipeline_state(
    tmp_path: Path,
) -> None:
    repository = CapturedTextRepository(db_path=tmp_path / "captured_text.db")
    repository.save(captured_text_record_from_payload(_event("event_1")))
    client = TestClient(
        create_app(
            captured_text_repository=repository,
            captured_text_view_generation_service=CapturedTextViewGenerationService(
                repository=repository
            ),
            auto_configure_pipeline=False,
        )
    )

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
    assert payload["analysis_status_counts"] == {}


def test_captured_text_debug_job_run_generates_views(
    tmp_path: Path,
) -> None:
    repository = CapturedTextRepository(db_path=tmp_path / "captured_text.db")
    repository.save(captured_text_record_from_payload(_event("event_1")))
    client = TestClient(
        create_app(
            captured_text_repository=repository,
            auto_configure_pipeline=False,
        )
    )

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


def test_captured_text_debug_job_does_not_analyze_existing_ready_views(
    tmp_path: Path,
) -> None:
    repository = CapturedTextRepository(db_path=tmp_path / "captured_text.db")
    repository.save(captured_text_record_from_payload(_event("event_1")))
    service = CapturedTextViewGenerationService(repository=repository)
    service.generate_pending_views(limit=10)
    client = TestClient(
        create_app(
            captured_text_repository=repository,
            captured_text_view_generation_service=service,
            auto_configure_pipeline=False,
        )
    )

    response = client.post(
        "/api/v1/captured-text/debug-job/run-view-generation",
        json=CapturedTextDebugJobRunRequestPayload(limit=10).model_dump(mode="json"),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_count"] == 0
    assert payload["generated_count"] == 0
    assert payload["generated_view_count"] == 1


def test_captured_text_debug_job_rejects_concurrent_manual_runs(
    tmp_path: Path,
) -> None:
    repository = CapturedTextRepository(db_path=tmp_path / "captured_text.db")
    started = threading.Event()
    release = threading.Event()

    class BlockingViewGenerationService(CapturedTextViewGenerationService):
        def generate_pending_views(self, *, limit: int = 100):
            started.set()
            assert release.wait(timeout=5)
            return super().generate_pending_views(limit=limit)

    client = TestClient(
        create_app(
            captured_text_repository=repository,
            captured_text_view_generation_service=BlockingViewGenerationService(
                repository=repository
            ),
            auto_configure_pipeline=False,
        )
    )
    payload = CapturedTextDebugJobRunRequestPayload(limit=10).model_dump(mode="json")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        first_response_future = executor.submit(
            client.post,
            "/api/v1/captured-text/debug-job/run-view-generation",
            json=payload,
        )
        assert started.wait(timeout=5)

        second_response = client.post(
            "/api/v1/captured-text/debug-job/run-view-generation",
            json=payload,
        )
        release.set()
        first_response = first_response_future.result(timeout=5)

    assert second_response.status_code == 409
    assert "이미 실행 중" in second_response.json()["detail"]
    assert first_response.status_code == 200


def test_captured_text_debug_job_config_toggles_state(
    tmp_path: Path,
) -> None:
    repository = CapturedTextRepository(db_path=tmp_path / "captured_text.db")
    client = TestClient(
        create_app(
            captured_text_repository=repository,
            auto_configure_pipeline=False,
        )
    )

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
