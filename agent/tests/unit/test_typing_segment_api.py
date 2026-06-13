"""Typing segment API and service tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from agent.src.api import typing_segments as typing_segments_api
from agent.src.api.main import app, create_app
from agent.src.contracts.typing_segment_contracts import (
    TypingSegmentBatchIngestRequestPayload,
    TypingSegmentPayload,
    TypingSurfaceType,
)
from agent.src.features.inference.pipeline_service import InferencePipelineResult
from agent.src.features.typing_segments.ingest_service import (
    TypingSegmentIngestService,
)
from shared.src.domain.entities.inference.events import AnalysisEvent


@dataclass(slots=True)
class StubPipelineService:
    """typing segment service test용 pipeline stub."""

    processed_texts: list[str]
    processed_source_types: list[str]

    def process(self, event):
        self.processed_texts.append(event.text)
        self.processed_source_types.append(event.source_type)
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


def _segment(segment_id: str = "segment_1") -> TypingSegmentPayload:
    return TypingSegmentPayload(
        segment_id=segment_id,
        surface_type=TypingSurfaceType.TEXTAREA,
        started_at=datetime(2026, 6, 3, 1, 0, tzinfo=timezone.utc),
        ended_at=datetime(2026, 6, 3, 1, 0, 5, tzinfo=timezone.utc),
        idle_ms=5000,
        final_text="오늘 너무 불안해",
    )


def test_typing_segment_service_converts_segment_to_query_event() -> None:
    pipeline = StubPipelineService(processed_texts=[], processed_source_types=[])
    service = TypingSegmentIngestService(pipeline_service=pipeline)

    response = service.process(_segment())

    assert response.segment_id == "segment_1"
    assert response.query_id == "segment_1"
    assert response.top_category == "risk"
    assert response.top_score == 0.9
    assert pipeline.processed_texts == ["오늘 너무 불안해"]
    assert pipeline.processed_source_types == ["unknown:textarea"]


def test_typing_segment_api_returns_service_response() -> None:
    pipeline = StubPipelineService(processed_texts=[], processed_source_types=[])
    service = TypingSegmentIngestService(pipeline_service=pipeline)

    response = typing_segments_api.ingest_typing_segment(_segment(), service=service)

    assert response.top_category == "risk"
    assert pipeline.processed_texts == ["오늘 너무 불안해"]


def test_typing_segment_batch_api_processes_segments() -> None:
    pipeline = StubPipelineService(processed_texts=[], processed_source_types=[])
    service = TypingSegmentIngestService(pipeline_service=pipeline)

    response = typing_segments_api.ingest_typing_segment_batch(
        TypingSegmentBatchIngestRequestPayload(
            segments=[_segment("segment_1"), _segment("segment_2")]
        ),
        service=service,
    )

    assert response.processed == 2
    assert [item.segment_id for item in response.results] == [
        "segment_1",
        "segment_2",
    ]


def test_typing_segment_router_is_registered_on_agent_app() -> None:
    route_paths = {route.path for route in app.routes}

    assert "/api/v1/typing-segments" in route_paths
    assert "/api/v1/typing-segments/batch" in route_paths


def test_typing_segment_endpoint_reports_missing_pipeline_without_traceback() -> None:
    client = TestClient(create_app(auto_configure_pipeline=False))

    response = client.post(
        "/api/v1/typing-segments",
        json=_segment().model_dump(mode="json"),
    )

    assert response.status_code == 503
    assert "pipeline_service" in response.json()["detail"]
