"""Captured text contract unit tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from agent.src.contracts.captured_text_contracts import (
    CapturedTextBatchIngestRequestPayload,
    CapturedTextDebugJobConfigRequestPayload,
    CapturedTextDebugJobRunResultPayload,
    CapturedTextDebugJobStatusPayload,
    CapturedTextEventPayload,
    CapturedTextSourceType,
    CapturedTextSurfaceType,
)


def test_captured_text_event_payload_accepts_canonical_fields() -> None:
    payload = CapturedTextEventPayload(
        event_id="event_1",
        occurred_at=datetime(2026, 6, 6, 1, 0, tzinfo=timezone.utc),
        text="오늘 학교 검색을 많이 했다",
        locale="ko",
        source_type=CapturedTextSourceType.SEARCH,
        surface_type=CapturedTextSurfaceType.SEARCH_BOX,
        page_url="https://example.test/search?q=test",
        page_title="Search",
        collector_version="family-extension@0.1.0",
        metadata={"tab_id": 1},
    )

    assert payload.schema_version == "captured_text_event.v1"
    assert payload.source_type == CapturedTextSourceType.SEARCH
    assert payload.surface_type == CapturedTextSurfaceType.SEARCH_BOX
    assert payload.metadata["tab_id"] == 1


def test_captured_text_event_rejects_blank_text() -> None:
    with pytest.raises(ValidationError, match="text must not be blank"):
        CapturedTextEventPayload(
            event_id="event_2",
            occurred_at=datetime(2026, 6, 6, 1, 0, tzinfo=timezone.utc),
            text="   ",
        )


def test_captured_text_batch_has_size_limit() -> None:
    event = CapturedTextEventPayload(
        event_id="event_3",
        occurred_at=datetime(2026, 6, 6, 1, 0, tzinfo=timezone.utc),
        text="reddit comment text",
        source_type=CapturedTextSourceType.REDDIT,
        surface_type=CapturedTextSurfaceType.REDDIT_COMMENT,
    )

    batch = CapturedTextBatchIngestRequestPayload(events=[event])

    assert batch.events == (event,)


def test_captured_text_debug_job_contracts_are_strict() -> None:
    config = CapturedTextDebugJobConfigRequestPayload(
        view_generation_enabled=True,
        view_generation_interval_seconds=30,
        view_generation_batch_size=20,
    )
    result = CapturedTextDebugJobRunResultPayload(
        selected_count=2,
        generated_count=2,
        failed_count=0,
        pending_remaining_count=0,
        generated_view_count=2,
    )

    status = CapturedTextDebugJobStatusPayload(
        view_generation_enabled=config.view_generation_enabled,
        view_generation_running=True,
        view_generation_interval_seconds=config.view_generation_interval_seconds,
        view_generation_batch_size=config.view_generation_batch_size,
        weak_text_provider_name="nllb",
        strong_text_provider_name="nllb_backtranslation",
        weak_text_identity_fallback=False,
        strong_text_identity_fallback=False,
        captured_text_event_count=2,
        generated_view_count=2,
        view_generation_status_counts={"ready": 2},
        last_run_result=result,
    )

    assert status.schema_version == "captured_text_debug_job_status.v1"
    assert status.last_run_result is not None
    assert status.last_run_result.generated_count == 2
    assert status.last_run_result.analysis_processed_count == 0
