"""Captured text contract unit tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from shared.src.contracts.captured_text_contracts import (
    CapturedTextBatchIngestRequestPayload,
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
