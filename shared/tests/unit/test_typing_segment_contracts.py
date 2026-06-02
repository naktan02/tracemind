"""Typing segment contract unit tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from shared.src.contracts.typing_segment_contracts import (
    TypingCaptureConfidence,
    TypingSegmentBatchIngestRequestPayload,
    TypingSegmentPayload,
    TypingSegmentSourceType,
    TypingSegmentStatsPayload,
    TypingSurfaceType,
)


def test_typing_segment_payload_accepts_canonical_fields() -> None:
    payload = TypingSegmentPayload(
        segment_id="segment_1",
        source_type=TypingSegmentSourceType.BROWSER_EXTENSION,
        surface_type=TypingSurfaceType.TEXTAREA,
        capture_confidence=TypingCaptureConfidence.HIGH,
        page_origin="https://example.test",
        started_at=datetime(2026, 6, 3, 1, 0, tzinfo=timezone.utc),
        ended_at=datetime(2026, 6, 3, 1, 0, 5, tzinfo=timezone.utc),
        idle_ms=5000,
        final_text="오늘 너무 불안해",
        deleted_text="죽고 싶",
        stats=TypingSegmentStatsPayload(
            insert_count=8,
            delete_count=1,
            composition_count=2,
        ),
    )

    assert payload.schema_version == "typing_segment.v1"
    assert payload.analysis_text == "오늘 너무 불안해"
    assert payload.source_type == TypingSegmentSourceType.BROWSER_EXTENSION


def test_typing_segment_uses_deleted_text_when_final_text_is_empty() -> None:
    payload = TypingSegmentPayload(
        segment_id="segment_2",
        started_at=datetime(2026, 6, 3, 1, 0, tzinfo=timezone.utc),
        ended_at=datetime(2026, 6, 3, 1, 0, 5, tzinfo=timezone.utc),
        idle_ms=5000,
        final_text="  ",
        deleted_text="사라지고 싶다",
    )

    assert payload.analysis_text == "사라지고 싶다"


def test_typing_segment_rejects_empty_analysis_text() -> None:
    with pytest.raises(ValidationError, match="final_text or deleted_text"):
        TypingSegmentPayload(
            segment_id="segment_3",
            started_at=datetime(2026, 6, 3, 1, 0, tzinfo=timezone.utc),
            ended_at=datetime(2026, 6, 3, 1, 0, 5, tzinfo=timezone.utc),
            idle_ms=5000,
        )


def test_typing_segment_rejects_reversed_time() -> None:
    with pytest.raises(ValidationError, match="ended_at"):
        TypingSegmentPayload(
            segment_id="segment_4",
            started_at=datetime(2026, 6, 3, 1, 1, tzinfo=timezone.utc),
            ended_at=datetime(2026, 6, 3, 1, 0, tzinfo=timezone.utc),
            idle_ms=5000,
            final_text="불안해",
        )


def test_typing_segment_batch_has_size_limit() -> None:
    segment = TypingSegmentPayload(
        segment_id="segment_5",
        started_at=datetime(2026, 6, 3, 1, 0, tzinfo=timezone.utc),
        ended_at=datetime(2026, 6, 3, 1, 0, 5, tzinfo=timezone.utc),
        idle_ms=5000,
        final_text="불안해",
    )

    batch = TypingSegmentBatchIngestRequestPayload(segments=[segment])

    assert batch.segments == (segment,)
