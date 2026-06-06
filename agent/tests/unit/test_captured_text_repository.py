"""CapturedTextRepository 단위 테스트."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent.src.infrastructure.repositories.captured_text_repository import (
    CapturedTextRecord,
    CapturedTextRepository,
    captured_text_record_from_payload,
)
from shared.src.contracts.captured_text_contracts import (
    CapturedTextEventPayload,
    CapturedTextSourceType,
    CapturedTextSurfaceType,
)


@pytest.fixture
def tmp_repo(tmp_path: Path) -> CapturedTextRepository:
    """임시 SQLite 경로를 쓰는 captured text 저장소."""

    return CapturedTextRepository(db_path=tmp_path / "captured_text.db")


def _make_record(
    *,
    event_id: str = "event_1",
    occurred_at: datetime | None = None,
    text: str = "불안해",
) -> CapturedTextRecord:
    return CapturedTextRecord(
        event_id=event_id,
        occurred_at=occurred_at or datetime.now(tz=timezone.utc),
        received_at=datetime.now(tz=timezone.utc),
        text=text,
        locale="ko",
        source_type="search",
        surface_type="search_box",
        page_url="https://example.test",
        metadata={"source": "unit-test"},
    )


def test_repo_save_and_get_round_trips_record(
    tmp_repo: CapturedTextRepository,
) -> None:
    record = _make_record()

    tmp_repo.save(record)

    loaded = tmp_repo.get("event_1")
    assert loaded is not None
    assert loaded.event_id == "event_1"
    assert loaded.text == "불안해"
    assert loaded.source_type == "search"
    assert loaded.surface_type == "search_box"
    assert loaded.metadata["source"] == "unit-test"


def test_repo_save_overwrites_same_event_id(
    tmp_repo: CapturedTextRepository,
) -> None:
    tmp_repo.save(_make_record(text="first"))
    tmp_repo.save(_make_record(text="second"))

    loaded = tmp_repo.get("event_1")
    assert loaded is not None
    assert tmp_repo.count() == 1
    assert loaded.text == "second"


def test_repo_get_recent_returns_latest_first(
    tmp_repo: CapturedTextRepository,
) -> None:
    now = datetime.now(tz=timezone.utc)
    tmp_repo.save(_make_record(event_id="old", occurred_at=now - timedelta(minutes=5)))
    tmp_repo.save(_make_record(event_id="new", occurred_at=now))

    recent = tmp_repo.get_recent(limit=1)

    assert len(recent) == 1
    assert recent[0].event_id == "new"


def test_captured_text_record_from_payload_preserves_source_metadata() -> None:
    occurred_at = datetime(2026, 6, 6, 1, 0, tzinfo=timezone.utc)
    payload = CapturedTextEventPayload(
        event_id="event_payload",
        occurred_at=occurred_at,
        text="reddit text",
        locale="en",
        source_type=CapturedTextSourceType.REDDIT,
        surface_type=CapturedTextSurfaceType.REDDIT_COMMENT,
        page_url="https://reddit.test/r/example/comments/1",
        page_title="Example Reddit thread",
        collector_version="collector@1",
        metadata={"comment_id": "c1"},
    )

    record = captured_text_record_from_payload(payload, received_at=occurred_at)

    assert record.event_id == "event_payload"
    assert record.source_type == "reddit"
    assert record.surface_type == "reddit_comment"
    assert record.page_title == "Example Reddit thread"
    assert record.metadata["comment_id"] == "c1"
