"""QueryBufferRepository 단위 테스트."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent.src.infrastructure.repositories.query_buffer_repository import (
    QueryBufferRecord,
    QueryBufferRepository,
    build_query_buffer_record,
)
from shared.src.domain.entities.inference.events import QueryEvent, ScoredEvent


@pytest.fixture
def tmp_repo(tmp_path: Path) -> QueryBufferRepository:
    """임시 SQLite 경로를 쓰는 query buffer 저장소."""
    return QueryBufferRepository(db_path=tmp_path / "query_buffer.db")


def _make_record(
    *,
    query_id: str = "q1",
    occurred_at: datetime | None = None,
    raw_text: str = "불안해",
) -> QueryBufferRecord:
    return QueryBufferRecord(
        query_id=query_id,
        occurred_at=occurred_at or datetime.now(tz=timezone.utc),
        raw_text=raw_text,
        locale="ko",
        source_type="manual",
        model_revision="seed_rev_001",
        predicted_label="anxiety",
        confidence=0.87,
        margin=0.42,
        runner_up_label="depression",
        runner_up_score=0.45,
        confidence_kind="prototype_similarity_top1",
        metadata={"source": "unit-test"},
    )


def test_repo_save_and_get_round_trips_record(
    tmp_repo: QueryBufferRepository,
) -> None:
    """저장한 query buffer 레코드를 query_id로 다시 읽을 수 있다."""
    record = _make_record()

    tmp_repo.save(record)

    loaded = tmp_repo.get("q1")
    assert loaded is not None
    assert loaded.query_id == "q1"
    assert loaded.raw_text == "불안해"
    assert loaded.predicted_label == "anxiety"
    assert loaded.metadata["source"] == "unit-test"


def test_repo_save_overwrites_same_query_id(
    tmp_repo: QueryBufferRepository,
) -> None:
    """같은 query_id로 저장하면 최신 snapshot으로 교체된다."""
    tmp_repo.save(_make_record(raw_text="first"))
    tmp_repo.save(_make_record(raw_text="second"))

    loaded = tmp_repo.get("q1")
    assert loaded is not None
    assert tmp_repo.count() == 1
    assert loaded.raw_text == "second"


def test_repo_get_recent_returns_latest_first(
    tmp_repo: QueryBufferRepository,
) -> None:
    """recent 조회는 최신 occurred_at 순서와 limit를 지킨다."""
    now = datetime.now(tz=timezone.utc)
    tmp_repo.save(
        _make_record(query_id="old", occurred_at=now - timedelta(minutes=5))
    )
    tmp_repo.save(
        _make_record(query_id="new", occurred_at=now)
    )

    recent = tmp_repo.get_recent(limit=1)

    assert len(recent) == 1
    assert recent[0].query_id == "new"


def test_repo_delete_older_than_removes_old_records(
    tmp_repo: QueryBufferRepository,
) -> None:
    """cutoff 이전 레코드는 purge 대상이다."""
    now = datetime.now(tz=timezone.utc)
    tmp_repo.save(
        _make_record(query_id="old", occurred_at=now - timedelta(days=31))
    )
    tmp_repo.save(_make_record(query_id="new", occurred_at=now))

    deleted = tmp_repo.delete_older_than(cutoff=now - timedelta(days=30))

    assert deleted == 1
    assert tmp_repo.get("old") is None
    assert tmp_repo.get("new") is not None


def test_repo_delete_oldest_excess_keeps_latest_records(
    tmp_repo: QueryBufferRepository,
) -> None:
    """capacity purge는 최신 keep_latest개만 남긴다."""
    now = datetime.now(tz=timezone.utc)
    tmp_repo.save(_make_record(query_id="q1", occurred_at=now - timedelta(minutes=2)))
    tmp_repo.save(_make_record(query_id="q2", occurred_at=now - timedelta(minutes=1)))
    tmp_repo.save(_make_record(query_id="q3", occurred_at=now))

    deleted = tmp_repo.delete_oldest_excess(keep_latest=2)

    assert deleted == 1
    assert tmp_repo.get("q1") is None
    assert tmp_repo.get("q2") is not None
    assert tmp_repo.get("q3") is not None


def test_build_query_buffer_record_uses_query_text_and_score_snapshot() -> None:
    """helper가 raw text와 top1/top2 snapshot을 올바르게 추출한다."""
    occurred_at = datetime.now(tz=timezone.utc)
    event = QueryEvent(
        query_id="q_build",
        text="I feel anxious",
        occurred_at=occurred_at,
        locale="en",
        source_type="manual",
    )
    scored_event = ScoredEvent(
        query_id="q_build",
        occurred_at=occurred_at,
        translated_text=None,
        embedding_model_id="embed_test",
        translation_model_id=None,
        category_scores={"depression": 0.3, "anxiety": 0.9, "normal": 0.1},
    )

    record = build_query_buffer_record(
        event=event,
        scored_event=scored_event,
        model_revision="seed_rev_001",
        confidence_kind="prototype_similarity_top1",
        metadata={"was_translated": False},
    )

    assert record.query_id == "q_build"
    assert record.raw_text == "I feel anxious"
    assert record.predicted_label == "anxiety"
    assert record.confidence == pytest.approx(0.9)
    assert record.runner_up_label == "depression"
    assert record.runner_up_score == pytest.approx(0.3)
    assert record.margin == pytest.approx(0.6)
    assert record.metadata["was_translated"] is False
