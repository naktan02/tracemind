"""Query buffer lifecycle service unit tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent.src.infrastructure.repositories.query_buffer_repository import (
    QueryBufferRecord,
    QueryBufferRepository,
)
from agent.src.services.training.selection.query_buffer_lifecycle_service import (
    QueryBufferLifecycleConfig,
    QueryBufferLifecycleService,
)


def _make_record(
    *,
    query_id: str,
    occurred_at: datetime,
) -> QueryBufferRecord:
    return QueryBufferRecord(
        query_id=query_id,
        occurred_at=occurred_at,
        raw_text=f"text:{query_id}",
        locale="ko",
        source_type="manual",
        model_revision="seed_rev_001",
        predicted_label="anxiety",
        confidence=0.87,
        margin=0.42,
        runner_up_label="depression",
        runner_up_score=0.45,
        confidence_kind="classifier_head_logit_top1",
        metadata={},
    )


def test_lifecycle_service_purges_by_retention_and_capacity(
    tmp_path: Path,
) -> None:
    repo = QueryBufferRepository(db_path=tmp_path / "query_buffer.db")
    now = datetime(2026, 4, 12, tzinfo=timezone.utc)
    repo.save(_make_record(query_id="old", occurred_at=now - timedelta(days=40)))
    repo.save(_make_record(query_id="mid", occurred_at=now - timedelta(days=1)))
    repo.save(_make_record(query_id="new", occurred_at=now))

    result = QueryBufferLifecycleService(
        config=QueryBufferLifecycleConfig(
            retention_days=30,
            max_records=1,
        )
    ).purge(repository=repo, as_of=now)

    assert result.deleted_by_retention == 1
    assert result.deleted_by_capacity == 1
    assert result.deleted_total == 2
    assert repo.get("old") is None
    assert repo.get("mid") is None
    assert repo.get("new") is not None
