"""CapturedTextRepository 단위 테스트."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent.src.contracts.captured_text_contracts import (
    CapturedTextEventPayload,
    CapturedTextSourceType,
    CapturedTextSurfaceType,
)
from agent.src.infrastructure.repositories.captured_text_repository import (
    CAPTURED_TEXT_VIEW_STATUS_DUPLICATE,
    CAPTURED_TEXT_VIEW_STATUS_FAILED,
    CAPTURED_TEXT_VIEW_STATUS_PENDING,
    CAPTURED_TEXT_VIEW_STATUS_READY,
    CapturedTextGeneratedViewRecord,
    CapturedTextRecord,
    CapturedTextRepository,
    captured_text_record_from_payload,
)
from agent.src.services.ingest.captured_text_lifecycle_service import (
    CapturedTextLifecycleConfig,
    CapturedTextLifecycleService,
    build_captured_text_lifecycle_service_from_env,
)
from agent.src.services.ingest.captured_text_view_generation_service import (
    CapturedTextViewGenerationService,
)
from agent.src.services.training.datasets.captured_text_training_source_service import (
    CapturedTextTrainingSourceService,
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


def test_repo_initialization_migrates_existing_table(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy_captured_text.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE captured_text_events (
                event_id          TEXT PRIMARY KEY,
                schema_version    TEXT NOT NULL,
                occurred_at       TEXT NOT NULL,
                received_at       TEXT NOT NULL,
                text              TEXT NOT NULL,
                locale            TEXT NOT NULL,
                source_type       TEXT NOT NULL,
                surface_type      TEXT NOT NULL,
                page_url          TEXT,
                page_title        TEXT,
                collector_version TEXT,
                metadata          TEXT NOT NULL
            );
            """
        )

    repository = CapturedTextRepository(db_path=db_path)
    repository.save(_make_record())

    loaded = repository.get("event_1")
    assert loaded is not None
    assert loaded.view_generation_status == CAPTURED_TEXT_VIEW_STATUS_PENDING
    assert loaded.text_fingerprint


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


def test_repo_marks_same_source_surface_text_as_duplicate(
    tmp_repo: CapturedTextRepository,
) -> None:
    first = _make_record(event_id="first", text="  같은   검색어 ")
    second = _make_record(event_id="second", text="같은 검색어")

    tmp_repo.save(first)
    tmp_repo.save(second)

    loaded_first = tmp_repo.get("first")
    loaded_second = tmp_repo.get("second")
    assert loaded_first is not None
    assert loaded_second is not None
    assert loaded_first.view_generation_status == CAPTURED_TEXT_VIEW_STATUS_PENDING
    assert loaded_second.view_generation_status == CAPTURED_TEXT_VIEW_STATUS_DUPLICATE
    assert loaded_second.duplicate_of_event_id == "first"
    assert loaded_second.metadata["duplicate_of_event_id"] == "first"


def test_repo_pending_view_generation_excludes_duplicates(
    tmp_repo: CapturedTextRepository,
) -> None:
    tmp_repo.save(_make_record(event_id="first", text="중복"))
    tmp_repo.save(_make_record(event_id="duplicate", text="중복"))
    tmp_repo.save(_make_record(event_id="ready", text="완료"))
    tmp_repo.mark_view_generation_status(
        event_id="ready",
        status=CAPTURED_TEXT_VIEW_STATUS_READY,
    )

    pending = tmp_repo.get_pending_view_generation(limit=10)
    counts = tmp_repo.count_by_view_generation_status()

    assert [record.event_id for record in pending] == ["first"]
    assert counts[CAPTURED_TEXT_VIEW_STATUS_PENDING] == 1
    assert counts[CAPTURED_TEXT_VIEW_STATUS_DUPLICATE] == 1
    assert counts[CAPTURED_TEXT_VIEW_STATUS_READY] == 1


def test_repo_saves_generated_view(
    tmp_repo: CapturedTextRepository,
) -> None:
    record = _make_record()
    tmp_repo.save(record)
    stored = tmp_repo.get("event_1")
    assert stored is not None

    tmp_repo.save_generated_view(
        CapturedTextGeneratedViewRecord(
            event_id=stored.event_id,
            generated_at=datetime.now(tz=timezone.utc),
            weak_text="I feel anxious",
            strong_text_0="I am feeling anxious",
            strong_text_1="I feel worried",
            generator_name="unit-test",
            generator_version="v1",
            source_text_fingerprint=stored.text_fingerprint,
            metadata={"source_locale": "ko"},
        )
    )

    generated = tmp_repo.get_generated_view("event_1")
    assert generated is not None
    assert generated.weak_text == "I feel anxious"
    assert tmp_repo.count_generated_views() == 1


def test_view_generation_service_materializes_identity_fallback(
    tmp_repo: CapturedTextRepository,
) -> None:
    tmp_repo.save(_make_record(event_id="event_1", text="불안해"))
    service = CapturedTextViewGenerationService(repository=tmp_repo)

    result = service.generate_pending_views(limit=10)

    generated = tmp_repo.get_generated_view("event_1")
    loaded = tmp_repo.get("event_1")
    assert generated is not None
    assert loaded is not None
    assert generated.weak_text == "불안해"
    assert generated.strong_text_0 == "불안해"
    assert generated.metadata["weak_text_provider"] == "identity"
    assert loaded.view_generation_status == CAPTURED_TEXT_VIEW_STATUS_READY
    assert result.generated_count == 1


def test_view_generation_service_regenerates_stale_ready_views(
    tmp_repo: CapturedTextRepository,
) -> None:
    class TranslationProvider:
        model_id = "translation-provider"

        def translate_batch(self, texts: list[str]) -> list[str]:
            return [f"translated:{text}" for text in texts]

    class StrongViewProvider:
        model_id = "strong-provider"

        def build_candidate_pairs(self, *, texts):
            class Pair:
                def __init__(self, text: str) -> None:
                    self.aug_0 = f"{text}:aug0"
                    self.aug_1 = f"{text}:aug1"

            return [Pair(str(text)) for text in texts]

    tmp_repo.save(_make_record(event_id="event_1", text="불안해"))
    identity_service = CapturedTextViewGenerationService(repository=tmp_repo)
    identity_service.generate_pending_views(limit=10)

    service = CapturedTextViewGenerationService(
        repository=tmp_repo,
        translation_provider=TranslationProvider(),
        strong_view_provider=StrongViewProvider(),
    )
    result = service.generate_pending_views(limit=10)

    generated = tmp_repo.get_generated_view("event_1")
    loaded = tmp_repo.get("event_1")
    assert generated is not None
    assert loaded is not None
    assert result.selected_count == 1
    assert result.generated_count == 1
    assert "stale generated view" in result.message
    assert generated.weak_text == "translated:불안해"
    assert generated.strong_text_0 == "translated:불안해:aug0"
    assert generated.metadata["weak_text_provider"] == "translation-provider"
    assert generated.metadata["strong_text_provider"] == "strong-provider"
    assert loaded.view_generation_status == CAPTURED_TEXT_VIEW_STATUS_READY


def test_generated_view_training_source_projection_uses_ready_views(
    tmp_repo: CapturedTextRepository,
) -> None:
    tmp_repo.save(_make_record(event_id="event_1", text="불안해"))
    service = CapturedTextViewGenerationService(repository=tmp_repo)
    service.generate_pending_views(limit=10)

    source_service = CapturedTextTrainingSourceService(repository=tmp_repo)
    source_rows = source_service.get_recent_source_rows(days=7, limit=10)

    assert len(source_rows) == 1
    assert source_rows[0].query_id == "event_1"
    assert source_rows[0].text == "불안해"
    assert source_rows[0].weak_text == "불안해"
    assert source_rows[0].strong_text == "불안해"


def test_generated_view_query_ssl_projection_uses_aug_candidates(
    tmp_repo: CapturedTextRepository,
) -> None:
    tmp_repo.save(_make_record(event_id="event_1", text="불안해"))
    service = CapturedTextViewGenerationService(repository=tmp_repo)
    service.generate_pending_views(limit=10)

    source_service = CapturedTextTrainingSourceService(repository=tmp_repo)
    rows = source_service.get_recent_query_ssl_unlabeled_rows(days=7, limit=10)

    assert len(rows) == 1
    assert rows[0]["query_id"] == "event_1"
    assert rows[0]["text"] == "불안해"
    assert rows[0]["aug_0"] == "불안해"
    assert rows[0]["aug_1"] == "불안해"
    assert rows[0]["annotation_source"] == "agent_local_unlabeled"


def test_view_generation_service_marks_provider_failure(
    tmp_repo: CapturedTextRepository,
) -> None:
    class FailingTranslationProvider:
        def translate_batch(self, texts: list[str]) -> list[str]:
            raise RuntimeError("translation failed")

    tmp_repo.save(_make_record(event_id="event_1", text="불안해"))
    service = CapturedTextViewGenerationService(
        repository=tmp_repo,
        translation_provider=FailingTranslationProvider(),
    )

    result = service.generate_pending_views(limit=10)

    loaded = tmp_repo.get("event_1")
    assert loaded is not None
    assert loaded.view_generation_status == CAPTURED_TEXT_VIEW_STATUS_FAILED
    assert result.failed_count == 1
    assert tmp_repo.count_generated_views() == 0


def test_repo_delete_older_than_and_capacity_purge(
    tmp_repo: CapturedTextRepository,
) -> None:
    now = datetime.now(tz=timezone.utc)
    tmp_repo.save(_make_record(event_id="old", occurred_at=now - timedelta(days=31)))
    tmp_repo.save(_make_record(event_id="mid", occurred_at=now - timedelta(days=1)))
    tmp_repo.save(_make_record(event_id="new", occurred_at=now))
    mid = tmp_repo.get("mid")
    assert mid is not None
    tmp_repo.save_generated_view(
        CapturedTextGeneratedViewRecord(
            event_id="mid",
            generated_at=now,
            weak_text="mid weak",
            strong_text_0="mid strong 0",
            strong_text_1="mid strong 1",
            generator_name="unit-test",
            generator_version="v1",
            source_text_fingerprint=mid.text_fingerprint,
        )
    )

    deleted_old = tmp_repo.delete_older_than(cutoff=now - timedelta(days=30))
    deleted_excess = tmp_repo.delete_oldest_excess(keep_latest=1)

    assert deleted_old == 1
    assert deleted_excess == 1
    assert tmp_repo.get("old") is None
    assert tmp_repo.get("mid") is None
    assert tmp_repo.get("new") is not None
    assert tmp_repo.get_generated_view("mid") is None


def test_captured_text_lifecycle_service_purges_by_policy(
    tmp_repo: CapturedTextRepository,
) -> None:
    now = datetime.now(tz=timezone.utc)
    tmp_repo.save(_make_record(event_id="old", occurred_at=now - timedelta(days=31)))
    tmp_repo.save(_make_record(event_id="mid", occurred_at=now - timedelta(days=1)))
    tmp_repo.save(_make_record(event_id="new", occurred_at=now))
    service = CapturedTextLifecycleService(
        config=CapturedTextLifecycleConfig(retention_days=30, max_records=1)
    )

    result = service.purge(repository=tmp_repo, as_of=now)

    assert result.deleted_by_retention == 1
    assert result.deleted_by_capacity == 1
    assert result.deleted_total == 2
    assert tmp_repo.get("new") is not None


def test_captured_text_lifecycle_default_is_short_for_development() -> None:
    service = build_captured_text_lifecycle_service_from_env(environ={})

    assert service.config.retention_days == 3
    assert service.config.max_records == 500


def test_captured_text_lifecycle_env_can_override_development_policy() -> None:
    service = build_captured_text_lifecycle_service_from_env(
        environ={
            "TRACEMIND_CAPTURED_TEXT_RETENTION_DAYS": "2",
            "TRACEMIND_CAPTURED_TEXT_MAX_RECORDS": "20",
        }
    )

    assert service.config.retention_days == 2
    assert service.config.max_records == 20


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
