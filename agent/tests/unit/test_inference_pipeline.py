"""ScoredEventRepository와 InferencePipelineService 단위 테스트."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.src.infrastructure.repositories.query_buffer_repository import (
    QueryBufferRepository,
)
from agent.src.infrastructure.repositories.scored_event_repository import (
    ScoredEventRepository,
)
from agent.src.services.inference.pipeline_service import (
    InferencePipelineService,
)
from agent.src.services.preprocess_service import PreprocessService
from shared.src.domain.entities.inference.events import QueryEvent, ScoredEvent


# ---------------------------------------------------------- #
# 픽스처                                                        #
# ---------------------------------------------------------- #


@pytest.fixture
def tmp_repo(tmp_path: Path) -> ScoredEventRepository:
    """임시 경로에 SQLite DB를 만든 리포지토리."""
    return ScoredEventRepository(db_path=tmp_path / "test_events.db")


def _make_scored_event(
    query_id: str = "q1",
    occurred_at: datetime | None = None,
    translated_text: str | None = "feeling anxious",
    category_scores: dict | None = None,
) -> ScoredEvent:
    return ScoredEvent(
        query_id=query_id,
        occurred_at=occurred_at or datetime.now(tz=timezone.utc),
        translated_text=translated_text,
        embedding_model_id="test-model",
        translation_model_id=None,
        category_scores=category_scores or {"anxiety": 0.9, "depression": 0.4},
    )


def _make_query_event(
    text: str = "불안해",
    locale: str = "ko",
) -> QueryEvent:
    return QueryEvent(
        query_id="q1",
        text=text,
        occurred_at=datetime.now(tz=timezone.utc),
        locale=locale,
        source_type="test",
    )


# ---------------------------------------------------------- #
# ScoredEventRepository 테스트                                  #
# ---------------------------------------------------------- #


def test_repo_save_and_get_recent(tmp_repo: ScoredEventRepository) -> None:
    """저장한 이벤트가 get_recent로 조회된다."""
    event = _make_scored_event()
    tmp_repo.save(event)
    events = tmp_repo.get_recent(days=7)
    assert len(events) == 1
    assert events[0].query_id == "q1"
    assert events[0].category_scores == {"anxiety": 0.9, "depression": 0.4}


def test_repo_get_recent_filters_old_events(tmp_repo: ScoredEventRepository) -> None:
    """기간 밖의 이벤트는 get_recent에서 제외된다."""
    old_event = _make_scored_event(
        query_id="old",
        occurred_at=datetime.now(tz=timezone.utc) - timedelta(days=30),
    )
    recent_event = _make_scored_event(query_id="new")
    tmp_repo.save(old_event)
    tmp_repo.save(recent_event)
    events = tmp_repo.get_recent(days=7)
    assert len(events) == 1
    assert events[0].query_id == "new"


def test_repo_save_overwrites_same_query_id(tmp_repo: ScoredEventRepository) -> None:
    """같은 query_id로 저장하면 기존 데이터를 덮어쓴다."""
    tmp_repo.save(_make_scored_event(query_id="q1", translated_text="first"))
    tmp_repo.save(_make_scored_event(query_id="q1", translated_text="second"))
    events = tmp_repo.get_recent(days=7)
    assert len(events) == 1
    assert events[0].translated_text == "second"


def test_repo_count_returns_stored_count(tmp_repo: ScoredEventRepository) -> None:
    """count()가 저장된 이벤트 수를 반환한다."""
    assert tmp_repo.count() == 0
    tmp_repo.save(_make_scored_event(query_id="a"))
    tmp_repo.save(_make_scored_event(query_id="b"))
    assert tmp_repo.count() == 2


def test_repo_event_without_translation(tmp_repo: ScoredEventRepository) -> None:
    """translated_text=None인 이벤트가 올바르게 저장·복원된다."""
    event = _make_scored_event(translated_text=None)
    tmp_repo.save(event)
    events = tmp_repo.get_recent(days=1)
    assert events[0].translated_text is None


# ---------------------------------------------------------- #
# InferencePipelineService 테스트                               #
# ---------------------------------------------------------- #


def _make_pipeline(
    tmp_path: Path,
    *,
    with_translation: bool = False,
) -> InferencePipelineService:
    """모킹된 서비스들로 파이프라인을 조립한다."""
    embedding_service = MagicMock()
    embedding_service.embed_batch.return_value = [[0.1, 0.2, 0.3]]

    scoring_service = MagicMock()
    scoring_service.score.return_value = {"anxiety": 0.85, "depression": 0.25}
    scoring_service.backend = MagicMock()
    scoring_service.backend.backend_name = "prototype_similarity"

    prototype_provider = MagicMock()
    prototype_provider.get_active_prototypes.return_value = {
        "anxiety": ([0.1, 0.2, 0.3],)
    }

    repo = ScoredEventRepository(db_path=tmp_path / "events.db")
    query_buffer_repo = QueryBufferRepository(db_path=tmp_path / "query_buffer.db")

    translation_service = None
    if with_translation:
        translation_service = MagicMock()
        translation_service.translate_batch.return_value = ["I feel anxious"]

    return InferencePipelineService(
        embedding_service=embedding_service,
        scoring_service=scoring_service,
        prototype_provider=prototype_provider,
        event_repository=repo,
        query_buffer_repository=query_buffer_repo,
        preprocess_service=PreprocessService(),
        translation_service=translation_service,
        translation_locales=frozenset({"ko", "ja"}),
        embedding_model_id="test-embed",
        model_revision="seed_rev_001",
    )


def test_pipeline_processes_english_without_translation(tmp_path: Path) -> None:
    """영어 이벤트는 번역 없이 임베딩→스코어링을 거쳐 저장된다."""
    pipeline = _make_pipeline(tmp_path, with_translation=True)
    event = _make_query_event(text="I feel anxious", locale="en")
    result = pipeline.process(event)
    assert result.was_translated is False
    assert result.scored_event.translated_text is None
    pipeline.translation_service.translate_batch.assert_not_called()


def test_pipeline_translates_korean_event(tmp_path: Path) -> None:
    """한국어 이벤트는 번역 서비스를 거친다."""
    pipeline = _make_pipeline(tmp_path, with_translation=True)
    event = _make_query_event(text="불안해", locale="ko")
    result = pipeline.process(event)
    assert result.was_translated is True
    assert result.scored_event.translated_text == "I feel anxious"
    pipeline.translation_service.translate_batch.assert_called_once()


def test_pipeline_skips_translation_when_service_is_none(tmp_path: Path) -> None:
    """translation_service=None이면 한국어도 번역 없이 처리된다."""
    pipeline = _make_pipeline(tmp_path, with_translation=False)
    event = _make_query_event(text="불안해", locale="ko")
    result = pipeline.process(event)
    assert result.was_translated is False


def test_pipeline_stores_event_in_repository(tmp_path: Path) -> None:
    """처리 완료된 이벤트가 저장소에 저장된다."""
    pipeline = _make_pipeline(tmp_path, with_translation=False)
    event = _make_query_event(text="I feel anxious", locale="en")
    pipeline.process(event)
    events = pipeline.event_repository.get_recent(days=1)
    assert len(events) == 1
    assert events[0].query_id == event.query_id


def test_pipeline_returns_correct_category_scores(tmp_path: Path) -> None:
    """ScoringService 결과가 ScoredEvent에 올바르게 반영된다."""
    pipeline = _make_pipeline(tmp_path)
    event = _make_query_event(locale="en")
    result = pipeline.process(event)
    assert result.scored_event.category_scores == {
        "anxiety": 0.85,
        "depression": 0.25,
    }


def test_pipeline_stores_query_buffer_record_with_same_query_id(tmp_path: Path) -> None:
    """ScoredEvent 저장 뒤 같은 query_id의 query buffer snapshot을 남긴다."""
    pipeline = _make_pipeline(tmp_path)
    event = _make_query_event(text="I feel anxious", locale="en")

    result = pipeline.process(event)

    assert result.query_buffer_record is not None
    assert result.query_buffer_record.query_id == event.query_id
    assert result.query_buffer_record.model_revision == "seed_rev_001"
    assert result.query_buffer_record.predicted_label == "anxiety"
    assert result.query_buffer_record.runner_up_label == "depression"
    assert result.query_buffer_record.confidence == pytest.approx(0.85)
    assert result.query_buffer_record.margin == pytest.approx(0.6)

    assert pipeline.query_buffer_repository is not None
    stored_record = pipeline.query_buffer_repository.get(event.query_id)
    assert stored_record is not None
    assert stored_record.query_id == result.scored_event.query_id
    assert stored_record.raw_text == "I feel anxious"
    assert stored_record.confidence_kind == "prototype_similarity_top1"
    assert stored_record.metadata["embedding_model_id"] == "test-embed"
    assert stored_record.metadata["was_translated"] is False


def test_pipeline_batch_processes_multiple_events(tmp_path: Path) -> None:
    """process_batch가 여러 이벤트를 순서대로 처리한다."""
    pipeline = _make_pipeline(tmp_path)
    events = [
        QueryEvent(
            query_id=f"q{i}",
            text="I feel anxious",
            occurred_at=datetime.now(tz=timezone.utc),
            locale="en",
            source_type="test",
        )
        for i in range(3)
    ]
    results = pipeline.process_batch(events)
    assert len(results) == 3
    assert pipeline.event_repository.count() == 3
    assert pipeline.query_buffer_repository is not None
    assert pipeline.query_buffer_repository.count() == 3


# ---------------------------------------------------------- #
# base_embedding 저장·복원 테스트                               #
# ---------------------------------------------------------- #


def test_repo_saves_and_restores_base_embedding(tmp_repo: ScoredEventRepository) -> None:
    """base_embedding을 함께 저장하면 get_recent_stored로 복원된다."""
    event = _make_scored_event()
    embedding = [0.1, 0.2, 0.3]
    tmp_repo.save(event, base_embedding=embedding)
    stored = tmp_repo.get_recent_stored(days=7)
    assert len(stored) == 1
    assert stored[0].base_embedding == pytest.approx(embedding)


def test_repo_get_recent_stored_returns_none_when_no_embedding(
    tmp_repo: ScoredEventRepository,
) -> None:
    """base_embedding 없이 저장한 이벤트는 get_recent_stored에서 None으로 반환된다."""
    tmp_repo.save(_make_scored_event())
    stored = tmp_repo.get_recent_stored(days=7)
    assert stored[0].base_embedding is None


def test_pipeline_stores_base_embedding_in_repository(tmp_path: Path) -> None:
    """파이프라인이 임베딩 결과를 저장소에 함께 저장한다."""
    pipeline = _make_pipeline(tmp_path)
    event = _make_query_event(text="I feel anxious", locale="en")
    result = pipeline.process(event)
    # 결과 객체에 base_embedding이 노출되는지 확인
    assert result.base_embedding == pytest.approx([0.1, 0.2, 0.3])
    # 저장소에도 embedding이 저장됐는지 확인
    stored = pipeline.event_repository.get_recent_stored(days=1)
    assert stored[0].base_embedding == pytest.approx([0.1, 0.2, 0.3])
