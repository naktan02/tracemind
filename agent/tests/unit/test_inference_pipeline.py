"""AnalysisEventRepository와 InferencePipelineService 단위 테스트."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.src.features.inference.pipeline_factory import (
    AGENT_SCORING_BACKEND_ENV,
    build_default_pipeline_service,
)
from agent.src.features.inference.pipeline_service import (
    InferencePipelineService,
)
from agent.src.infrastructure.repositories.analysis_event_repository import (
    AnalysisEventRepository,
)
from agent.src.services.language.preprocess_service import PreprocessService
from shared.src.contracts.model_contracts import make_embedding_manifest
from shared.src.domain.entities.inference.events import AnalysisEvent, QueryEvent

# ---------------------------------------------------------- #
# 픽스처                                                        #
# ---------------------------------------------------------- #


@pytest.fixture
def tmp_repo(tmp_path: Path) -> AnalysisEventRepository:
    """임시 경로에 SQLite DB를 만든 리포지토리."""
    return AnalysisEventRepository(db_path=tmp_path / "test_events.db")


def _make_analysis_event(
    query_id: str = "q1",
    occurred_at: datetime | None = None,
    translated_text: str | None = "feeling anxious",
    category_scores: dict | None = None,
) -> AnalysisEvent:
    return AnalysisEvent(
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


def _get_analysis_metadata_row(
    pipeline: InferencePipelineService,
    analysis_id: str,
) -> dict[str, object]:
    with sqlite3.connect(pipeline.event_repository.db_path) as conn:
        row = conn.execute(
            """
            SELECT model_revision, metadata
            FROM analysis_events
            WHERE analysis_id = ?
            """,
            (analysis_id,),
        ).fetchone()
    assert row is not None
    return {
        "model_revision": str(row[0]),
        "metadata": json.loads(str(row[1])),
    }


# ---------------------------------------------------------- #
# AnalysisEventRepository 테스트                                  #
# ---------------------------------------------------------- #


def test_repo_save_and_get_recent(tmp_repo: AnalysisEventRepository) -> None:
    """저장한 이벤트가 get_recent로 조회된다."""
    event = _make_analysis_event()
    tmp_repo.save(event)
    events = tmp_repo.get_recent(days=7)
    assert len(events) == 1
    assert events[0].query_id == "q1"
    assert events[0].category_scores == {"anxiety": 0.9, "depression": 0.4}


def test_repo_uses_analysis_event_schema(tmp_repo: AnalysisEventRepository) -> None:
    """저장소가 method-agnostic analysis schema를 생성한다."""
    with sqlite3.connect(tmp_repo.db_path) as conn:
        table_names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        analysis_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(analysis_events)")
        }
        score_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(analysis_category_scores)")
        }

    assert "analysis_events" in table_names
    assert "analysis_category_scores" in table_names
    assert {
        "analysis_id",
        "source_event_id",
        "scorer_family",
        "scorer_name",
        "model_revision",
        "metadata",
    }.issubset(analysis_columns)
    assert "confidence_kind" not in analysis_columns
    assert {"analysis_id", "category", "score"}.issubset(score_columns)


def test_repo_drops_legacy_confidence_kind_column(tmp_path: Path) -> None:
    """기존 analysis_events.confidence_kind 컬럼을 제거한다."""
    db_path = tmp_path / "legacy_events.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE analysis_events (
                analysis_id          TEXT PRIMARY KEY,
                source_event_id      TEXT NOT NULL,
                occurred_at          TEXT NOT NULL,
                translated_text      TEXT,
                scorer_family        TEXT NOT NULL,
                scorer_name          TEXT NOT NULL,
                model_revision       TEXT NOT NULL,
                top_category         TEXT,
                top_score            REAL,
                confidence_kind      TEXT NOT NULL,
                embedding_model_id   TEXT,
                translation_model_id TEXT,
                base_embedding       TEXT,
                metadata             TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE analysis_category_scores (
                analysis_id TEXT NOT NULL,
                category    TEXT NOT NULL,
                score       REAL NOT NULL,
                PRIMARY KEY (analysis_id, category),
                FOREIGN KEY (analysis_id)
                    REFERENCES analysis_events (analysis_id)
                    ON DELETE CASCADE
            );
            """
        )
        conn.execute(
            """
            INSERT INTO analysis_events
                (analysis_id, source_event_id, occurred_at, translated_text,
                 scorer_family, scorer_name, model_revision, top_category, top_score,
                 confidence_kind, embedding_model_id, translation_model_id,
                 base_embedding, metadata)
            VALUES (
                'q1', 'q1', '2026-03-29T00:00:00+00:00', NULL,
                'classifier', 'classifier_head_logits', 'rev_001',
                'anxiety', 0.9, 'legacy_top1', 'embed_001', NULL,
                NULL, '{}'
            );
            """
        )
        conn.execute(
            """
            INSERT INTO analysis_category_scores (analysis_id, category, score)
            VALUES ('q1', 'anxiety', 0.9);
            """
        )

    repo = AnalysisEventRepository(db_path=db_path)

    with sqlite3.connect(repo.db_path) as conn:
        analysis_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(analysis_events)")
        }
        score_rows = conn.execute(
            """
            SELECT analysis_id, category, score
            FROM analysis_category_scores
            """
        ).fetchall()

    assert "confidence_kind" not in analysis_columns
    assert score_rows == [("q1", "anxiety", 0.9)]


def test_repo_get_recent_filters_old_events(tmp_repo: AnalysisEventRepository) -> None:
    """기간 밖의 이벤트는 get_recent에서 제외된다."""
    old_event = _make_analysis_event(
        query_id="old",
        occurred_at=datetime.now(tz=timezone.utc) - timedelta(days=30),
    )
    recent_event = _make_analysis_event(query_id="new")
    tmp_repo.save(old_event)
    tmp_repo.save(recent_event)
    events = tmp_repo.get_recent(days=7)
    assert len(events) == 1
    assert events[0].query_id == "new"


def test_default_pipeline_requires_explicit_scoring_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_repo: AnalysisEventRepository,
) -> None:
    monkeypatch.delenv(AGENT_SCORING_BACKEND_ENV, raising=False)

    with pytest.raises(ValueError, match=AGENT_SCORING_BACKEND_ENV):
        build_default_pipeline_service(
            analysis_event_repository=tmp_repo,
            shared_adapter_runtime_service=MagicMock(),
            translation_service=None,
        )


def test_default_pipeline_uses_configured_scoring_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_repo: AnalysisEventRepository,
) -> None:
    class _FakeEmbeddingAdapter:
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] for _text in texts]

    monkeypatch.setenv(AGENT_SCORING_BACKEND_ENV, "classifier_head_logits")
    monkeypatch.setattr(
        "agent.src.features.inference.pipeline_factory.EmbeddingAdapterFactory.create",
        lambda _spec: _FakeEmbeddingAdapter(),
    )

    pipeline = build_default_pipeline_service(
        analysis_event_repository=tmp_repo,
        shared_adapter_runtime_service=MagicMock(),
        translation_service=None,
    )

    assert pipeline.scoring_service.backend_name == "classifier_head_logits"


def test_repo_save_overwrites_same_query_id(tmp_repo: AnalysisEventRepository) -> None:
    """같은 query_id로 저장하면 기존 데이터를 덮어쓴다."""
    tmp_repo.save(_make_analysis_event(query_id="q1", translated_text="first"))
    tmp_repo.save(_make_analysis_event(query_id="q1", translated_text="second"))
    events = tmp_repo.get_recent(days=7)
    assert len(events) == 1
    assert events[0].translated_text == "second"


def test_repo_count_returns_stored_count(tmp_repo: AnalysisEventRepository) -> None:
    """count()가 저장된 이벤트 수를 반환한다."""
    assert tmp_repo.count() == 0
    tmp_repo.save(_make_analysis_event(query_id="a"))
    tmp_repo.save(_make_analysis_event(query_id="b"))
    assert tmp_repo.count() == 2


def test_repo_event_without_translation(tmp_repo: AnalysisEventRepository) -> None:
    """translated_text=None인 이벤트가 올바르게 저장·복원된다."""
    event = _make_analysis_event(translated_text=None)
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
    with_scoring_asset_provider: bool = True,
    shared_adapter_provider=None,
    local_adapter_provider=None,
) -> InferencePipelineService:
    """모킹된 서비스들로 파이프라인을 조립한다."""
    embedding_service = MagicMock()
    embedding_service.embed_batch.return_value = [[0.1, 0.2, 0.3]]

    scoring_service = MagicMock()
    scoring_service.score.return_value = {"anxiety": 0.85, "depression": 0.25}
    scoring_service.backend_name = "classifier_head_logits"

    scoring_asset_provider = None
    if with_scoring_asset_provider:
        scoring_asset_provider = MagicMock()
        scoring_asset_provider.get_scoring_assets.return_value = {
            "anxiety": ([0.1, 0.2, 0.3],)
        }

    repo = AnalysisEventRepository(db_path=tmp_path / "events.db")
    translation_service = None
    if with_translation:
        translation_service = MagicMock()
        translation_service.translate_batch.return_value = ["I feel anxious"]

    return InferencePipelineService(
        embedding_service=embedding_service,
        scoring_service=scoring_service,
        event_repository=repo,
        scoring_asset_provider=scoring_asset_provider,
        shared_adapter_provider=shared_adapter_provider,
        local_adapter_provider=local_adapter_provider,
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
    assert result.analysis_event.translated_text is None
    pipeline.translation_service.translate_batch.assert_not_called()


def test_pipeline_can_score_without_scoring_asset_provider(tmp_path: Path) -> None:
    """classifier 계열처럼 asset 없이 점수를 내는 scorer를 허용한다."""
    pipeline = _make_pipeline(tmp_path, with_scoring_asset_provider=False)
    pipeline.scoring_service.backend_name = "classifier_head_logits"

    event = _make_query_event(text="I feel anxious", locale="en")
    result = pipeline.process(event)

    assert result.analysis_event.category_scores == {
        "anxiety": 0.85,
        "depression": 0.25,
    }
    pipeline.scoring_service.score.assert_called_once_with(
        [0.1, 0.2, 0.3],
        {},
        shared_state=None,
    )


def test_pipeline_translates_korean_event(tmp_path: Path) -> None:
    """한국어 이벤트는 번역 서비스를 거친다."""
    pipeline = _make_pipeline(tmp_path, with_translation=True)
    event = _make_query_event(text="불안해", locale="ko")
    result = pipeline.process(event)
    assert result.was_translated is True
    assert result.analysis_event.translated_text == "I feel anxious"
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
    """ScoringService 결과가 AnalysisEvent에 올바르게 반영된다."""
    pipeline = _make_pipeline(tmp_path)
    event = _make_query_event(locale="en")
    result = pipeline.process(event)
    assert result.analysis_event.category_scores == {
        "anxiety": 0.85,
        "depression": 0.25,
    }


def test_pipeline_stores_analysis_event_metadata(tmp_path: Path) -> None:
    """AnalysisEvent 저장 뒤 inference metadata를 남긴다."""
    pipeline = _make_pipeline(tmp_path)
    event = _make_query_event(text="I feel anxious", locale="en")

    result = pipeline.process(event)

    stored = pipeline.event_repository.get_recent_stored(days=1)
    assert stored[0].analysis_event.query_id == result.analysis_event.query_id
    row = _get_analysis_metadata_row(pipeline, event.query_id)
    assert row["model_revision"] == "seed_rev_001"
    assert row["metadata"]["embedding_model_id"] == "test-embed"
    assert row["metadata"]["was_translated"] is False


def test_pipeline_uses_active_shared_adapter_state_for_scoring(
    tmp_path: Path,
) -> None:
    """active shared adapter state가 있으면 변환 embedding과 revision을 사용한다."""
    shared_state = MagicMock()
    shared_state.adapter_kind = "peft_classifier"
    shared_state.model_revision = "global_rev_001"
    shared_state.apply.side_effect = lambda embedding: [float(v) for v in embedding]
    shared_adapter_provider = MagicMock()
    shared_adapter_provider.get_active_state.return_value = shared_state
    shared_adapter_provider.get_active_manifest.return_value = make_embedding_manifest(
        model_id="test-embed",
        model_revision="global_rev_001",
        auxiliary_artifact_versions={"calibration_set": "calib_001"},
        artifact_ref="shared_adapter_state::global_rev_001",
    )
    pipeline = _make_pipeline(
        tmp_path,
        shared_adapter_provider=shared_adapter_provider,
    )
    event = _make_query_event(text="I feel anxious", locale="en")

    pipeline.process(event)

    pipeline.scoring_service.score.assert_called_once()
    _, kwargs = pipeline.scoring_service.score.call_args
    assert kwargs["shared_state"].model_revision == "global_rev_001"
    row = _get_analysis_metadata_row(pipeline, event.query_id)
    assert row["model_revision"] == "global_rev_001"
    assert row["metadata"]["shared_model_revision"] == "global_rev_001"
    assert row["metadata"]["adapter_kind"] == "peft_classifier"


def test_pipeline_can_apply_future_local_adapter_after_shared_state(
    tmp_path: Path,
) -> None:
    """local state가 있으면 inference에만 shared 이후 순서로 적용한다."""
    shared_state = MagicMock()
    shared_state.adapter_kind = "peft_classifier"
    shared_state.model_revision = "global_rev_001"
    shared_state.apply.side_effect = lambda embedding: [float(v) for v in embedding]
    shared_adapter_provider = MagicMock()
    shared_adapter_provider.get_active_state.return_value = shared_state
    shared_adapter_provider.get_active_manifest.return_value = make_embedding_manifest(
        model_id="test-embed",
        model_revision="global_rev_001",
        auxiliary_artifact_versions={"calibration_set": "calib_001"},
        artifact_ref="shared_adapter_state::global_rev_001",
    )
    local_state = MagicMock()
    local_state.adapter_kind = "local_fake"
    local_state.model_revision = "local_rev_001"
    local_state.apply.return_value = [0.9, 0.8, 0.7]
    local_adapter_provider = MagicMock()
    local_adapter_provider.get_active_state.return_value = local_state
    pipeline = _make_pipeline(
        tmp_path,
        shared_adapter_provider=shared_adapter_provider,
        local_adapter_provider=local_adapter_provider,
    )
    event = _make_query_event(text="I feel anxious", locale="en")

    pipeline.process(event)

    local_state.apply.assert_called_once()
    score_embedding = pipeline.scoring_service.score.call_args.args[0]
    assert score_embedding == [0.9, 0.8, 0.7]
    row = _get_analysis_metadata_row(pipeline, event.query_id)
    assert row["model_revision"] == "global_rev_001"
    assert row["metadata"]["local_adapter_revision"] == "local_rev_001"
    assert row["metadata"]["local_adapter_kind"] == "local_fake"


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


# ---------------------------------------------------------- #
# base_embedding 저장·복원 테스트                               #
# ---------------------------------------------------------- #


def test_repo_saves_and_restores_base_embedding(
    tmp_repo: AnalysisEventRepository,
) -> None:
    """base_embedding을 함께 저장하면 get_recent_stored로 복원된다."""
    event = _make_analysis_event()
    embedding = [0.1, 0.2, 0.3]
    tmp_repo.save(event, base_embedding=embedding)
    stored = tmp_repo.get_recent_stored(days=7)
    assert len(stored) == 1
    assert stored[0].base_embedding == pytest.approx(embedding)


def test_repo_get_recent_stored_returns_none_when_no_embedding(
    tmp_repo: AnalysisEventRepository,
) -> None:
    """base_embedding 없이 저장한 이벤트는 get_recent_stored에서 None으로 반환된다."""
    tmp_repo.save(_make_analysis_event())
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


def test_pipeline_stores_scorer_metadata_in_analysis_event(tmp_path: Path) -> None:
    """파이프라인이 analysis event에 scorer metadata를 저장한다."""
    pipeline = _make_pipeline(tmp_path)
    event = _make_query_event(text="I feel anxious", locale="en")
    pipeline.process(event)

    with sqlite3.connect(pipeline.event_repository.db_path) as conn:
        row = conn.execute(
            """
            SELECT scorer_family, scorer_name, model_revision
            FROM analysis_events
            WHERE analysis_id = ?
            """,
            (event.query_id,),
        ).fetchone()

    assert row == (
        "classifier",
        "classifier_head_logits",
        "seed_rev_001",
    )
