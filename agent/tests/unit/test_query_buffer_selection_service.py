"""Query buffer projection / selection service unit tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from agent.src.infrastructure.repositories.query_buffer_repository import (
    build_query_buffer_record,
)
from agent.src.services.training.selection.query_buffer_projection import (
    QUERY_BUFFER_PROJECTION_BACKEND_NAME,
    build_query_buffer_evidence,
    build_query_buffer_evidences,
)
from agent.src.services.training.selection.query_buffer_selection_service import (
    QueryBufferSelectionService,
)
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
    TrainingTask,
)
from shared.src.domain.entities.inference.events import AnalysisEvent, QueryEvent


def _build_task(
    *,
    pseudo_label_algorithm_name: str | None = "top1_confidence_only",
    confidence_threshold: float = 0.8,
    margin_threshold: float = 0.02,
) -> TrainingTask:
    return TrainingTask(
        schema_version="training_task.v1",
        task_id="task_001",
        round_id="round_0001",
        model_id="tracemind-embed",
        model_revision="rev_001",
        task_type="pseudo_label_self_training",
        training_scope="adapter_only",
        local_epochs=1,
        batch_size=8,
        learning_rate=1e-2,
        max_steps=10,
        objective_config=TrainingObjectiveConfig(
            training_backend_name="peft_classifier_trainer",
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
            pseudo_label_algorithm_name=pseudo_label_algorithm_name,
        ),
        selection_policy=TrainingSelectionPolicy(max_examples=8),
    )


def _build_pair(
    *,
    query_id: str,
    text: str,
    category_scores: dict[str, float],
) -> tuple[QueryEvent, AnalysisEvent]:
    occurred_at = datetime(2026, 4, 12, 12, 0, tzinfo=timezone.utc)
    query_event = QueryEvent(
        query_id=query_id,
        text=text,
        occurred_at=occurred_at,
        locale="ko-KR",
        source_type="user_message",
    )
    analysis_event = AnalysisEvent(
        query_id=query_id,
        occurred_at=occurred_at,
        translated_text=None,
        embedding_model_id="tracemind-embed",
        translation_model_id=None,
        category_scores=category_scores,
    )
    return query_event, analysis_event


def test_build_query_buffer_evidence_preserves_snapshot_metadata() -> None:
    query_event, analysis_event = _build_pair(
        query_id="q1",
        text="숨이 차고 불안해요",
        category_scores={"anxiety": 0.9, "depression": 0.4, "normal": 0.1},
    )
    record = build_query_buffer_record(
        event=query_event,
        analysis_event=analysis_event,
        model_revision="rev_001",
        confidence_kind="prototype_similarity_top1",
        metadata={
            "scorer_backend_name": "prototype_similarity",
            "was_translated": False,
        },
    )

    evidence = build_query_buffer_evidence(
        record=record,
        analysis_event=analysis_event,
    )

    assert evidence.evidence_id == "evidence:q1"
    assert evidence.source_event_ref == "q1"
    assert evidence.top1_label == "anxiety"
    assert evidence.top2_label == "depression"
    assert evidence.margin == pytest.approx(0.5)
    assert evidence.confidence_kind == "prototype_similarity_top1"
    assert evidence.raw_scores == analysis_event.category_scores
    assert (
        evidence.metadata["evidence_backend_name"]
        == QUERY_BUFFER_PROJECTION_BACKEND_NAME
    )
    assert evidence.metadata["query_buffer_model_revision"] == "rev_001"
    assert evidence.metadata["query_buffer_locale"] == "ko-KR"
    assert evidence.metadata["query_buffer_source_type"] == "user_message"
    assert (
        evidence.metadata["query_buffer.scorer_backend_name"] == "prototype_similarity"
    )
    assert evidence.metadata["translation_used"] is False


def test_build_query_buffer_evidences_requires_matching_analysis_event() -> None:
    query_event, analysis_event = _build_pair(
        query_id="q1",
        text="잠이 안 와요",
        category_scores={"anxiety": 0.7, "normal": 0.1},
    )
    record = build_query_buffer_record(
        event=query_event,
        analysis_event=analysis_event,
        model_revision="rev_001",
        confidence_kind="prototype_similarity_top1",
    )

    with pytest.raises(ValueError, match="Missing AnalysisEvent"):
        build_query_buffer_evidences(records=(record,), analysis_events=())


def test_build_query_buffer_evidence_rejects_snapshot_mismatch() -> None:
    query_event, analysis_event = _build_pair(
        query_id="q1",
        text="계속 긴장돼요",
        category_scores={"anxiety": 0.8, "depression": 0.2},
    )
    record = build_query_buffer_record(
        event=query_event,
        analysis_event=analysis_event,
        model_revision="rev_001",
        confidence_kind="prototype_similarity_top1",
    )
    mismatched_record = replace(record, confidence=0.1)

    with pytest.raises(ValueError, match="confidence"):
        build_query_buffer_evidence(
            record=mismatched_record,
            analysis_event=analysis_event,
        )


def test_query_buffer_selection_service_filters_candidates_with_policy() -> None:
    query_event_1, analysis_event_1 = _build_pair(
        query_id="q1",
        text="불안이 심해요",
        category_scores={"anxiety": 0.85, "depression": 0.2, "normal": 0.1},
    )
    query_event_2, analysis_event_2 = _build_pair(
        query_id="q2",
        text="그냥 조금 예민해요",
        category_scores={"anxiety": 0.62, "depression": 0.6, "normal": 0.1},
    )
    record_1 = build_query_buffer_record(
        event=query_event_1,
        analysis_event=analysis_event_1,
        model_revision="rev_001",
        confidence_kind="prototype_similarity_top1",
    )
    record_2 = build_query_buffer_record(
        event=query_event_2,
        analysis_event=analysis_event_2,
        model_revision="rev_001",
        confidence_kind="prototype_similarity_top1",
    )

    service = QueryBufferSelectionService()
    result = service.select(
        records=(record_1, record_2),
        analysis_events=(analysis_event_1, analysis_event_2),
        training_task=_build_task(),
    )

    assert len(result.evidences) == 2
    assert result.accepted_count == 1
    accepted = result.accepted_candidates[0]
    rejected = next(
        candidate
        for candidate in result.candidates
        if candidate.source_event_ref == "q2"
    )

    assert accepted.source_event_ref == "q1"
    assert accepted.evidence_ref == "evidence:q1"
    assert accepted.selection_context is not None
    assert accepted.selection_context.evidence_backend_name == (
        QUERY_BUFFER_PROJECTION_BACKEND_NAME
    )
    assert accepted.confidence_kind == "prototype_similarity_top1"
    assert rejected.accepted is False
