"""Query adaptation dataset assembly unit tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agent.src.infrastructure.repositories.query_buffer_repository import (
    build_query_buffer_record,
)
from agent.src.services.training.datasets.query_adaptation_dataset_service import (
    QueryAdaptationDatasetConfig,
    QueryAdaptationDatasetService,
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


def _build_task() -> TrainingTask:
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
            pseudo_label_algorithm_name="top1_confidence_only",
            extras={"selection.confidence_threshold": 0.8},
        ),
        selection_policy=TrainingSelectionPolicy(max_examples=8),
    )


def _build_pair(
    *,
    query_id: str,
    text: str,
    translated_text: str | None,
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
        translated_text=translated_text,
        embedding_model_id="tracemind-embed",
        translation_model_id=None if translated_text is None else "nllb",
        category_scores=category_scores,
    )
    return query_event, analysis_event


def test_query_adaptation_dataset_service_builds_raw_text_examples() -> None:
    query_event_1, analysis_event_1 = _build_pair(
        query_id="q1",
        text="불안이 심해요",
        translated_text="I feel anxious",
        category_scores={"anxiety": 0.85, "depression": 0.2, "normal": 0.1},
    )
    query_event_2, analysis_event_2 = _build_pair(
        query_id="q2",
        text="요즘 너무 가라앉아요",
        translated_text=None,
        category_scores={"depression": 0.92, "anxiety": 0.1, "normal": 0.05},
    )
    record_1 = build_query_buffer_record(
        event=query_event_1,
        analysis_event=analysis_event_1,
        model_revision="rev_001",
        confidence_kind="classifier_head_logit_top1",
        metadata={"was_translated": True},
    )
    record_2 = build_query_buffer_record(
        event=query_event_2,
        analysis_event=analysis_event_2,
        model_revision="rev_001",
        confidence_kind="classifier_head_logit_top1",
        metadata={"was_translated": False},
    )
    selection_result = QueryBufferSelectionService().select(
        records=(record_1, record_2),
        analysis_events=(analysis_event_1, analysis_event_2),
        training_task=_build_task(),
    )

    dataset = QueryAdaptationDatasetService().build_dataset(
        selection_result=selection_result,
        records=(record_1, record_2),
        analysis_events=(analysis_event_1, analysis_event_2),
    )

    assert dataset.count == 2
    assert dataset.label_by_query_id == {
        "q1": "anxiety",
        "q2": "depression",
    }
    assert dataset.source_rows[0].text == "불안이 심해요"
    assert dataset.source_rows[0].translated_text == "I feel anxious"
    assert dataset.source_rows[1].text == "요즘 너무 가라앉아요"
    assert dataset.source_rows[1].translated_text is None
    assert dataset.examples[0].provenance.locale == "ko-KR"
    assert dataset.examples[0].provenance.selection_context is not None
    assert (
        dataset.examples[0].provenance.selection_context.selection_stage.value
        == "accepted"
    )
    assert (
        dataset.examples[0].provenance.selection_context.pseudo_label_algorithm_name
        == "top1_confidence_only"
    )
    assert (
        dataset.examples[0].provenance.query_buffer_metadata["was_translated"] is True
    )
    assert dataset.examples[0].label_source == "pseudo_label"


def test_query_adaptation_dataset_service_requires_query_buffer_record() -> None:
    query_event, analysis_event = _build_pair(
        query_id="q1",
        text="계속 예민해요",
        translated_text=None,
        category_scores={"anxiety": 0.82, "depression": 0.1},
    )
    record = build_query_buffer_record(
        event=query_event,
        analysis_event=analysis_event,
        model_revision="rev_001",
        confidence_kind="classifier_head_logit_top1",
    )
    selection_result = QueryBufferSelectionService().select(
        records=(record,),
        analysis_events=(analysis_event,),
        training_task=_build_task(),
    )

    with pytest.raises(ValueError, match="Missing QueryBufferRecord"):
        QueryAdaptationDatasetService().build_dataset(
            selection_result=selection_result,
            records=(),
            analysis_events=(analysis_event,),
        )


def test_query_adaptation_dataset_service_rejects_duplicate_record_key() -> None:
    query_event, analysis_event = _build_pair(
        query_id="q1",
        text="숨이 차요",
        translated_text=None,
        category_scores={"anxiety": 0.9, "normal": 0.1},
    )
    record = build_query_buffer_record(
        event=query_event,
        analysis_event=analysis_event,
        model_revision="rev_001",
        confidence_kind="classifier_head_logit_top1",
    )
    selection_result = QueryBufferSelectionService().select(
        records=(record,),
        analysis_events=(analysis_event,),
        training_task=_build_task(),
    )

    with pytest.raises(ValueError, match="Duplicate QueryBufferRecord"):
        QueryAdaptationDatasetService().build_dataset(
            selection_result=selection_result,
            records=(record, record),
            analysis_events=(analysis_event,),
        )


def test_query_adaptation_dataset_service_rejects_manual_labels_in_pseudo_mode() -> (
    None
):
    query_event, analysis_event = _build_pair(
        query_id="q1",
        text="숨이 차요",
        translated_text=None,
        category_scores={"anxiety": 0.9, "normal": 0.1},
    )
    record = build_query_buffer_record(
        event=query_event,
        analysis_event=analysis_event,
        model_revision="rev_001",
        confidence_kind="classifier_head_logit_top1",
    )
    selection_result = QueryBufferSelectionService().select(
        records=(record,),
        analysis_events=(analysis_event,),
        training_task=_build_task(),
    )

    with pytest.raises(ValueError, match="manual_label_by_query_id"):
        QueryAdaptationDatasetService().build_dataset(
            selection_result=selection_result,
            records=(record,),
            analysis_events=(analysis_event,),
            manual_label_by_query_id={"q1": "depression"},
        )


def test_query_adaptation_dataset_service_can_prefer_manual_labels_later() -> None:
    query_event, analysis_event = _build_pair(
        query_id="q1",
        text="계속 우울해요",
        translated_text=None,
        category_scores={"anxiety": 0.9, "depression": 0.2},
    )
    record = build_query_buffer_record(
        event=query_event,
        analysis_event=analysis_event,
        model_revision="rev_001",
        confidence_kind="classifier_head_logit_top1",
    )
    selection_result = QueryBufferSelectionService().select(
        records=(record,),
        analysis_events=(analysis_event,),
        training_task=_build_task(),
    )

    dataset = QueryAdaptationDatasetService(
        config=QueryAdaptationDatasetConfig(label_policy_name="prefer_manual_label")
    ).build_dataset(
        selection_result=selection_result,
        records=(record,),
        analysis_events=(analysis_event,),
        manual_label_by_query_id={"q1": "depression"},
    )

    assert dataset.count == 1
    assert dataset.examples[0].label == "depression"
    assert dataset.examples[0].label_source == "manual_label"
