"""Query buffer selection diagnostics unit tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agent.src.infrastructure.repositories.query_buffer_repository import (
    build_query_buffer_record,
)
from agent.src.services.training.selection.query_buffer_selection_diagnostics import (
    QUERY_BUFFER_SELECTION_SUMMARY_SCHEMA_VERSION,
    QUERY_BUFFER_SELECTION_TRACE_SCHEMA_VERSION,
    QueryBufferSelectionDiagnosticsService,
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
    confidence_threshold: float = 0.7,
    max_examples: int = 1,
) -> TrainingTask:
    return TrainingTask(
        schema_version="training_task.v1",
        task_id="task_diag_001",
        round_id="round_diag_001",
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
            extras={"selection.confidence_threshold": confidence_threshold},
        ),
        selection_policy=TrainingSelectionPolicy(max_examples=max_examples),
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


def test_query_buffer_selection_diagnostics_service_builds_summary_and_trace() -> None:
    pair_1 = _build_pair(
        query_id="q1",
        text="불안이 너무 심해요",
        category_scores={"anxiety": 0.95, "depression": 0.2, "normal": 0.1},
    )
    pair_2 = _build_pair(
        query_id="q2",
        text="계속 가라앉아요",
        category_scores={"depression": 0.9, "anxiety": 0.4, "normal": 0.2},
    )
    pair_3 = _build_pair(
        query_id="q3",
        text="조금 예민한 정도예요",
        category_scores={"anxiety": 0.55, "depression": 0.5, "normal": 0.1},
    )
    records = tuple(
        build_query_buffer_record(
            event=query_event,
            analysis_event=analysis_event,
            model_revision="rev_001",
            confidence_kind="classifier_head_logit_top1",
            metadata={
                "scorer_backend_name": "classifier_head_logits",
                "was_translated": False,
            },
        )
        for query_event, analysis_event in (pair_1, pair_2, pair_3)
    )
    analysis_events = tuple(
        analysis_event for _, analysis_event in (pair_1, pair_2, pair_3)
    )

    selection_result = QueryBufferSelectionService().select(
        records=records,
        analysis_events=analysis_events,
        training_task=_build_task(),
    )
    diagnostics = QueryBufferSelectionDiagnosticsService().build(
        selection_result=selection_result,
        records=records,
    )

    summary = diagnostics.summary
    assert summary["schema_version"] == QUERY_BUFFER_SELECTION_SUMMARY_SCHEMA_VERSION
    assert summary["total_candidates"] == 3
    assert summary["final_accepted_count"] == 1
    assert summary["stage_counts"] == {
        "accepted": 1,
        "dropped_by_cap": 1,
        "policy_rejected": 1,
    }
    assert summary["accepted_label_counts"] == {"anxiety": 1}
    assert summary["pseudo_label_counts"] == {"anxiety": 2, "depression": 1}
    assert summary["locale_counts"] == {"ko-KR": 3}
    assert summary["model_revision_counts"] == {"rev_001": 3}
    assert summary["selection_parameter_counts"] == {"confidence_threshold=0.7": 3}
    assert summary["max_examples_counts"] == {"1": 3}
    assert summary["evidence_backend_name_counts"] == {"query_buffer_projection": 3}
    assert summary["pseudo_label_algorithm_name_counts"] == {"top1_confidence_only": 3}
    confidence_stats = summary["confidence_stats"]
    assert isinstance(confidence_stats, dict)
    assert confidence_stats["count"] == 3
    assert confidence_stats["max"] == pytest.approx(0.95)
    summary_payload = summary
    confidence_stats_payload = summary_payload["confidence_stats"]
    assert isinstance(confidence_stats_payload, dict)
    assert confidence_stats_payload["max"] == pytest.approx(0.95)

    trace_by_query_id = {str(row["query_id"]): row for row in diagnostics.trace_rows}
    assert trace_by_query_id["q1"]["schema_version"] == (
        QUERY_BUFFER_SELECTION_TRACE_SCHEMA_VERSION
    )
    assert trace_by_query_id["q1"]["selection_stage"] == "accepted"
    assert trace_by_query_id["q1"]["selected_by_cap"] is True
    assert trace_by_query_id["q1"]["policy_accepted"] is True
    assert trace_by_query_id["q1"]["evidence_backend_name"] == (
        "query_buffer_projection"
    )
    assert trace_by_query_id["q1"]["pseudo_label_algorithm_name"] == (
        "top1_confidence_only"
    )
    assert trace_by_query_id["q1"]["raw_scores"] == {
        "anxiety": 0.95,
        "depression": 0.2,
        "normal": 0.1,
    }
    assert trace_by_query_id["q1"]["query_buffer_metadata"] == {
        "scorer_backend_name": "classifier_head_logits",
        "was_translated": False,
    }
    q1_payload = trace_by_query_id["q1"]
    assert q1_payload["occurred_at"] == pair_1[0].occurred_at.isoformat()
    assert list(q1_payload["raw_scores"]) == [
        "anxiety",
        "depression",
        "normal",
    ]
    assert list(q1_payload["query_buffer_metadata"]) == [
        "scorer_backend_name",
        "was_translated",
    ]
    assert trace_by_query_id["q2"]["selection_stage"] == "dropped_by_cap"
    assert trace_by_query_id["q2"]["selected_by_cap"] is False
    assert trace_by_query_id["q2"]["pre_cap_rank"] == 2
    assert trace_by_query_id["q3"]["selection_stage"] == "policy_rejected"
    assert trace_by_query_id["q3"]["policy_accepted"] is False


def test_query_buffer_selection_diagnostics_service_requires_matching_record() -> None:
    query_event, analysis_event = _build_pair(
        query_id="q1",
        text="계속 긴장돼요",
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
        training_task=_build_task(max_examples=4),
    )

    with pytest.raises(ValueError, match="Missing QueryBufferRecord"):
        QueryBufferSelectionDiagnosticsService().build(
            selection_result=selection_result,
            records=(),
        )
