"""Query buffer selection diagnostics IO tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from agent.src.infrastructure.repositories.query_buffer_repository import (
    build_query_buffer_record,
)
from agent.src.services.training.selection.query_buffer_selection_diagnostics import (
    QueryBufferSelectionDiagnosticsService,
)
from agent.src.services.training.selection.query_buffer_selection_service import (
    QueryBufferSelectionService,
)
from scripts.query_buffer_selection_diagnostics import (
    write_query_buffer_selection_diagnostics,
)
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
    TrainingTask,
)
from shared.src.domain.entities.inference.events import QueryEvent, ScoredEvent


def _build_task() -> TrainingTask:
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
            loss="diagonal_scale_heuristic",
            confidence_threshold=0.75,
            margin_threshold=0.02,
            pseudo_label_algorithm_name="top1_confidence_only",
        ),
        selection_policy=TrainingSelectionPolicy(max_examples=4),
    )


def _build_pair(
    *,
    query_id: str,
    text: str,
    category_scores: dict[str, float],
) -> tuple[QueryEvent, ScoredEvent]:
    occurred_at = datetime(2026, 4, 12, 12, 0, tzinfo=timezone.utc)
    query_event = QueryEvent(
        query_id=query_id,
        text=text,
        occurred_at=occurred_at,
        locale="ko-KR",
        source_type="user_message",
    )
    scored_event = ScoredEvent(
        query_id=query_id,
        occurred_at=occurred_at,
        translated_text=None,
        embedding_model_id="tracemind-embed",
        translation_model_id=None,
        category_scores=category_scores,
    )
    return query_event, scored_event


def test_write_query_buffer_selection_diagnostics_writes_summary_and_trace(
    tmp_path,
) -> None:
    pair_1 = _build_pair(
        query_id="q1",
        text="불안해서 잠이 안 와요",
        category_scores={"anxiety": 0.9, "depression": 0.2, "normal": 0.1},
    )
    pair_2 = _build_pair(
        query_id="q2",
        text="요즘 계속 우울해요",
        category_scores={"depression": 0.88, "anxiety": 0.3, "normal": 0.1},
    )
    records = tuple(
        build_query_buffer_record(
            event=query_event,
            scored_event=scored_event,
            model_revision="rev_001",
            confidence_kind="prototype_similarity_top1",
            metadata={"was_translated": False},
        )
        for query_event, scored_event in (pair_1, pair_2)
    )
    scored_events = tuple(scored_event for _, scored_event in (pair_1, pair_2))
    selection_result = QueryBufferSelectionService().select(
        records=records,
        scored_events=scored_events,
        training_task=_build_task(),
    )
    diagnostics = QueryBufferSelectionDiagnosticsService().build(
        selection_result=selection_result,
        records=records,
    )

    outputs = write_query_buffer_selection_diagnostics(
        diagnostics=diagnostics,
        output_prefix=tmp_path / "round_0001",
    )

    assert outputs.candidates_path.exists()
    assert outputs.summary_path.exists()

    summary = json.loads(outputs.summary_path.read_text(encoding="utf-8"))
    assert summary["total_candidates"] == 2
    assert summary["final_accepted_count"] == 2
    assert summary["stage_counts"] == {"accepted": 2}

    candidate_lines = outputs.candidates_path.read_text(encoding="utf-8").splitlines()
    assert len(candidate_lines) == 2
    first_row = json.loads(candidate_lines[0])
    assert first_row["query_id"] == "q1"
    assert first_row["selection_stage"] == "accepted"
    assert first_row["query_buffer_label"] == "anxiety"
