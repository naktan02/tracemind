"""PseudoLabelSelectionService unit tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agent.src.services.training.pseudo_label_service import (
    PseudoLabelSelectionService,
)
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
    TrainingTask,
)
from shared.src.domain.entities.inference.events import ScoredEvent


def _build_task(
    *,
    acceptance_policy_name: str | None = None,
    evidence_backend_name: str | None = None,
    confidence_threshold: float = 0.6,
) -> TrainingTask:
    return TrainingTask(
        schema_version="training_task.v1",
        task_id="task_001",
        round_id="round_0001",
        model_id="tracemind-embed",
        model_revision="rev_000",
        task_type="pseudo_label_self_training",
        training_scope="adapter_only",
        local_epochs=1,
        batch_size=8,
        learning_rate=1e-2,
        max_steps=10,
        objective_config=TrainingObjectiveConfig(
            loss="diagonal_scale_heuristic",
            confidence_threshold=confidence_threshold,
            margin_threshold=0.02,
            evidence_backend_name=evidence_backend_name,
            acceptance_policy_name=acceptance_policy_name,
        ),
        selection_policy=TrainingSelectionPolicy(max_examples=8),
    )


def test_selection_service_keeps_top1_margin_threshold_as_default() -> None:
    service = PseudoLabelSelectionService()

    result = service.select(
        scored_events=(
            ScoredEvent(
                query_id="q1",
                occurred_at=datetime(2026, 3, 29, tzinfo=timezone.utc),
                translated_text=None,
                embedding_model_id="tracemind-embed",
                translation_model_id=None,
                category_scores={"anxiety": 0.62, "depression": 0.61, "normal": 0.1},
            ),
        ),
        training_task=_build_task(),
    )

    candidate = result.candidates[0]
    evidence = result.evidences[0]

    assert evidence.confidence_kind == "prototype_similarity"
    assert evidence.top1_label == "anxiety"
    assert candidate.accepted is False
    assert candidate.evidence_ref == "evidence:q1"
    assert candidate.confidence_kind == "prototype_similarity"
    assert candidate.sample_weight == pytest.approx(0.62)
    assert candidate.margin == pytest.approx(0.01)
    assert candidate.metadata["evidence_backend_name"] == "prototype_similarity_evidence"
    assert candidate.metadata["acceptance_policy_name"] == "top1_margin_threshold"


def test_selection_service_can_switch_policy_from_training_task() -> None:
    service = PseudoLabelSelectionService()

    result = service.select(
        scored_events=(
            ScoredEvent(
                query_id="q1",
                occurred_at=datetime(2026, 3, 29, tzinfo=timezone.utc),
                translated_text=None,
                embedding_model_id="tracemind-embed",
                translation_model_id=None,
                category_scores={"anxiety": 0.62, "depression": 0.61, "normal": 0.1},
            ),
        ),
        training_task=_build_task(
            acceptance_policy_name="top1_confidence_only"
        ),
    )

    candidate = result.candidates[0]
    assert candidate.accepted is True
    assert candidate.evidence_ref == "evidence:q1"
    assert candidate.metadata["acceptance_policy_name"] == "top1_confidence_only"
    assert result.accepted_count == 1


def test_selection_service_can_switch_to_fixmatch_weak_view_evidence() -> None:
    service = PseudoLabelSelectionService()

    result = service.select(
        scored_events=(
            ScoredEvent(
                query_id="q1",
                occurred_at=datetime(2026, 3, 29, tzinfo=timezone.utc),
                translated_text=None,
                embedding_model_id="tracemind-embed",
                translation_model_id=None,
                category_scores={"anxiety": 6.0, "depression": 0.0, "normal": -1.0},
            ),
        ),
        training_task=_build_task(
            acceptance_policy_name="top1_confidence_only",
            evidence_backend_name="fixmatch_weak_view_evidence",
            confidence_threshold=0.95,
        ),
    )

    candidate = result.candidates[0]
    evidence = result.evidences[0]

    assert evidence.confidence_kind == "posterior_probability"
    assert evidence.view_kind == "weak_view"
    assert evidence.label_distribution is not None
    assert sum(evidence.label_distribution.values()) == pytest.approx(1.0)
    assert evidence.metadata["temperature"] == 1.0
    assert candidate.accepted is True
    assert candidate.confidence_kind == "posterior_probability"
    assert candidate.sample_weight == pytest.approx(1.0)
    assert candidate.metadata["evidence_backend_name"] == "fixmatch_weak_view_evidence"
    assert result.accepted_count == 1
