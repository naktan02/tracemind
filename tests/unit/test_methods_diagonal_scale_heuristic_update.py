"""Diagonal-scale heuristic update method core tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from methods.adaptation.diagonal_scale.config import (
    DiagonalScaleHeuristicTrainingBackendConfig,
)
from methods.adaptation.diagonal_scale.heuristic_update import (
    build_diagonal_scale_client_metrics,
    build_diagonal_scale_heuristic_update,
)
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import (
    ClientMetricKeys,
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
    TrainingTask,
)
from shared.src.domain.entities.training.pseudo_label_candidate import (
    PseudoLabelCandidate,
)


@dataclass(slots=True)
class _AcceptedExample:
    update_embedding: list[float]
    candidate: PseudoLabelCandidate | None


def _build_manifest() -> ModelManifest:
    return ModelManifest(
        schema_version="model_manifest.v1",
        model_id="tracemind-embed",
        model_revision="rev_000",
        published_at=datetime(2026, 5, 4, tzinfo=timezone.utc),
        artifact_kind="shared_adapter_state",
        artifact_ref="/tmp/rev_000.json",
        prototype_version="proto_000",
        training_scope="adapter_only",
        training_enabled=True,
        compatible_task_types=("pseudo_label_self_training",),
    )


def _build_task() -> TrainingTask:
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
        learning_rate=0.01,
        max_steps=10,
        objective_config=TrainingObjectiveConfig(
            training_backend_name="diagonal_scale_heuristic",
        ),
        selection_policy=TrainingSelectionPolicy(max_examples=2),
        min_required_examples=1,
        gradient_clip_norm=None,
    )


def _candidate(
    *,
    label: str,
    confidence: float,
    margin: float,
    sample_weight: float,
) -> PseudoLabelCandidate:
    return PseudoLabelCandidate(
        schema_version="pseudo_label_candidate.v1",
        candidate_id=f"candidate_{label}",
        source_event_ref=f"query_{label}",
        occurred_at=datetime(2026, 5, 4, tzinfo=timezone.utc),
        label=label,
        confidence=confidence,
        margin=margin,
        accepted=True,
        sample_weight=sample_weight,
    )


def test_build_diagonal_scale_heuristic_update_uses_weighted_average() -> None:
    update = build_diagonal_scale_heuristic_update(
        training_task=_build_task(),
        model_manifest=_build_manifest(),
        accepted_examples=(
            _AcceptedExample(
                update_embedding=[2.0, 0.0],
                candidate=_candidate(
                    label="anxiety",
                    confidence=0.8,
                    margin=0.3,
                    sample_weight=2.0,
                ),
            ),
            _AcceptedExample(
                update_embedding=[0.0, 2.0],
                candidate=_candidate(
                    label="normal",
                    confidence=0.6,
                    margin=0.2,
                    sample_weight=1.0,
                ),
            ),
        ),
        created_at=datetime(2026, 5, 4, tzinfo=timezone.utc),
        config=DiagonalScaleHeuristicTrainingBackendConfig(
            delta_scale_multiplier=1.0,
            max_abs_delta=1.0,
            minimum_effective_scale=0.0,
        ),
    )

    assert update.model_id == "tracemind-embed"
    assert update.base_model_revision == "rev_000"
    assert update.dimension_deltas == pytest.approx([4.0 / 30.0, 2.0 / 30.0])
    assert update.example_count == 2
    assert update.mean_confidence == pytest.approx(0.7)
    assert update.mean_margin == pytest.approx(0.25)
    assert update.label_counts == {"anxiety": 1, "normal": 1}
    assert update.adapter_kind == "diagonal_scale"


def test_build_diagonal_scale_heuristic_update_clamps_delta() -> None:
    update = build_diagonal_scale_heuristic_update(
        training_task=_build_task(),
        model_manifest=_build_manifest(),
        accepted_examples=(
            _AcceptedExample(
                update_embedding=[100.0],
                candidate=_candidate(
                    label="anxiety",
                    confidence=0.9,
                    margin=0.8,
                    sample_weight=1.0,
                ),
            ),
        ),
        created_at=datetime(2026, 5, 4, tzinfo=timezone.utc),
        config=DiagonalScaleHeuristicTrainingBackendConfig(
            delta_scale_multiplier=10.0,
            max_abs_delta=0.05,
            minimum_effective_scale=0.0,
        ),
    )

    assert update.dimension_deltas == pytest.approx([0.05])


def test_build_diagonal_scale_heuristic_update_rejects_empty_examples() -> None:
    with pytest.raises(ValueError, match="accepted_examples must not be empty"):
        build_diagonal_scale_heuristic_update(
            training_task=_build_task(),
            model_manifest=_build_manifest(),
            accepted_examples=(),
            created_at=datetime(2026, 5, 4, tzinfo=timezone.utc),
            config=DiagonalScaleHeuristicTrainingBackendConfig(),
        )


def test_build_diagonal_scale_heuristic_update_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="same dimension"):
        build_diagonal_scale_heuristic_update(
            training_task=_build_task(),
            model_manifest=_build_manifest(),
            accepted_examples=(
                _AcceptedExample(
                    update_embedding=[1.0, 0.0],
                    candidate=_candidate(
                        label="anxiety",
                        confidence=0.9,
                        margin=0.8,
                        sample_weight=1.0,
                    ),
                ),
                _AcceptedExample(
                    update_embedding=[1.0],
                    candidate=_candidate(
                        label="normal",
                        confidence=0.7,
                        margin=0.5,
                        sample_weight=1.0,
                    ),
                ),
            ),
            created_at=datetime(2026, 5, 4, tzinfo=timezone.utc),
            config=DiagonalScaleHeuristicTrainingBackendConfig(),
        )


def test_build_diagonal_scale_heuristic_update_rejects_missing_candidate() -> None:
    with pytest.raises(ValueError, match="pseudo-label candidate"):
        build_diagonal_scale_heuristic_update(
            training_task=_build_task(),
            model_manifest=_build_manifest(),
            accepted_examples=(
                _AcceptedExample(update_embedding=[1.0, 0.0], candidate=None),
            ),
            created_at=datetime(2026, 5, 4, tzinfo=timezone.utc),
            config=DiagonalScaleHeuristicTrainingBackendConfig(),
        )


def test_build_diagonal_scale_client_metrics_reads_update_summary() -> None:
    update = build_diagonal_scale_heuristic_update(
        training_task=_build_task(),
        model_manifest=_build_manifest(),
        accepted_examples=(
            _AcceptedExample(
                update_embedding=[3.0, 4.0],
                candidate=_candidate(
                    label="anxiety",
                    confidence=0.9,
                    margin=0.8,
                    sample_weight=1.0,
                ),
            ),
        ),
        created_at=datetime(2026, 5, 4, tzinfo=timezone.utc),
        config=DiagonalScaleHeuristicTrainingBackendConfig(
            delta_scale_multiplier=0.1,
            max_abs_delta=1.0,
            minimum_effective_scale=0.0,
        ),
    )

    metrics = build_diagonal_scale_client_metrics(update)

    assert metrics[ClientMetricKeys.MEAN_CONFIDENCE] == pytest.approx(0.9)
    assert metrics[ClientMetricKeys.MEAN_MARGIN] == pytest.approx(0.8)
    assert metrics[ClientMetricKeys.DELTA_L2_NORM] == pytest.approx(update.l2_norm())
