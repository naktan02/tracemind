"""로컬 pseudo-label update 생성 서비스 tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from agent.src.infrastructure.repositories.training_artifact_repository import (
    TrainingArtifactRepository,
)
from agent.src.services.training.local_training_service import (
    EmbeddedTrainingExample,
    LocalTrainingRequest,
    LocalTrainingService,
)
from shared.src.domain.entities.artifacts.model_manifest import ModelManifest
from shared.src.domain.entities.inference.events import ScoredEvent
from shared.src.domain.entities.training.training_task import TrainingTask


def _build_manifest() -> ModelManifest:
    return ModelManifest(
        schema_version="model_manifest.v1",
        model_id="tracemind-embed",
        model_revision="rev_000",
        published_at=datetime(2026, 3, 29, tzinfo=timezone.utc),
        artifact_kind="vector_adapter_state",
        artifact_ref="/tmp/rev_000.json",
        prototype_version="proto_000",
        training_scope="adapter_only",
        training_enabled=True,
        compatible_task_types=("pseudo_label_self_training",),
    )


def _build_task(*, min_required_examples: int = 1) -> TrainingTask:
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
        objective_config={
            "loss": "synthetic_vector_adapter",
            "confidence_threshold": 0.6,
            "margin_threshold": 0.02,
        },
        selection_policy={"max_examples": 1},
        min_required_examples=min_required_examples,
        gradient_clip_norm=0.05,
    )


def _make_example(
    *,
    query_id: str,
    scores: dict[str, float],
    embedding: list[float],
) -> EmbeddedTrainingExample:
    return EmbeddedTrainingExample(
        scored_event=ScoredEvent(
            query_id=query_id,
            occurred_at=datetime(2026, 3, 29, tzinfo=timezone.utc),
            translated_text=None,
            embedding_model_id="tracemind-embed",
            translation_model_id=None,
            category_scores=scores,
        ),
        embedding=embedding,
    )


def test_local_training_service_creates_update_from_top_candidates(
    tmp_path: Path,
) -> None:
    repository = TrainingArtifactRepository(state_root=tmp_path / "training_updates")
    service = LocalTrainingService(repository=repository)

    result = service.run(
        LocalTrainingRequest(
            training_examples=(
                _make_example(
                    query_id="q1",
                    scores={"anxiety": 0.91, "depression": 0.2, "normal": 0.1},
                    embedding=[1.0, 0.0],
                ),
                _make_example(
                    query_id="q2",
                    scores={"depression": 0.88, "anxiety": 0.5, "normal": 0.1},
                    embedding=[0.0, 1.0],
                ),
                _make_example(
                    query_id="q3",
                    scores={"anxiety": 0.62, "depression": 0.58, "normal": 0.1},
                    embedding=[0.7, 0.7],
                ),
            ),
            training_task=_build_task(),
            model_manifest=_build_manifest(),
        )
    )

    assert result.selection_result.total_count == 3
    assert result.selection_result.accepted_count == 1
    assert result.update_envelope is not None
    assert result.update_payload is not None
    assert result.update_envelope.example_count == 1
    assert result.update_payload.label_counts == {"anxiety": 1}
    assert result.update_envelope.payload_format == "vector_adapter_delta"
    assert Path(result.update_envelope.payload_ref).exists()


def test_local_training_service_skips_update_when_examples_are_insufficient(
    tmp_path: Path,
) -> None:
    repository = TrainingArtifactRepository(state_root=tmp_path / "training_updates")
    service = LocalTrainingService(repository=repository)

    result = service.run(
        LocalTrainingRequest(
            training_examples=(
                _make_example(
                    query_id="q1",
                    scores={"anxiety": 0.91, "depression": 0.2},
                    embedding=[1.0, 0.0],
                ),
            ),
            training_task=_build_task(min_required_examples=2),
            model_manifest=_build_manifest(),
        )
    )

    assert result.selection_result.accepted_count == 1
    assert result.update_envelope is None
    assert result.update_payload is None
