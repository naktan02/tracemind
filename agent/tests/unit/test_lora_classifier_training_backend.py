"""LoRA-classifier local training backend tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent.src.infrastructure.repositories.training_artifact_repository import (
    TrainingArtifactRepository,
)
from agent.src.services.training.backends.training.lora_classifier_trainer import (
    LoraClassifierTrainingBackend,
)
from agent.src.services.training.backends.training.registry import (
    build_shared_adapter_training_backend,
    list_registered_shared_adapter_training_backend_names,
    list_shared_adapter_training_backend_catalog_entries,
)
from agent.src.services.training.examples.models import EmbeddedTrainingExample
from agent.src.services.training.execution.local_training_service import (
    LocalTrainingRequest,
    LocalTrainingService,
)
from shared.src.contracts.adapter_contracts import LoraClassifierDelta
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
    TrainingTask,
)
from shared.src.domain.entities.inference.events import ScoredEvent
from shared.src.domain.entities.training.pseudo_label_candidate import (
    PseudoLabelCandidate,
)


def _build_manifest() -> ModelManifest:
    return ModelManifest(
        schema_version="model_manifest.v1",
        model_id="tracemind-lora",
        model_revision="rev_000",
        published_at=datetime(2026, 4, 21, tzinfo=timezone.utc),
        artifact_kind="shared_adapter_state",
        artifact_ref="shared_adapter_state::rev_000",
        prototype_version="proto_000",
        training_scope="adapter_only",
        training_enabled=True,
        compatible_task_types=("pseudo_label_self_training",),
    )


def _build_task(
    *,
    extras: dict[str, str | int | float | bool] | None = None,
) -> TrainingTask:
    return TrainingTask(
        schema_version="training_task.v1",
        task_id="task_lora_001",
        round_id="round_lora_001",
        model_id="tracemind-lora",
        model_revision="rev_000",
        task_type="pseudo_label_self_training",
        training_scope="adapter_only",
        local_epochs=1,
        batch_size=2,
        learning_rate=1e-4,
        max_steps=1,
        objective_config=TrainingObjectiveConfig(
            training_backend_name="lora_classifier_trainer",
            confidence_threshold=0.6,
            margin_threshold=0.02,
            scorer_backend_name="prototype_similarity",
            acceptance_policy_name="top1_margin_threshold",
            pseudo_label_algorithm_name="top1_margin_threshold",
            privacy_guard_name="noop",
            extras={} if extras is None else extras,
        ),
        selection_policy=TrainingSelectionPolicy(max_examples=2),
        min_required_examples=1,
        gradient_clip_norm=None,
    )


def _candidate(
    *,
    query_id: str,
    label: str,
    confidence: float = 0.92,
    margin: float = 0.71,
) -> PseudoLabelCandidate:
    return PseudoLabelCandidate(
        schema_version="pseudo_label_candidate.v1",
        candidate_id=f"cand_{query_id}",
        source_event_ref=query_id,
        occurred_at=datetime(2026, 4, 21, tzinfo=timezone.utc),
        label=label,
        confidence=confidence,
        margin=margin,
        accepted=True,
    )


def _example(
    *,
    query_id: str = "q1",
    raw_text: str | None = "오늘 너무 불안해요",
    translated_text: str | None = None,
    candidate: PseudoLabelCandidate | None = None,
) -> EmbeddedTrainingExample:
    metadata: dict[str, str | int | float | bool] = {}
    if raw_text is not None:
        metadata["raw_text"] = raw_text
        metadata["training_text"] = raw_text
    return EmbeddedTrainingExample(
        scored_event=ScoredEvent(
            query_id=query_id,
            occurred_at=datetime(2026, 4, 21, tzinfo=timezone.utc),
            translated_text=translated_text,
            embedding_model_id="tracemind-lora",
            translation_model_id=None,
            category_scores={"anxiety": 0.92, "depression": 0.2, "normal": 0.1},
        ),
        embedding=[1.0, 0.0],
        candidate=candidate or _candidate(query_id=query_id, label="anxiety"),
        metadata=metadata,
    )


def test_lora_classifier_backend_builds_artifact_ref_update_without_text() -> None:
    backend = LoraClassifierTrainingBackend.from_objective_config(
        TrainingObjectiveConfig(
            training_backend_name="lora_classifier_trainer",
            extras={
                "lora_classifier_trainer.rank": 16,
                "lora_classifier_trainer.alpha": 32,
                "lora_classifier_trainer.label_schema": ("anxiety,depression,normal"),
            },
        )
    )

    update = backend.build_update(
        training_task=_build_task(),
        model_manifest=_build_manifest(),
        accepted_examples=(_example(),),
        created_at=datetime(2026, 4, 21, 12, 30, tzinfo=timezone.utc),
    )

    assert isinstance(update, LoraClassifierDelta)
    assert update.adapter_kind == "lora_classifier"
    assert update.schema_version == "lora_classifier_delta.v1"
    assert update.label_schema == ["anxiety", "depression", "normal"]
    assert update.lora_config.rank == 16
    assert update.lora_config.alpha == 32
    assert update.delta_format == "agent_local_artifact_ref"
    assert update.lora_delta_artifact_ref == (
        "agent-local://lora_classifier/round_lora_001/task_lora_001/"
        "20260421T123000000000Z/lora_delta"
    )
    assert update.classifier_head_delta_artifact_ref is not None
    assert update.mean_confidence == pytest.approx(0.92)
    assert update.label_counts == {"anxiety": 1}
    assert update.l2_norm() == 0.0
    assert "오늘 너무 불안해요" not in update.model_dump_json()


def test_lora_classifier_backend_rejects_fixed_embedding_only_examples() -> None:
    backend = LoraClassifierTrainingBackend()

    with pytest.raises(
        ValueError,
        match="requires raw text or translated text",
    ):
        backend.build_update(
            training_task=_build_task(),
            model_manifest=_build_manifest(),
            accepted_examples=(_example(raw_text=None, translated_text=None),),
            created_at=datetime(2026, 4, 21, tzinfo=timezone.utc),
        )


def test_lora_classifier_training_backend_is_registered() -> None:
    backend = build_shared_adapter_training_backend(
        "lora_classifier_trainer",
        objective_config=TrainingObjectiveConfig(
            training_backend_name="lora_classifier_trainer"
        ),
    )
    catalog_entries = {
        entry.item_name: entry
        for entry in list_shared_adapter_training_backend_catalog_entries()
    }

    assert "lora_classifier_trainer" in (
        list_registered_shared_adapter_training_backend_names()
    )
    assert backend.adapter_kind == "lora_classifier"
    assert backend.payload_format == "lora_classifier_update"
    assert (
        catalog_entries["lora_classifier_trainer"].metadata["requires_raw_text"] is True
    )
    assert (
        catalog_entries["lora_classifier_trainer"].metadata[
            "supports_live_stored_event_runtime"
        ]
        is False
    )


def test_local_training_service_uses_lora_classifier_backend(
    tmp_path: Path,
) -> None:
    repository = TrainingArtifactRepository(state_root=tmp_path / "agent_state")
    service = LocalTrainingService(repository=repository)

    result = service.run(
        LocalTrainingRequest(
            training_examples=(_example(candidate=None),),
            training_task=_build_task(),
            model_manifest=_build_manifest(),
            created_at=datetime(2026, 4, 21, 12, 30, tzinfo=timezone.utc),
        )
    )

    assert result.update_envelope is not None
    assert result.update_payload is not None
    assert result.update_payload.adapter_kind == "lora_classifier"
    assert result.update_envelope.payload_format == "lora_classifier_update"
    assert result.update_envelope.client_metrics["lora_training_rows"] == 1.0
    assert result.update_envelope.client_metrics["delta_l2_norm"] == 0.0
    assert result.update_payload.label_counts == {"anxiety": 1}

    loaded_payload = repository.load_shared_adapter_update(
        result.update_envelope.update_id
    )
    assert isinstance(loaded_payload, LoraClassifierDelta)
