from __future__ import annotations

from datetime import datetime, timezone

from shared.src.contracts.model_contracts import ModelManifestPayload
from shared.src.contracts.personalization_contracts import (
    PersonalizationStatePayload,
)
from shared.src.contracts.training_contracts import (
    DecisionFeedbackSignalPayload,
    TrainingObjectiveConfigPayload,
    TrainingSelectionPolicyPayload,
    TrainingTaskPayload,
    TrainingUpdateEnvelopePayload,
)


def test_model_manifest_payload_accepts_active_fields() -> None:
    payload = ModelManifestPayload(
        schema_version="model_manifest.v1",
        model_id="tracemind-embed",
        model_revision="rev_001",
        published_at=datetime.now(tz=timezone.utc),
        artifact_kind="embedding_backbone",
        artifact_ref="models/rev_001",
        prototype_version="proto_001",
        training_scope="adapter_only",
        training_enabled=True,
        compatible_task_types=["pseudo_label_self_training"],
    )

    assert payload.training_enabled is True
    assert payload.training_scope == "adapter_only"


def test_training_payloads_capture_round_and_revision() -> None:
    task = TrainingTaskPayload(
        schema_version="training_task.v1",
        task_id="task_001",
        round_id="round_001",
        model_id="tracemind-embed",
        model_revision="rev_001",
        task_type="pseudo_label_self_training",
        training_scope="adapter_only",
        local_epochs=1,
        batch_size=8,
        learning_rate=1e-4,
        max_steps=10,
        objective_config=TrainingObjectiveConfigPayload(loss="contrastive"),
        selection_policy=TrainingSelectionPolicyPayload(max_examples=32),
    )
    update = TrainingUpdateEnvelopePayload(
        schema_version="training_update_envelope.v1",
        update_id="update_001",
        round_id="round_001",
        task_id="task_001",
        model_id="tracemind-embed",
        base_model_revision="rev_001",
        training_scope="adapter_only",
        payload_ref="updates/update_001",
        payload_format="adapter_weights",
        example_count=12,
        client_metrics={"mean_loss": 0.5},
    )

    assert task.round_id == update.round_id
    assert update.base_model_revision == "rev_001"
    assert task.selection_policy.max_examples == 32


def test_feedback_and_personalization_payloads_are_local_friendly() -> None:
    signal = DecisionFeedbackSignalPayload(
        schema_version="decision_feedback_signal.v1",
        signal_id="signal_001",
        signal_type="pseudo_label",
        label="depression_rising",
        confidence=0.9,
        occurred_at=datetime.now(tz=timezone.utc),
    )
    state = PersonalizationStatePayload(
        schema_version="personalization_state.v1",
        state_version="ps_001",
        baseline_by_category={"depression": 0.2},
        threshold_by_category={"depression": 0.6},
        warmup_status="ready",
    )

    assert signal.signal_type == "pseudo_label"
    assert state.threshold_by_category["depression"] == 0.6
