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
from shared.src.domain.entities.training.training_task_config import (
    TrainingObjectiveConfig,
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


def test_training_objective_config_payload_accepts_policy_fields() -> None:
    payload = TrainingObjectiveConfigPayload(
        loss="contrastive",
        confidence_threshold=0.7,
        margin_threshold=0.05,
        score_policy_name="topk_mean_cosine",
        score_top_k=3,
        acceptance_policy_name="top1_margin_threshold",
    )

    assert payload.score_policy_name == "topk_mean_cosine"
    assert payload.score_top_k == 3
    assert payload.acceptance_policy_name == "top1_margin_threshold"


def test_training_objective_config_round_trips_policy_fields() -> None:
    config = TrainingObjectiveConfig.from_mapping(
        {
            "loss": "contrastive",
            "confidence_threshold": 0.65,
            "margin_threshold": 0.03,
            "score_policy_name": "topk_mean_cosine",
            "score_top_k": 2,
            "acceptance_policy_name": "top1_confidence_only",
            "temperature": 0.8,
        }
    )

    assert config.score_policy_name == "topk_mean_cosine"
    assert config.score_top_k == 2
    assert config.acceptance_policy_name == "top1_confidence_only"
    assert config.extras == {"temperature": 0.8}
    assert config.to_mapping() == {
        "loss": "contrastive",
        "confidence_threshold": 0.65,
        "margin_threshold": 0.03,
        "score_policy_name": "topk_mean_cosine",
        "score_top_k": 2,
        "acceptance_policy_name": "top1_confidence_only",
        "temperature": 0.8,
    }
