from __future__ import annotations

from shared.src.contracts.common_types import (
    TrainingScope,
    TrainingTaskType,
)
from shared.src.contracts.model_contracts import ArtifactKind
from shared.src.contracts.personalization_contracts import (
    PersonalizationWarmupStatus,
)
from shared.src.contracts.training_contracts import (
    FeedbackSignalType,
    TrainingObjectiveConfig,
    TrainingObjectiveConfigPayload,
)


def test_model_manifest_payload_accepts_active_fields(
    make_model_manifest_payload,
) -> None:
    payload = make_model_manifest_payload(
        model_id="tracemind-embed",
        model_revision="rev_001",
        artifact_kind=ArtifactKind.EMBEDDING_BACKBONE,
        artifact_ref="models/rev_001",
        prototype_version="proto_001",
        training_scope=TrainingScope.ADAPTER_ONLY,
        training_enabled=True,
        compatible_task_types=[TrainingTaskType.PSEUDO_LABEL_SELF_TRAINING],
    )

    assert payload.training_enabled is True
    assert payload.training_scope == TrainingScope.ADAPTER_ONLY


def test_training_payloads_capture_round_and_revision(
    make_training_task_payload,
    make_training_update_envelope_payload,
) -> None:
    task = make_training_task_payload(
        task_id="task_001",
        round_id="round_001",
        model_id="tracemind-embed",
        model_revision="rev_001",
        task_type=TrainingTaskType.PSEUDO_LABEL_SELF_TRAINING,
        training_scope=TrainingScope.ADAPTER_ONLY,
        local_epochs=1,
        batch_size=8,
        learning_rate=1e-4,
        max_steps=10,
        objective_config=TrainingObjectiveConfigPayload(loss="contrastive"),
    )
    update = make_training_update_envelope_payload(
        update_id="update_001",
        round_id="round_001",
        task_id="task_001",
        model_id="tracemind-embed",
        base_model_revision="rev_001",
        training_scope=TrainingScope.ADAPTER_ONLY,
        payload_ref="updates/update_001",
        payload_format="diagonal_scale_update",
        example_count=12,
        client_metrics={"mean_loss": 0.5},
    )

    assert task.round_id == update.round_id
    assert update.base_model_revision == "rev_001"
    assert task.selection_policy.max_examples == 32


def test_feedback_and_personalization_payloads_are_local_friendly(
    make_feedback_signal_payload,
    make_personalization_state_payload,
) -> None:
    signal = make_feedback_signal_payload(
        signal_type=FeedbackSignalType.PSEUDO_LABEL,
        label="depression_rising",
    )
    state = make_personalization_state_payload(
        warmup_status=PersonalizationWarmupStatus.READY,
    )

    assert signal.signal_type == FeedbackSignalType.PSEUDO_LABEL
    assert state.threshold_by_category["depression"] == 0.6


def test_training_objective_config_payload_accepts_policy_fields() -> None:
    payload = TrainingObjectiveConfigPayload(
        loss="contrastive",
        confidence_threshold=0.7,
        margin_threshold=0.05,
        score_policy_name="topk_mean_cosine",
        score_top_k=3,
        acceptance_policy_name="top1_margin_threshold",
        privacy_guard_name="diagonal_scale_clip_only",
    )

    assert payload.score_policy_name == "topk_mean_cosine"
    assert payload.score_top_k == 3
    assert payload.acceptance_policy_name == "top1_margin_threshold"
    assert payload.privacy_guard_name == "diagonal_scale_clip_only"


def test_training_objective_config_round_trips_policy_fields() -> None:
    config = TrainingObjectiveConfig.from_mapping(
        {
            "loss": "contrastive",
            "confidence_threshold": 0.65,
            "margin_threshold": 0.03,
            "score_policy_name": "topk_mean_cosine",
            "score_top_k": 2,
            "acceptance_policy_name": "top1_confidence_only",
            "privacy_guard_name": "noop",
            "temperature": 0.8,
        }
    )

    assert config.score_policy_name == "topk_mean_cosine"
    assert config.score_top_k == 2
    assert config.acceptance_policy_name == "top1_confidence_only"
    assert config.privacy_guard_name == "noop"
    assert config.extras == {"temperature": 0.8}
    assert config.to_mapping() == {
        "loss": "contrastive",
        "confidence_threshold": 0.65,
        "margin_threshold": 0.03,
        "score_policy_name": "topk_mean_cosine",
        "score_top_k": 2,
        "acceptance_policy_name": "top1_confidence_only",
        "privacy_guard_name": "noop",
        "temperature": 0.8,
    }
