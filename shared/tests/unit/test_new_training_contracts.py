from __future__ import annotations

import pytest

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
    SecureAggregationConfig,
    SecureAggregationConfigPayload,
    SecureAggregationSubmissionPayload,
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
        training_scope=TrainingScope.ADAPTER_ONLY,
        training_enabled=True,
        compatible_task_types=[TrainingTaskType.PSEUDO_LABEL_SELF_TRAINING],
    )

    assert payload.training_enabled is True
    assert payload.training_scope == TrainingScope.ADAPTER_ONLY
    assert "prototype_version" not in payload.model_dump(mode="json")


def test_model_manifest_payload_keeps_prototype_only_as_auxiliary(
    make_model_manifest_payload,
) -> None:
    payload = make_model_manifest_payload(
        auxiliary_artifact_versions={"prototype_pack": "proto_001"},
    )

    assert payload.auxiliary_artifact_versions == {"prototype_pack": "proto_001"}
    assert "prototype_version" not in payload.model_dump(mode="json")


def test_model_manifest_payload_migrates_legacy_prototype_version(
    make_model_manifest_payload,
) -> None:
    payload = make_model_manifest_payload(
        prototype_version="proto_001",
        translation_model_id="legacy-translator",
        translation_model_revision="legacy-rev",
    )

    assert payload.auxiliary_artifact_versions == {"prototype_pack": "proto_001"}
    dumped = payload.model_dump(mode="json")
    assert "prototype_version" not in dumped
    assert "translation_model_id" not in dumped
    assert "translation_model_revision" not in dumped


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
        objective_config=TrainingObjectiveConfigPayload(
            training_backend_name="contrastive"
        ),
        secure_aggregation=SecureAggregationConfigPayload(required=True),
    )
    update = make_training_update_envelope_payload(
        update_id="update_001",
        round_id="round_001",
        task_id="task_001",
        model_id="tracemind-embed",
        base_model_revision="rev_001",
        training_scope=TrainingScope.ADAPTER_ONLY,
        payload_ref="updates/update_001",
        payload_format="classifier_head_update",
        example_count=12,
        client_metrics={"mean_loss": 0.5},
        secure_aggregation=SecureAggregationSubmissionPayload(
            aggregation_backend_name="he_ckks"
        ),
    )

    assert task.round_id == update.round_id
    assert update.base_model_revision == "rev_001"
    assert task.selection_policy.max_examples == 32
    assert task.secure_aggregation.required is True
    assert update.secure_aggregation is not None
    assert update.secure_aggregation.aggregation_backend_name == "he_ckks"


def test_federated_ssl_method_local_step_is_canonical_training_task_type(
    make_training_task_payload,
) -> None:
    payload = make_training_task_payload(
        task_type=TrainingTaskType.FEDERATED_SSL_METHOD_LOCAL_STEP,
    )

    assert payload.task_type == TrainingTaskType.FEDERATED_SSL_METHOD_LOCAL_STEP
    assert (
        payload.model_dump(mode="json")["task_type"]
        == "federated_ssl_method_local_step"
    )


def test_training_task_payload_migrates_legacy_fedmatch_task_type(
    make_training_task_payload,
) -> None:
    payload = make_training_task_payload(task_type="fedmatch_local_step")

    assert payload.task_type == TrainingTaskType.FEDERATED_SSL_METHOD_LOCAL_STEP
    assert (
        payload.model_dump(mode="json")["task_type"]
        == "federated_ssl_method_local_step"
    )


def test_training_update_envelope_accepts_custom_payload_format(
    make_training_update_envelope_payload,
) -> None:
    payload = make_training_update_envelope_payload(payload_format="lora_update")

    assert payload.payload_format == "lora_update"


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
        training_backend_name="contrastive",
        algorithm_profile_name="prototype_pseudo_label_v1",
        loss_name="cross_entropy",
        confidence_threshold=0.7,
        margin_threshold=0.05,
        example_generation_backend_name="prototype_rescore",
        evidence_backend_name="prototype_similarity_evidence",
        scorer_backend_name="prototype_similarity",
        score_policy_name="topk_mean_cosine",
        score_top_k=3,
        pseudo_label_algorithm_name="top1_margin_threshold",
        acceptance_policy_name="top1_margin_threshold",
        privacy_guard_name="clip_only",
    )

    assert payload.training_backend_name == "contrastive"
    assert payload.algorithm_profile_name == "prototype_pseudo_label_v1"
    assert payload.loss_name == "cross_entropy"
    assert payload.example_generation_backend_name == "prototype_rescore"
    assert payload.evidence_backend_name == "prototype_similarity_evidence"
    assert payload.scorer_backend_name == "prototype_similarity"
    assert payload.score_policy_name == "topk_mean_cosine"
    assert payload.score_top_k == 3
    assert payload.pseudo_label_algorithm_name == "top1_margin_threshold"
    assert payload.acceptance_policy_name == "top1_margin_threshold"
    assert payload.privacy_guard_name == "clip_only"


def test_training_objective_config_round_trips_policy_fields() -> None:
    config = TrainingObjectiveConfig.from_mapping(
        {
            "training_backend_name": "peft_classifier_trainer",
            "algorithm_profile_name": "prototype_pseudo_label_v1",
            "loss_name": "cross_entropy",
            "confidence_threshold": 0.65,
            "margin_threshold": 0.03,
            "example_generation_backend_name": "prototype_rescore",
            "evidence_backend_name": "prototype_similarity_evidence",
            "scorer_backend_name": "prototype_similarity",
            "score_policy_name": "topk_mean_cosine",
            "score_top_k": 2,
            "pseudo_label_algorithm_name": "top1_confidence_only",
            "acceptance_policy_name": "top1_confidence_only",
            "privacy_guard_name": "noop",
            "temperature": 0.8,
        }
    )

    assert config.training_backend_name == "peft_classifier_trainer"
    assert config.algorithm_profile_name == "prototype_pseudo_label_v1"
    assert config.loss_name == "cross_entropy"
    assert config.example_generation_backend_name == "prototype_rescore"
    assert config.evidence_backend_name == "prototype_similarity_evidence"
    assert config.scorer_backend_name == "prototype_similarity"
    assert config.score_policy_name == "topk_mean_cosine"
    assert config.score_top_k == 2
    assert config.pseudo_label_algorithm_name == "top1_confidence_only"
    assert config.acceptance_policy_name == "top1_confidence_only"
    assert config.privacy_guard_name == "noop"
    assert config.extras == {"temperature": 0.8}
    assert config.to_mapping() == {
        "training_backend_name": "peft_classifier_trainer",
        "algorithm_profile_name": "prototype_pseudo_label_v1",
        "loss_name": "cross_entropy",
        "confidence_threshold": 0.65,
        "margin_threshold": 0.03,
        "example_generation_backend_name": "prototype_rescore",
        "evidence_backend_name": "prototype_similarity_evidence",
        "scorer_backend_name": "prototype_similarity",
        "score_policy_name": "topk_mean_cosine",
        "score_top_k": 2,
        "pseudo_label_algorithm_name": "top1_confidence_only",
        "acceptance_policy_name": "top1_confidence_only",
        "privacy_guard_name": "noop",
        "temperature": 0.8,
    }


def test_training_objective_config_normalizes_nested_component_extras() -> None:
    config = TrainingObjectiveConfig.from_mapping(
        {
            "training_backend_name": "peft_classifier_trainer",
            "evidence_backend": {"temperature": 0.7},
            "query_ssl": {
                "method_name": "fixmatch_usb_v1",
                "algorithm_name": "fixmatch",
            },
            "peft_classifier": {
                "delta_format": "server_uploaded_artifact_ref",
                "rank": 8,
            },
        }
    )

    assert config.extras == {
        "evidence_backend.temperature": 0.7,
        "query_ssl.method_name": "fixmatch_usb_v1",
        "query_ssl.algorithm_name": "fixmatch",
        "peft_classifier.delta_format": "server_uploaded_artifact_ref",
        "peft_classifier.rank": 8,
    }
    assert config.get_component_extras("query_ssl") == {
        "method_name": "fixmatch_usb_v1",
        "algorithm_name": "fixmatch",
    }
    assert config.get_component_extras("peft_classifier") == {
        "delta_format": "server_uploaded_artifact_ref",
        "rank": 8,
    }


def test_training_objective_config_preserves_algorithm_profile_without_expansion() -> (
    None
):
    config = TrainingObjectiveConfig.from_mapping(
        {
            "training_backend_name": "peft_classifier_trainer",
            "algorithm_profile_name": "prototype_top1_confidence_v1",
        }
    )

    assert config.algorithm_profile_name == "prototype_top1_confidence_v1"
    assert config.training_backend_name == "peft_classifier_trainer"
    assert config.example_generation_backend_name is None
    assert config.evidence_backend_name is None
    assert config.pseudo_label_algorithm_name is None
    assert config.acceptance_policy_name is None
    assert config.margin_threshold is None


def test_training_objective_config_does_not_expand_unknown_algorithm_profile() -> None:
    config = TrainingObjectiveConfig.from_mapping(
        {
            "training_backend_name": "peft_classifier_trainer",
            "algorithm_profile_name": "custom_pseudo_label_v1",
        }
    )

    assert config.algorithm_profile_name == "custom_pseudo_label_v1"
    assert config.training_backend_name == "peft_classifier_trainer"
    assert config.example_generation_backend_name is None
    assert config.scorer_backend_name is None
    assert config.pseudo_label_algorithm_name is None
    assert config.privacy_guard_name is None


def test_training_objective_config_requires_training_backend_name() -> None:
    with pytest.raises(ValueError, match="training_backend_name is required"):
        TrainingObjectiveConfig.from_mapping(
            {"algorithm_profile_name": "prototype_top1_confidence_v1"}
        )


def test_training_objective_config_from_mapping_keeps_legacy_algorithm_fallback() -> (
    None
):
    config = TrainingObjectiveConfig.from_mapping(
        {
            "training_backend_name": "peft_classifier_trainer",
            "acceptance_policy_name": "top1_margin_threshold",
        }
    )

    assert config.pseudo_label_algorithm_name == "top1_margin_threshold"
    assert config.acceptance_policy_name == "top1_margin_threshold"


def test_training_objective_config_accepts_legacy_loss_alias() -> None:
    payload = TrainingObjectiveConfigPayload(loss="legacy_backend")

    assert payload.training_backend_name == "legacy_backend"
    assert payload.loss == "legacy_backend"


def test_training_task_payload_accepts_legacy_secure_aggregation_required(
    make_training_task_payload,
) -> None:
    payload = make_training_task_payload(secure_aggregation_required=True)

    assert payload.secure_aggregation.required is True
    assert payload.secure_aggregation_required is True


def test_secure_aggregation_config_round_trips_metadata() -> None:
    config = SecureAggregationConfig.from_mapping(
        {
            "required": True,
            "aggregation_backend_name": "he_ckks",
            "encryption_scheme_name": "ckks",
            "key_ref": "keys/tenant_a_pubkey",
            "ciphertext_format": "ckks_vector_v1",
            "poly_modulus_degree": 8192,
        }
    )

    assert config.required is True
    assert config.aggregation_backend_name == "he_ckks"
    assert config.encryption_scheme_name == "ckks"
    assert config.key_ref == "keys/tenant_a_pubkey"
    assert config.ciphertext_format == "ckks_vector_v1"
    assert config.extras == {"poly_modulus_degree": 8192}
    assert config.to_mapping() == {
        "required": True,
        "aggregation_backend_name": "he_ckks",
        "encryption_scheme_name": "ckks",
        "key_ref": "keys/tenant_a_pubkey",
        "ciphertext_format": "ckks_vector_v1",
        "poly_modulus_degree": 8192,
    }
