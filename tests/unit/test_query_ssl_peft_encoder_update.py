"""Query SSL PEFT encoder update payload 조립 단위 검증."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from methods.adaptation.peft_text_classifier.config import (
    PEFT_ENCODER_DELTA_FORMAT_SERVER_UPLOADED,
    PeftEncoderTrainingBackendConfig,
)
from methods.adaptation.peft_text_classifier.update.query_ssl_update import (
    build_query_ssl_peft_encoder_update_payload,
)
from methods.adaptation.query_text_views.local_training_budget import (
    build_query_ssl_local_step_plan,
)
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PeftClassifierDelta,
)
from shared.src.contracts.common_types import TrainingTaskType
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.model_contracts import make_embedding_manifest
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
    TrainingTask,
)


def _row(query_id: str, label: str) -> LabeledQueryRow:
    return {
        "query_id": query_id,
        "text": f"{label} text",
        "raw_label_scheme": "unit",
        "raw_label": label,
        "mapped_label_4": label,
        "locale": "eng_Latn",
        "annotation_source": "unit",
        "approved_by": "unit",
        "created_at": "2026-04-01T00:00:00+00:00",
    }


def _training_task() -> TrainingTask:
    return TrainingTask(
        task_id="task_round_0001",
        round_id="round_0001",
        model_id="mxbai-peft-classifier",
        model_revision="sim_rev_0000",
        task_type=TrainingTaskType.PSEUDO_LABEL_SELF_TRAINING,
        training_scope="adapter_only",
        local_epochs=1,
        batch_size=2,
        learning_rate=1e-4,
        max_steps=3,
        objective_config=TrainingObjectiveConfig.from_mapping(
            {
                "training_backend_name": "peft_classifier_trainer",
                "confidence_threshold": 0.0,
                "margin_threshold": 0.0,
            }
        ),
        selection_policy=TrainingSelectionPolicy.from_mapping({"max_examples": 4}),
        gradient_clip_norm=1.0,
        min_required_examples=1,
    )


def test_query_ssl_peft_encoder_update_payload_uses_server_refs_without_inline() -> (
    None
):
    result = build_query_ssl_peft_encoder_update_payload(
        training_task=_training_task(),
        model_manifest=make_embedding_manifest(
            model_id="mxbai-peft-classifier",
            model_revision="sim_rev_0000",
            auxiliary_artifact_versions={"prototype_pack": "proto_v0"},
            artifact_ref="shared_adapter_state::sim_rev_0000",
        ),
        lora_config=PeftEncoderTrainingBackendConfig(
            delta_format=PEFT_ENCODER_DELTA_FORMAT_SERVER_UPLOADED
        ),
        labels=("anxiety", "normal"),
        labeled_rows=[_row("l1", "anxiety")],
        unlabeled_rows=[_row("u1", "normal"), _row("u2", "anxiety")],
        step_plan=build_query_ssl_local_step_plan(
            labeled_loader_steps=1,
            unlabeled_loader_steps=2,
            uses_labeled_batches=True,
            local_epochs=1,
            max_steps=3,
        ),
        history_record={
            "train_util_ratio": 0.5,
            "train_unsup_loss": 0.25,
            "ignored_text": "not numeric",
        },
        peft_parameter_deltas={"encoder.q_proj.lora_A": [0.1, -0.2]},
        classifier_head_weight_deltas={
            "anxiety": [0.3, -0.1],
            "normal": [-0.3, 0.1],
        },
        classifier_head_bias_deltas={"anxiety": 0.05, "normal": -0.05},
        created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        delta_format=PEFT_ENCODER_DELTA_FORMAT_SERVER_UPLOADED,
        peft_adapter_delta_artifact_ref="aggregation_artifact::round_0001/agent_01/peft",
        classifier_head_delta_artifact_ref=(
            "aggregation_artifact::round_0001/agent_01/head"
        ),
        include_inline_deltas=False,
    )

    payload = result.update_payload
    assert isinstance(payload, PeftClassifierDelta)
    assert payload.delta_format == PEFT_ENCODER_DELTA_FORMAT_SERVER_UPLOADED
    assert payload.peft_adapter_delta_artifact_ref == (
        "aggregation_artifact::round_0001/agent_01/peft"
    )
    assert payload.classifier_head_delta_artifact_ref == (
        "aggregation_artifact::round_0001/agent_01/head"
    )
    assert payload.peft_parameter_deltas is None
    assert payload.classifier_head_weight_deltas is None
    assert payload.classifier_head_bias_deltas == {}
    assert result.client_metrics["query_ssl_local_steps"] == 2.0
    assert result.client_metrics["pseudo_label_acceptance_rate"] == 0.5
    assert result.client_metrics["unlabeled_loss"] == 0.25


def test_query_ssl_peft_encoder_update_payload_requires_refs_for_artifact_mode() -> (
    None
):
    with pytest.raises(ValueError, match="requires adapter/head delta refs"):
        build_query_ssl_peft_encoder_update_payload(
            training_task=_training_task(),
            model_manifest=make_embedding_manifest(
                model_id="mxbai-peft-classifier",
                model_revision="sim_rev_0000",
                auxiliary_artifact_versions={"prototype_pack": "proto_v0"},
                artifact_ref="shared_adapter_state::sim_rev_0000",
            ),
            lora_config=PeftEncoderTrainingBackendConfig(),
            labels=("anxiety", "normal"),
            labeled_rows=[_row("l1", "anxiety")],
            unlabeled_rows=[_row("u1", "normal")],
            step_plan=build_query_ssl_local_step_plan(
                labeled_loader_steps=1,
                unlabeled_loader_steps=1,
                uses_labeled_batches=True,
                local_epochs=1,
                max_steps=1,
            ),
            history_record={},
            peft_parameter_deltas={"encoder.q_proj.lora_A": [0.1]},
            classifier_head_weight_deltas={
                "anxiety": [0.3, -0.1],
                "normal": [-0.3, 0.1],
            },
            classifier_head_bias_deltas={"anxiety": 0.05, "normal": -0.05},
            created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
            delta_format=PEFT_ENCODER_DELTA_FORMAT_SERVER_UPLOADED,
            include_inline_deltas=False,
        )
