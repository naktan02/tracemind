"""SecureUpdateCodec seam tests."""

from __future__ import annotations

import pytest

from shared.src.contracts.training_contracts import (
    SecureAggregationConfigPayload,
    TrainingObjectiveConfigPayload,
    TrainingSelectionPolicyPayload,
    TrainingTask,
    make_training_update_envelope,
)
from shared.src.services.secure_update_codec import NoOpSecureUpdateCodec


def _task(*, secure_required: bool = False) -> TrainingTask:
    return TrainingTask(
        task_id="task_1",
        round_id="round_1",
        model_id="model",
        model_revision="rev_1",
        task_type="pseudo_label_self_training",
        training_scope="adapter_only",
        local_epochs=1,
        batch_size=1,
        learning_rate=0.1,
        max_steps=1,
        objective_config=TrainingObjectiveConfigPayload(
            training_backend_name="diagonal_scale_heuristic"
        ),
        selection_policy=TrainingSelectionPolicyPayload(),
        secure_aggregation=SecureAggregationConfigPayload(
            required=secure_required,
            aggregation_backend_name="secure_sum" if secure_required else None,
        ),
    )


def _envelope():
    return make_training_update_envelope(
        round_id="round_1",
        task_id="task_1",
        model_id="model",
        base_model_revision="rev_1",
        payload_ref="/tmp/update.json",
        example_count=1,
        client_metrics={"accepted_ratio": 1.0},
    )


def test_noop_secure_update_codec_passes_plaintext_when_not_required() -> None:
    codec = NoOpSecureUpdateCodec()
    task = _task()
    envelope = _envelope()

    assert (
        codec.encode_for_submission(envelope=envelope, training_task=task) is envelope
    )
    assert codec.decode_submission(envelope=envelope, training_task=task) is envelope


def test_noop_secure_update_codec_rejects_required_secure_aggregation() -> None:
    codec = NoOpSecureUpdateCodec()

    with pytest.raises(ValueError, match="cannot satisfy required"):
        codec.encode_for_submission(
            envelope=_envelope(),
            training_task=_task(secure_required=True),
        )
