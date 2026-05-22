"""Training update submission contract tests."""

from __future__ import annotations

import pytest

from shared.src.contracts.adapter_contract_families.diagonal_scale import (
    DIAGONAL_SCALE_UPDATE_PAYLOAD_FORMAT,
)
from shared.src.contracts.adapter_contract_families.factories import (
    make_diagonal_delta_payload,
    make_lora_classifier_delta_payload,
)
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LORA_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
)
from shared.src.contracts.training_contracts import (
    TrainingUpdateSubmissionPayload,
    make_training_update_envelope,
    make_training_update_submission,
)


def _update_payload():
    return make_diagonal_delta_payload(
        model_id="model",
        base_model_revision="rev_1",
        dimension_deltas=[0.1, 0.2],
        example_count=2,
        mean_confidence=0.8,
    )


def _envelope(
    *,
    example_count: int = 2,
    payload_format: str = DIAGONAL_SCALE_UPDATE_PAYLOAD_FORMAT,
):
    return make_training_update_envelope(
        round_id="round_1",
        task_id="task_1",
        model_id="model",
        base_model_revision="rev_1",
        payload_ref="client-submission::update_1",
        payload_format=payload_format,
        example_count=example_count,
        client_metrics={"accepted_ratio": 1.0},
    )


def test_training_update_submission_parses_inline_update_payload() -> None:
    submission = make_training_update_submission(
        envelope=_envelope(),
        update_payload=_update_payload(),
    )

    parsed = TrainingUpdateSubmissionPayload.model_validate_json(
        submission.model_dump_json()
    )

    assert parsed.envelope.update_id == submission.envelope.update_id
    assert parsed.update_payload.model_id == "model"
    assert parsed.update_payload.example_count == 2


def test_training_update_submission_rejects_misaligned_payload() -> None:
    with pytest.raises(ValueError, match="example_count"):
        make_training_update_submission(
            envelope=_envelope(example_count=3),
            update_payload=_update_payload(),
        )


def test_training_update_submission_rejects_payload_format_mismatch() -> None:
    with pytest.raises(ValueError, match="payload_format"):
        make_training_update_submission(
            envelope=_envelope(payload_format=LORA_CLASSIFIER_UPDATE_PAYLOAD_FORMAT),
            update_payload=_update_payload(),
        )


def test_training_update_submission_parses_lora_classifier_update_payload() -> None:
    update_payload = make_lora_classifier_delta_payload(
        model_id="model",
        base_model_revision="rev_1",
        backbone={
            "backbone_model_id": "mixedbread-ai/mxbai-embed-large-v1",
            "backbone_revision": "main",
            "tokenizer_model_id": "mixedbread-ai/mxbai-embed-large-v1",
            "tokenizer_revision": "main",
            "pooling": "mean",
            "max_length": 256,
            "task_prefix": "",
        },
        lora_config={
            "peft_adapter_name": "lora",
            "rank": 8,
            "alpha": 16,
            "dropout": 0.1,
            "bias": "none",
            "target_modules": "all-linear",
            "use_rslora": False,
        },
        label_schema=["anxiety", "normal"],
        example_count=2,
        lora_delta_artifact_ref="client-submission::lora_delta",
    )
    submission = make_training_update_submission(
        envelope=_envelope(payload_format=LORA_CLASSIFIER_UPDATE_PAYLOAD_FORMAT),
        update_payload=update_payload,
    )

    parsed = TrainingUpdateSubmissionPayload.model_validate_json(
        submission.model_dump_json()
    )

    assert parsed.update_payload.adapter_kind == "lora_classifier"
    assert parsed.envelope.payload_format == "lora_classifier_update"
