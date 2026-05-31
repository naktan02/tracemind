"""Training update submission contract tests."""

from __future__ import annotations

import pytest

from shared.src.contracts.adapter_contract_families.classifier_head import (
    CLASSIFIER_HEAD_UPDATE_PAYLOAD_FORMAT,
    LINEAR_CLASSIFIER_HEAD_KIND,
    ClassifierHeadAdapterUpdatePayload,
)
from shared.src.contracts.adapter_contract_families.factories import (
    make_classifier_head_delta_payload,
)
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PEFT_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
)
from shared.src.contracts.common_types import TrainingScope
from shared.src.contracts.training_contracts import (
    TrainingUpdateSubmissionPayload,
    make_training_update_envelope,
    make_training_update_submission,
)


def _update_payload():
    return make_classifier_head_delta_payload(
        model_id="model",
        base_model_revision="rev_1",
        training_scope=TrainingScope.HEAD_ONLY,
        label_weight_deltas={"anxiety": [0.1, 0.2], "normal": [-0.1, -0.2]},
        label_bias_deltas={"anxiety": 0.01, "normal": -0.01},
        example_count=2,
        mean_confidence=0.8,
    )


def _envelope(
    *,
    example_count: int = 2,
    payload_format: str = CLASSIFIER_HEAD_UPDATE_PAYLOAD_FORMAT,
    training_scope: TrainingScope = TrainingScope.HEAD_ONLY,
):
    return make_training_update_envelope(
        round_id="round_1",
        task_id="task_1",
        model_id="model",
        base_model_revision="rev_1",
        training_scope=training_scope,
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
    assert parsed.update_payload.head_kind == LINEAR_CLASSIFIER_HEAD_KIND


def test_classifier_head_update_rejects_unknown_head_kind() -> None:
    payload = _update_payload().model_dump(mode="json")
    payload["head_kind"] = "mlp"

    with pytest.raises(ValueError, match="head_kind"):
        ClassifierHeadAdapterUpdatePayload.model_validate(payload)


def test_training_update_submission_rejects_misaligned_payload() -> None:
    with pytest.raises(ValueError, match="example_count"):
        make_training_update_submission(
            envelope=_envelope(example_count=3),
            update_payload=_update_payload(),
        )


def test_training_update_submission_rejects_payload_format_mismatch() -> None:
    with pytest.raises(ValueError, match="payload_format"):
        make_training_update_submission(
            envelope=_envelope(payload_format=PEFT_CLASSIFIER_UPDATE_PAYLOAD_FORMAT),
            update_payload=_update_payload(),
        )


def test_training_update_submission_parses_peft_classifier_update_payload() -> None:
    from shared.src.contracts.adapter_contract_families.factories import (
        make_peft_classifier_delta_payload,
    )

    update_payload = make_peft_classifier_delta_payload(
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
        peft_adapter_config={
            "peft_adapter_name": "lora",
            "parameters": {
                "rank": 8,
                "alpha": 16,
                "dropout": 0.1,
                "bias": "none",
                "target_modules": "all-linear",
                "use_rslora": False,
            },
        },
        label_schema=["anxiety", "normal"],
        example_count=2,
        peft_adapter_delta_artifact_ref="client-submission::peft_delta",
    )
    submission = make_training_update_submission(
        envelope=_envelope(
            payload_format=PEFT_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
            training_scope=TrainingScope.ADAPTER_ONLY,
        ),
        update_payload=update_payload,
    )

    parsed = TrainingUpdateSubmissionPayload.model_validate_json(
        submission.model_dump_json()
    )

    assert parsed.update_payload.adapter_kind == "peft_classifier"
    assert parsed.envelope.payload_format == "peft_classifier_update"
