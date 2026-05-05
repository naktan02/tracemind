"""Training update submission contract tests."""

from __future__ import annotations

import pytest

from shared.src.contracts.adapter_contracts import make_diagonal_delta_payload
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


def _envelope(*, example_count: int = 2):
    return make_training_update_envelope(
        round_id="round_1",
        task_id="task_1",
        model_id="model",
        base_model_revision="rev_1",
        payload_ref="client-submission::update_1",
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
