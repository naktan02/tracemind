"""Shared adapter contract golden fixture tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from shared.src.contracts.adapter_contract_families.base import (
    CLASSIFIER_HEAD_STATE_V1,
    PEFT_CLASSIFIER_DELTA_V2,
)
from shared.src.contracts.adapter_contract_families.classifier_head import (
    LINEAR_CLASSIFIER_HEAD_KIND,
)
from shared.src.contracts.adapter_contract_families.registry import (
    parse_shared_adapter_state_payload,
    parse_shared_adapter_update_payload,
)
from shared.src.contracts.training_contracts import (
    TRAINING_UPDATE_SUBMISSION_V1,
    TrainingUpdateSubmissionPayload,
)

FIXTURE_DIR = Path("tests/contracts/fixtures")


def test_peft_classifier_delta_golden_fixture_round_trips_shape() -> None:
    fixture = _load_fixture("peft_classifier_delta.v2.json")

    parsed = parse_shared_adapter_update_payload(fixture)
    dumped = parsed.model_dump(mode="json")

    assert parsed.schema_version == PEFT_CLASSIFIER_DELTA_V2
    assert dumped == fixture


def test_classifier_head_state_golden_fixture_round_trips_shape() -> None:
    fixture = _load_fixture("classifier_head_state.v1.json")

    parsed = parse_shared_adapter_state_payload(fixture)
    dumped = parsed.model_dump(mode="json")

    assert parsed.schema_version == CLASSIFIER_HEAD_STATE_V1
    assert parsed.head_kind == LINEAR_CLASSIFIER_HEAD_KIND
    assert dumped == fixture


def test_classifier_head_state_defaults_missing_head_kind_to_linear() -> None:
    fixture = _load_fixture("classifier_head_state.v1.json")
    fixture.pop("head_kind")

    parsed = parse_shared_adapter_state_payload(fixture)

    assert parsed.head_kind == LINEAR_CLASSIFIER_HEAD_KIND


def test_classifier_head_state_rejects_unknown_head_kind() -> None:
    fixture = _load_fixture("classifier_head_state.v1.json")
    fixture["head_kind"] = "mlp"

    with pytest.raises(ValidationError):
        parse_shared_adapter_state_payload(fixture)


def test_training_update_submission_golden_fixture_round_trips_shape() -> None:
    fixture = _load_fixture("training_update_submission.v1.json")

    parsed = TrainingUpdateSubmissionPayload.model_validate(fixture)
    dumped = parsed.model_dump(mode="json")

    assert parsed.schema_version == TRAINING_UPDATE_SUBMISSION_V1
    assert dumped == fixture


def test_shared_adapter_golden_fixtures_reject_unknown_fields() -> None:
    fixture = _load_fixture("peft_classifier_delta.v2.json")
    fixture["unexpected_field"] = "must_fail"

    with pytest.raises(ValidationError):
        parse_shared_adapter_update_payload(fixture)


def _load_fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))
