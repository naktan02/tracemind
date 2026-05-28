"""PEFT encoder base state materialization cache identity."""

from __future__ import annotations

from typing import cast

from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PeftClassifierState,
)

PEFT_ENCODER_BASE_STATE_MATERIALIZER_NAME = "peft_encoder_base_state.v1"


def peft_encoder_base_state_artifact_refs(
    state: PeftClassifierState,
) -> tuple[tuple[str, str], ...]:
    """Base state cache key에 들어갈 artifact ref fingerprint를 반환한다."""

    return (
        (
            "peft_adapter_artifact_ref",
            _ref_value(state.peft_adapter_artifact_ref),
        ),
        (
            "classifier_head_artifact_ref",
            _ref_value(state.classifier_head_artifact_ref),
        ),
    )


def _ref_value(value: str | None) -> str:
    return "" if value is None else cast(str, value)
