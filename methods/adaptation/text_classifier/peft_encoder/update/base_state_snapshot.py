"""LoRA-classifier base state materialization cache identity."""

from __future__ import annotations

from typing import cast

from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LoraClassifierState,
)
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PeftClassifierState,
)

LORA_CLASSIFIER_BASE_STATE_MATERIALIZER_NAME = "lora_classifier_base_state.v1"


def lora_classifier_base_state_artifact_refs(
    state: LoraClassifierState | PeftClassifierState,
) -> tuple[tuple[str, str], ...]:
    """Base state cache key에 들어갈 artifact ref fingerprint를 반환한다."""

    if isinstance(state, PeftClassifierState):
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
    return (
        (
            "lora_adapter_artifact_ref",
            _ref_value(state.lora_adapter_artifact_ref),
        ),
        (
            "classifier_head_artifact_ref",
            _ref_value(state.classifier_head_artifact_ref),
        ),
    )


def _ref_value(value: str | None) -> str:
    return "" if value is None else cast(str, value)
