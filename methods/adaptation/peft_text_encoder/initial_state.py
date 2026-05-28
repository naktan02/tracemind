"""PEFT encoder classifier initial shared state construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Protocol

from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PeftClassifierState,
)


class PeftEncoderInitialStateConfig(Protocol):
    """Initial shared state 생성에 필요한 PEFT encoder classifier config surface."""

    artifact_format: str
    peft_adapter_artifact_ref: str | None
    classifier_head_artifact_ref: str | None

    def backbone_payload(self) -> Mapping[str, str | int]:
        """State payload에 기록할 backbone/tokenizer snapshot."""

    def lora_config_payload(self) -> Mapping[str, str | int | float | bool]:
        """State payload에 기록할 LoRA config snapshot."""

    def peft_adapter_config_payload(self) -> Mapping[str, object]:
        """State payload에 기록할 PEFT adapter config snapshot."""


def build_initial_peft_classifier_state(
    *,
    config: PeftEncoderInitialStateConfig,
    model_id: str,
    model_revision: str,
    training_scope: str,
    labels: Sequence[str],
    updated_at: datetime,
) -> PeftClassifierState:
    """simulation/runtime bootstrap용 PEFT-classifier initial state를 만든다."""

    return PeftClassifierState(
        model_id=model_id,
        model_revision=model_revision,
        training_scope=training_scope,
        updated_at=updated_at,
        backbone=dict(config.backbone_payload()),
        peft_adapter_config=dict(config.peft_adapter_config_payload()),
        label_schema=[str(label) for label in labels],
        peft_adapter_artifact_ref=config.peft_adapter_artifact_ref,
        classifier_head_artifact_ref=config.classifier_head_artifact_ref,
        artifact_format=config.artifact_format,
    )
