"""LoRA-classifier initial shared state construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Protocol

from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LoraClassifierState,
)


class LoraClassifierInitialStateConfig(Protocol):
    """Initial shared state 생성에 필요한 LoRA-classifier config surface."""

    artifact_format: str
    lora_adapter_artifact_ref: str | None
    classifier_head_artifact_ref: str | None

    def backbone_payload(self) -> Mapping[str, str | int]:
        """State payload에 기록할 backbone/tokenizer snapshot."""

    def lora_config_payload(self) -> Mapping[str, str | int | float | bool]:
        """State payload에 기록할 LoRA config snapshot."""


def build_initial_lora_classifier_state(
    *,
    config: LoraClassifierInitialStateConfig,
    model_id: str,
    model_revision: str,
    training_scope: str,
    labels: Sequence[str],
    updated_at: datetime,
) -> LoraClassifierState:
    """simulation/runtime bootstrap용 LoRA-classifier initial state를 만든다."""

    return LoraClassifierState(
        model_id=model_id,
        model_revision=model_revision,
        training_scope=training_scope,
        updated_at=updated_at,
        backbone=dict(config.backbone_payload()),
        lora_config=dict(config.lora_config_payload()),
        label_schema=[str(label) for label in labels],
        lora_adapter_artifact_ref=config.lora_adapter_artifact_ref,
        classifier_head_artifact_ref=config.classifier_head_artifact_ref,
        artifact_format=config.artifact_format,
    )
