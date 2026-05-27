"""PEFT-backed classifier runtime/objective compatibility rules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from methods.adaptation.runtime_objective_compatibility import (
    register_runtime_objective_compatibility_validator,
)
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LORA_CLASSIFIER_ADAPTER_KIND,
)
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PEFT_CLASSIFIER_ADAPTER_KIND,
)
from shared.src.contracts.training_contracts import TrainingObjectiveConfig

from .config import (
    build_lora_classifier_training_backend_config,
    build_peft_classifier_training_backend_config,
)


@runtime_checkable
class LoraClassifierRuntimePayloadConfig(Protocol):
    """Runtime state와 local objective payload 비교에 필요한 config surface."""

    def backbone_payload(self) -> Mapping[str, str | int]:
        """State/update payload에 기록할 backbone/tokenizer snapshot."""

    def lora_config_payload(self) -> Mapping[str, str | int | float | bool]:
        """State/update payload에 기록할 LoRA config snapshot."""

    def peft_adapter_config_payload(self) -> Mapping[str, object]:
        """State/update payload에 기록할 PEFT adapter config snapshot."""


PeftEncoderRuntimePayloadConfig = LoraClassifierRuntimePayloadConfig


@register_runtime_objective_compatibility_validator(LORA_CLASSIFIER_ADAPTER_KIND)
def require_lora_classifier_runtime_matches_objective(
    *,
    runtime_config: object,
    objective_config: TrainingObjectiveConfig | None,
) -> None:
    """bootstrap state와 local update가 같은 backbone/LoRA snapshot을 쓰게 한다."""

    lora_runtime_config = _as_lora_classifier_runtime_config(runtime_config)
    objective_backend_config = build_lora_classifier_training_backend_config(
        objective_config
    )
    mismatches: dict[str, object] = {}
    if lora_runtime_config.backbone_payload() != (
        objective_backend_config.to_backbone_payload()
    ):
        mismatches["backbone"] = {
            "round_runtime": lora_runtime_config.backbone_payload(),
            "training_objective": objective_backend_config.to_backbone_payload(),
        }
    if lora_runtime_config.lora_config_payload() != (
        objective_backend_config.to_lora_config_payload()
    ):
        mismatches["lora_config"] = {
            "round_runtime": lora_runtime_config.lora_config_payload(),
            "training_objective": objective_backend_config.to_lora_config_payload(),
        }
    if mismatches:
        raise ValueError(
            "LoRA-classifier round_runtime.lora_classifier must match "
            f"training_task.objective shared payload config: {mismatches}."
        )


@register_runtime_objective_compatibility_validator(PEFT_CLASSIFIER_ADAPTER_KIND)
def require_peft_encoder_runtime_matches_objective(
    *,
    runtime_config: object,
    objective_config: TrainingObjectiveConfig | None,
) -> None:
    """v2 PEFT-classifier state와 local update config drift를 막는다."""

    runtime_payload_config = _as_peft_encoder_runtime_config(runtime_config)
    objective_backend_config = build_peft_classifier_training_backend_config(
        objective_config
    )
    mismatches: dict[str, object] = {}
    if runtime_payload_config.backbone_payload() != (
        objective_backend_config.to_backbone_payload()
    ):
        mismatches["backbone"] = {
            "round_runtime": runtime_payload_config.backbone_payload(),
            "training_objective": objective_backend_config.to_backbone_payload(),
        }
    if runtime_payload_config.peft_adapter_config_payload() != (
        objective_backend_config.to_peft_adapter_config_payload()
    ):
        mismatches["peft_adapter_config"] = {
            "round_runtime": runtime_payload_config.peft_adapter_config_payload(),
            "training_objective": (
                objective_backend_config.to_peft_adapter_config_payload()
            ),
        }
    if mismatches:
        raise ValueError(
            "PEFT-classifier round_runtime.peft_classifier must match "
            f"training_task.objective shared payload config: {mismatches}."
        )


def _as_lora_classifier_runtime_config(
    runtime_config: object,
) -> LoraClassifierRuntimePayloadConfig:
    return _as_peft_encoder_runtime_config(runtime_config)


def _as_peft_encoder_runtime_config(
    runtime_config: object,
) -> PeftEncoderRuntimePayloadConfig:
    if not isinstance(runtime_config, PeftEncoderRuntimePayloadConfig):
        raise TypeError(
            "PEFT-backed classifier runtime compatibility requires "
            "backbone_payload(), lora_config_payload(), and "
            "peft_adapter_config_payload()."
        )
    return runtime_config


require_peft_classifier_runtime_matches_objective = (
    require_peft_encoder_runtime_matches_objective
)
