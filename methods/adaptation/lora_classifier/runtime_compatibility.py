"""LoRA-classifier runtime/objective compatibility rules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from shared.src.contracts.training_contracts import TrainingObjectiveConfig

from .config import build_lora_classifier_training_backend_config


class LoraClassifierRuntimePayloadConfig(Protocol):
    """Runtime state와 local objective payload 비교에 필요한 config surface."""

    def backbone_payload(self) -> Mapping[str, str | int]:
        """State/update payload에 기록할 backbone/tokenizer snapshot."""

    def lora_config_payload(self) -> Mapping[str, str | int | float | bool]:
        """State/update payload에 기록할 LoRA config snapshot."""


def require_lora_classifier_runtime_matches_objective(
    *,
    runtime_config: LoraClassifierRuntimePayloadConfig,
    objective_config: TrainingObjectiveConfig | None,
) -> None:
    """bootstrap state와 local update가 같은 backbone/LoRA snapshot을 쓰게 한다."""

    objective_backend_config = build_lora_classifier_training_backend_config(
        objective_config
    )
    mismatches: dict[str, object] = {}
    if runtime_config.backbone_payload() != (
        objective_backend_config.to_backbone_payload()
    ):
        mismatches["backbone"] = {
            "round_runtime": runtime_config.backbone_payload(),
            "training_objective": objective_backend_config.to_backbone_payload(),
        }
    if runtime_config.lora_config_payload() != (
        objective_backend_config.to_lora_config_payload()
    ):
        mismatches["lora_config"] = {
            "round_runtime": runtime_config.lora_config_payload(),
            "training_objective": objective_backend_config.to_lora_config_payload(),
        }
    if mismatches:
        raise ValueError(
            "LoRA-classifier round_runtime.lora_classifier must match "
            f"training_task.objective shared payload config: {mismatches}."
        )
