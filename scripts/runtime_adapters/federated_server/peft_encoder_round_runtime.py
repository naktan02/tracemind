"""PEFT encoder round-runtime config surface for FL simulation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from methods.adaptation.peft_text_classifier.config import (
    LoraClassifierTrainingBackendConfig,
)
from methods.adaptation.peft_text_classifier.runtime_family import (
    peft_encoder_runtime_payload,
)


@dataclass(slots=True)
class FederatedPeftEncoderRuntimeConfig:
    """PEFT-backed classifier simulation bootstrap에 필요한 fixed scaffold snapshot."""

    training_backend_config: LoraClassifierTrainingBackendConfig
    artifact_format: str = "simulation_peft_classifier_state_ref"
    lora_adapter_artifact_ref: str | None = None
    peft_adapter_artifact_ref: str | None = None
    classifier_head_artifact_ref: str | None = None

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, object],
        *,
        default_artifact_format: str = "simulation_peft_classifier_state_ref",
    ) -> "FederatedPeftEncoderRuntimeConfig":
        """Hydra round_runtime classifier mapping을 typed config로 해석한다."""

        artifact_format = str(
            source.get("artifact_format", default_artifact_format)
        ).strip()
        if not artifact_format:
            raise ValueError("round_runtime classifier artifact_format invalid.")
        return cls(
            training_backend_config=LoraClassifierTrainingBackendConfig.from_mapping(
                {
                    key: value
                    for key, value in source.items()
                    if key not in _PEFT_ENCODER_RUNTIME_ARTIFACT_KEYS
                }
            ),
            artifact_format=artifact_format,
            lora_adapter_artifact_ref=_optional_str(
                source.get("lora_adapter_artifact_ref")
            ),
            peft_adapter_artifact_ref=(
                _optional_str(source.get("peft_adapter_artifact_ref"))
                or _optional_str(source.get("lora_adapter_artifact_ref"))
            ),
            classifier_head_artifact_ref=_optional_str(
                source.get("classifier_head_artifact_ref")
            ),
        )

    def backbone_payload(self) -> dict[str, str | int]:
        """shared PEFT-backed classifier state에 넣을 backbone/tokenizer snapshot."""

        return self.training_backend_config.to_backbone_payload()

    def lora_config_payload(self) -> dict[str, str | int | float | bool]:
        """legacy lora_classifier state에 넣을 LoRA mechanism config snapshot."""

        return self.training_backend_config.to_lora_config_payload()

    def peft_adapter_config_payload(self) -> dict[str, object]:
        """shared peft_classifier state에 넣을 PEFT mechanism config snapshot."""

        return self.training_backend_config.to_peft_adapter_config_payload()


def build_peft_encoder_round_runtime_payloads(
    round_runtime_mapping: Mapping[str, object],
) -> dict[str, object]:
    """Hydra round_runtime mapping에서 PEFT-backed classifier payload를 만든다."""

    payloads: dict[str, object] = {}
    peft_config = _optional_mapping(round_runtime_mapping.get("peft_classifier"))
    if peft_config is not None:
        payloads["peft_classifier"] = FederatedPeftEncoderRuntimeConfig.from_mapping(
            peft_config,
            default_artifact_format="simulation_peft_classifier_state_ref",
        )
    return payloads


def resolve_peft_encoder_runtime_payload(round_runtime_config: Any) -> object | None:
    """PEFT-backed classifier family면 adapter family에 맞는 payload를 돌려준다."""

    return peft_encoder_runtime_payload(round_runtime_config)


_PEFT_ENCODER_RUNTIME_ARTIFACT_KEYS = frozenset(
    {
        "artifact_format",
        "lora_adapter_artifact_ref",
        "peft_adapter_artifact_ref",
        "classifier_head_artifact_ref",
    }
)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _optional_mapping(value: object) -> Mapping[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("round_runtime PEFT payload config must be a mapping.")
    return value
