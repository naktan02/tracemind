"""PEFT encoder round-runtime config surface for FL simulation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from methods.adaptation.peft_text_classifier.config import (
    PeftEncoderTrainingBackendConfig,
)


@dataclass(slots=True)
class FederatedPeftEncoderRuntimeConfig:
    """PEFT-backed classifier simulation bootstrap에 필요한 fixed scaffold snapshot."""

    training_backend_config: PeftEncoderTrainingBackendConfig
    artifact_format: str = "simulation_peft_classifier_state_ref"
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
            training_backend_config=PeftEncoderTrainingBackendConfig.from_mapping(
                {
                    key: value
                    for key, value in source.items()
                    if key not in _PEFT_ENCODER_RUNTIME_ARTIFACT_KEYS
                }
            ),
            artifact_format=artifact_format,
            peft_adapter_artifact_ref=_optional_str(
                source.get("peft_adapter_artifact_ref")
            ),
            classifier_head_artifact_ref=_optional_str(
                source.get("classifier_head_artifact_ref")
            ),
        )

    def backbone_payload(self) -> dict[str, str | int]:
        """shared PEFT-backed classifier state에 넣을 backbone/tokenizer snapshot."""

        return self.training_backend_config.to_backbone_payload()

    def lora_config_payload(self) -> dict[str, str | int | float | bool]:
        """PEFT adapter mechanism이 LoRA일 때 쓰는 config snapshot."""

        return self.training_backend_config.to_lora_config_payload()

    def peft_adapter_config_payload(self) -> dict[str, object]:
        """shared peft_classifier state에 넣을 PEFT mechanism config snapshot."""

        return self.training_backend_config.to_peft_adapter_config_payload()


def build_peft_encoder_round_runtime_payloads(
    round_runtime_mapping: Mapping[str, object],
) -> dict[str, object]:
    """Hydra round_runtime mapping에서 PEFT-backed classifier payload를 만든다."""

    payload_key = _runtime_payload_key(round_runtime_mapping)
    runtime_payloads = _required_runtime_payloads(round_runtime_mapping)
    peft_config = _required_runtime_payload_config(
        runtime_payloads=runtime_payloads,
        payload_key=payload_key,
    )
    return {
        payload_key: FederatedPeftEncoderRuntimeConfig.from_mapping(
            peft_config,
            default_artifact_format="simulation_peft_classifier_state_ref",
        )
    }


_PEFT_ENCODER_RUNTIME_ARTIFACT_KEYS = frozenset(
    {
        "artifact_format",
        "peft_adapter_artifact_ref",
        "classifier_head_artifact_ref",
    }
)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _runtime_payload_key(round_runtime_mapping: Mapping[str, object]) -> str:
    value = _optional_str(round_runtime_mapping.get("runtime_payload_key"))
    if value is None:
        value = _optional_str(round_runtime_mapping.get("update_family_name"))
    if value is None:
        raise ValueError(
            "round_runtime must define runtime_payload_key or update_family_name."
        )
    return value.lower().replace("-", "_")


def _required_runtime_payloads(
    round_runtime_mapping: Mapping[str, object],
) -> Mapping[str, object]:
    runtime_payloads = _optional_mapping(round_runtime_mapping.get("runtime_payloads"))
    if runtime_payloads is None:
        raise ValueError("round_runtime.runtime_payloads must define PEFT payloads.")
    return runtime_payloads


def _required_runtime_payload_config(
    *,
    runtime_payloads: Mapping[str, object],
    payload_key: str,
) -> Mapping[str, object]:
    config = _optional_mapping(runtime_payloads.get(payload_key))
    if config is None:
        raise ValueError(
            f"round_runtime.runtime_payloads must include payload key: {payload_key!r}."
        )
    return config


def _optional_mapping(value: object) -> Mapping[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("round_runtime PEFT payload config must be a mapping.")
    return value
