"""PEFT text encoder/head initial shared state construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, cast

from methods.adaptation.initial_state import (
    SharedAdapterInitialStateRequest,
    register_shared_adapter_initial_state_builder,
)
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PEFT_CLASSIFIER_ADAPTER_KIND,
    PeftClassifierState,
)

from .config import PeftEncoderTrainingBackendConfig

PEFT_TEXT_ENCODER_UPDATE_FAMILY = "peft_text_encoder"


class PeftEncoderInitialStateConfig(Protocol):
    """Initial shared state 생성에 필요한 PEFT text encoder/head config surface."""

    artifact_format: str
    peft_adapter_artifact_ref: str | None
    classifier_head_artifact_ref: str | None

    def backbone_payload(self) -> Mapping[str, str | int]:
        """State payload에 기록할 backbone/tokenizer snapshot."""

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
    """simulation/runtime bootstrap용 PEFT text encoder/head initial state를 만든다."""

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


@dataclass(frozen=True, slots=True)
class DefaultPeftEncoderInitialStateConfig:
    """live/simulation 공통 PEFT initial state 기본 config surface."""

    training_backend_config: PeftEncoderTrainingBackendConfig = field(
        default_factory=PeftEncoderTrainingBackendConfig
    )
    artifact_format: str = "artifact_ref"
    peft_adapter_artifact_ref: str | None = None
    classifier_head_artifact_ref: str | None = None

    def backbone_payload(self) -> Mapping[str, str | int]:
        """State payload에 기록할 backbone/tokenizer snapshot."""

        return self.training_backend_config.to_backbone_payload()

    def peft_adapter_config_payload(self) -> Mapping[str, object]:
        """State payload에 기록할 PEFT adapter config snapshot."""

        return self.training_backend_config.to_peft_adapter_config_payload()


@register_shared_adapter_initial_state_builder(PEFT_CLASSIFIER_ADAPTER_KIND)
def build_initial_peft_text_encoder_shared_adapter_state(
    request: SharedAdapterInitialStateRequest,
) -> PeftClassifierState:
    """PEFT text encoder update family의 initial shared state를 만든다."""

    if _normalize_update_family(request.update_family_name) != (
        PEFT_TEXT_ENCODER_UPDATE_FAMILY
    ):
        raise ValueError(
            "PEFT classifier payload initial state requires peft_text_encoder "
            f"update family, got {request.update_family_name!r}."
        )
    if not request.labels:
        raise ValueError("PEFT text encoder initial state requires labels.")
    return build_initial_peft_classifier_state(
        config=_peft_initial_state_config_from_request(request),
        model_id=request.model_id,
        model_revision=request.model_revision,
        training_scope=request.training_scope,
        labels=request.labels,
        updated_at=request.updated_at or datetime.now(tz=timezone.utc),
    )


def _peft_initial_state_config_from_request(
    request: SharedAdapterInitialStateRequest,
) -> PeftEncoderInitialStateConfig:
    runtime_payload = _runtime_payload_for_update_family(request.round_runtime_config)
    if runtime_payload is None:
        return DefaultPeftEncoderInitialStateConfig()
    return _require_peft_initial_state_config(runtime_payload)


def _runtime_payload_for_update_family(round_runtime_config: object | None) -> object:
    if round_runtime_config is None:
        return None
    runtime_payload_reader = getattr(
        round_runtime_config,
        "runtime_payload_for_update_family",
        None,
    )
    if runtime_payload_reader is None:
        return None
    return runtime_payload_reader()


def _require_peft_initial_state_config(
    runtime_payload: object,
) -> PeftEncoderInitialStateConfig:
    required_attributes = (
        "artifact_format",
        "peft_adapter_artifact_ref",
        "classifier_head_artifact_ref",
        "backbone_payload",
        "peft_adapter_config_payload",
    )
    missing = [
        attribute
        for attribute in required_attributes
        if not hasattr(runtime_payload, attribute)
    ]
    if missing:
        raise TypeError(
            "PEFT text encoder initial state runtime payload is missing required "
            f"surface: {missing}."
        )
    return cast(PeftEncoderInitialStateConfig, runtime_payload)


def _normalize_update_family(update_family_name: str) -> str:
    return update_family_name.strip().lower().replace("-", "_")
