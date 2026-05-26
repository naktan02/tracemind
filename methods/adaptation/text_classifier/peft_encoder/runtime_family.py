"""PEFT-backed classifier runtime family helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime
from typing import Protocol

from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LORA_CLASSIFIER_ADAPTER_KIND,
    LoraClassifierState,
)
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PEFT_CLASSIFIER_ADAPTER_KIND,
    PEFT_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
    PeftClassifierState,
)
from shared.src.contracts.training_contracts import TrainingObjectiveConfig
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState

from .config import (
    PEFT_CLASSIFIER_FAMILY_EXTRA_SCOPE,
    PEFT_CLASSIFIER_PAYLOAD_ADAPTER_KIND,
    PEFT_CLASSIFIER_TRAINING_BACKEND_EXTRA_SCOPE,
    PEFT_CLASSIFIER_TRAINING_BACKEND_NAME,
    LoraClassifierTrainingBackendConfig,
    build_lora_classifier_training_backend_config,
)
from .initial_state import (
    LoraClassifierInitialStateConfig,
    build_initial_lora_classifier_state,
    build_initial_peft_classifier_state,
)
from .training_backend import LoraClassifierTrainingBackend

PeftEncoderState = LoraClassifierState | PeftClassifierState
PEFT_ENCODER_ADAPTER_KINDS = (
    PEFT_CLASSIFIER_ADAPTER_KIND,
    LORA_CLASSIFIER_ADAPTER_KIND,
)


class PeftEncoderRoundRuntimeConfig(Protocol):
    """PEFT-backed classifier runtime payload를 가진 round runtime surface."""

    adapter_family_name: str
    lora_classifier: LoraClassifierInitialStateConfig | None
    peft_classifier: LoraClassifierInitialStateConfig | None


def is_peft_encoder_adapter_family(adapter_family_name: object) -> bool:
    """runtime adapter family가 PEFT-backed classifier 계열인지 판정한다."""

    return _normalize_adapter_family_name(adapter_family_name) in (
        PEFT_ENCODER_ADAPTER_KINDS
    )


def peft_encoder_runtime_payload(
    round_runtime_config: PeftEncoderRoundRuntimeConfig,
) -> LoraClassifierInitialStateConfig | None:
    """adapter family 이름에 맞는 runtime payload를 반환한다."""

    adapter_family_name = _normalize_adapter_family_name(
        round_runtime_config.adapter_family_name
    )
    if adapter_family_name == PEFT_CLASSIFIER_ADAPTER_KIND:
        return (
            round_runtime_config.peft_classifier or round_runtime_config.lora_classifier
        )
    if adapter_family_name == LORA_CLASSIFIER_ADAPTER_KIND:
        return round_runtime_config.lora_classifier
    return None


def build_initial_peft_encoder_state(
    *,
    round_runtime_config: PeftEncoderRoundRuntimeConfig,
    model_id: str,
    model_revision: str,
    training_scope: str,
    labels: Sequence[str],
    updated_at: datetime,
) -> SharedAdapterState | None:
    """지원 family면 initial shared state를 만들고, 아니면 None을 반환한다."""

    adapter_family_name = _normalize_adapter_family_name(
        round_runtime_config.adapter_family_name
    )
    runtime_payload = peft_encoder_runtime_payload(round_runtime_config)
    if adapter_family_name == LORA_CLASSIFIER_ADAPTER_KIND:
        if runtime_payload is None:
            raise ValueError(
                "lora_classifier round runtime requires lora_classifier "
                "bootstrap config."
            )
        return build_initial_lora_classifier_state(
            config=runtime_payload,
            model_id=model_id,
            model_revision=model_revision,
            training_scope=training_scope,
            labels=labels,
            updated_at=updated_at,
        )
    if adapter_family_name == PEFT_CLASSIFIER_ADAPTER_KIND:
        if runtime_payload is None:
            raise ValueError(
                "peft_classifier round runtime requires peft_classifier "
                "bootstrap config."
            )
        return build_initial_peft_classifier_state(
            config=runtime_payload,
            model_id=model_id,
            model_revision=model_revision,
            training_scope=training_scope,
            labels=labels,
            updated_at=updated_at,
        )
    return None


def build_training_backend_config_for_peft_encoder_state(
    *,
    active_adapter_state: PeftEncoderState,
    objective_config: TrainingObjectiveConfig | None,
) -> LoraClassifierTrainingBackendConfig:
    """active state family에 맞는 local trainer config를 만든다."""

    if isinstance(active_adapter_state, PeftClassifierState):
        config = build_lora_classifier_training_backend_config(
            objective_config,
            family_extra_scope=PEFT_CLASSIFIER_FAMILY_EXTRA_SCOPE,
            training_backend_extra_scope=PEFT_CLASSIFIER_TRAINING_BACKEND_EXTRA_SCOPE,
        )
        return replace(
            config,
            payload_adapter_kind=PEFT_CLASSIFIER_PAYLOAD_ADAPTER_KIND,
            artifact_ref_prefix="agent-local://peft_classifier"
            if config.artifact_ref_prefix == "agent-local://lora_classifier"
            else config.artifact_ref_prefix,
        )
    return build_lora_classifier_training_backend_config(objective_config)


def build_training_backend_for_peft_encoder_state(
    *,
    active_adapter_state: PeftEncoderState,
    objective_config: TrainingObjectiveConfig | None,
) -> LoraClassifierTrainingBackend:
    """active state family에 맞는 Query SSL local backend를 만든다."""

    config = build_training_backend_config_for_peft_encoder_state(
        active_adapter_state=active_adapter_state,
        objective_config=objective_config,
    )
    if isinstance(active_adapter_state, PeftClassifierState):
        return LoraClassifierTrainingBackend(
            backend_name=PEFT_CLASSIFIER_TRAINING_BACKEND_NAME,
            payload_format=PEFT_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
            adapter_kind=PEFT_CLASSIFIER_ADAPTER_KIND,
            config=config,
        )
    return LoraClassifierTrainingBackend(config=config)


def _normalize_adapter_family_name(adapter_family_name: object) -> str:
    return str(adapter_family_name).strip().lower().replace("-", "_")
