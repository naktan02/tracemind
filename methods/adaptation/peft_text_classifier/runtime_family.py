"""PEFT-backed classifier runtime family helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Protocol, cast

from shared.src.contracts.adapter_contract_families.lora_classifier import (
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
    PEFT_CLASSIFIER_TRAINING_BACKEND_NAME,
    PeftEncoderTrainingBackendConfig,
    build_legacy_lora_classifier_training_backend_config,
    build_peft_classifier_training_backend_config,
)
from .initial_state import (
    LoraClassifierInitialStateConfig,
    build_initial_peft_classifier_state,
)
from .training_backend import PeftEncoderTrainingBackend

PeftEncoderState = LoraClassifierState | PeftClassifierState
PEFT_TEXT_CLASSIFIER_UPDATE_FAMILY = "peft_text_classifier"
PEFT_ENCODER_UPDATE_FAMILIES = (PEFT_TEXT_CLASSIFIER_UPDATE_FAMILY,)


class PeftEncoderRoundRuntimeConfig(Protocol):
    """PEFT-backed classifier runtime payload를 가진 round runtime surface."""

    update_family_name: str

    def runtime_payload_for_update_family(self) -> object | None:
        """update family config가 해석한 runtime payload를 반환한다."""


def is_peft_encoder_update_family(update_family_name: object) -> bool:
    """runtime update family가 PEFT-backed classifier 계열인지 판정한다."""

    return _normalize_family_name(update_family_name) in PEFT_ENCODER_UPDATE_FAMILIES


def peft_encoder_runtime_payload(
    round_runtime_config: PeftEncoderRoundRuntimeConfig,
) -> LoraClassifierInitialStateConfig | None:
    """PEFT-backed update family runtime payload를 반환한다."""

    update_family_name = _normalize_family_name(round_runtime_config.update_family_name)
    if update_family_name == PEFT_TEXT_CLASSIFIER_UPDATE_FAMILY:
        runtime_payload = round_runtime_config.runtime_payload_for_update_family()
        if runtime_payload is None:
            return None
        return _require_peft_encoder_runtime_payload(runtime_payload)
    return None


def build_initial_peft_encoder_state(
    *,
    round_runtime_config: PeftEncoderRoundRuntimeConfig,
    model_id: str,
    model_revision: str,
    training_scope: str,
    embedding_dim: int,
    labels: Sequence[str],
    updated_at: datetime,
) -> SharedAdapterState | None:
    """지원 family면 initial shared state를 만들고, 아니면 None을 반환한다."""

    _ = embedding_dim
    update_family_name = _normalize_family_name(round_runtime_config.update_family_name)
    runtime_payload = peft_encoder_runtime_payload(round_runtime_config)
    if update_family_name == PEFT_TEXT_CLASSIFIER_UPDATE_FAMILY:
        if runtime_payload is None:
            raise ValueError(
                "peft_text_classifier round runtime requires configured runtime "
                "payload."
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


def build_peft_text_classifier_composition_slug(
    *,
    round_runtime_mapping: Mapping[str, object],
    update_family_name: str,
) -> str:
    """PEFT text classifier run layout slug를 family owner 쪽에서 만든다."""

    family_name = _normalize_family_name(update_family_name)
    runtime_payload = _runtime_payload_mapping(
        round_runtime_mapping=round_runtime_mapping,
        update_family_name=update_family_name,
    )
    if not isinstance(runtime_payload, Mapping):
        return family_name
    peft_adapter_name = str(runtime_payload.get("peft_adapter_name") or "").strip()
    if not peft_adapter_name:
        return family_name
    normalized_adapter_name = peft_adapter_name.lower().replace("-", "_")
    if family_name.startswith(f"{normalized_adapter_name}_"):
        return family_name
    if family_name.endswith(f"_{normalized_adapter_name}"):
        return family_name
    return f"{family_name}_{peft_adapter_name}"


def _runtime_payload_mapping(
    *,
    round_runtime_mapping: Mapping[str, object],
    update_family_name: str,
) -> Mapping[str, object] | None:
    runtime_payloads = round_runtime_mapping.get("runtime_payloads")
    if not isinstance(runtime_payloads, Mapping):
        return None
    payload_key = str(
        round_runtime_mapping.get("runtime_payload_key") or update_family_name
    )
    normalized_payload_key = payload_key.strip().lower().replace("-", "_")
    payload = runtime_payloads.get(normalized_payload_key)
    if isinstance(payload, Mapping):
        return payload
    for key, value in runtime_payloads.items():
        if str(key).strip().lower().replace("-", "_") == normalized_payload_key:
            return value if isinstance(value, Mapping) else None
    return None


def build_training_backend_config_for_peft_encoder_state(
    *,
    active_adapter_state: PeftEncoderState,
    objective_config: TrainingObjectiveConfig | None,
) -> PeftEncoderTrainingBackendConfig:
    """active state family에 맞는 local trainer config를 만든다."""

    if isinstance(active_adapter_state, PeftClassifierState):
        return build_peft_classifier_training_backend_config(objective_config)
    return build_legacy_lora_classifier_training_backend_config(objective_config)


def build_training_backend_for_peft_encoder_state(
    *,
    active_adapter_state: PeftEncoderState,
    objective_config: TrainingObjectiveConfig | None,
) -> PeftEncoderTrainingBackend:
    """active state family에 맞는 Query SSL local backend를 만든다."""

    config = build_training_backend_config_for_peft_encoder_state(
        active_adapter_state=active_adapter_state,
        objective_config=objective_config,
    )
    if isinstance(active_adapter_state, PeftClassifierState):
        return PeftEncoderTrainingBackend(
            backend_name=PEFT_CLASSIFIER_TRAINING_BACKEND_NAME,
            payload_format=PEFT_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
            adapter_kind=PEFT_CLASSIFIER_ADAPTER_KIND,
            config=config,
        )
    return PeftEncoderTrainingBackend(config=config)


def _normalize_family_name(family_name: object) -> str:
    return str(family_name).strip().lower().replace("-", "_")


def _require_peft_encoder_runtime_payload(
    runtime_payload: object,
) -> LoraClassifierInitialStateConfig:
    required_attributes = (
        "artifact_format",
        "lora_adapter_artifact_ref",
        "classifier_head_artifact_ref",
        "backbone_payload",
        "lora_config_payload",
        "peft_adapter_config_payload",
    )
    missing = [
        attribute
        for attribute in required_attributes
        if not hasattr(runtime_payload, attribute)
    ]
    if missing:
        raise TypeError(
            "PEFT text classifier runtime payload is missing required surface: "
            f"{missing}"
        )
    return cast(LoraClassifierInitialStateConfig, runtime_payload)
