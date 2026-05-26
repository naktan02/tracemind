"""Initial shared-state factories for federated simulation."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from methods.adaptation.text_classifier.peft_encoder.initial_state import (
    build_initial_lora_classifier_state,
)
from shared.src.contracts.adapter_contract_families.classifier_head import (
    CLASSIFIER_HEAD_ADAPTER_KIND,
    ClassifierHeadState,
)
from shared.src.contracts.adapter_contract_families.diagonal_scale import (
    DIAGONAL_SCALE_ADAPTER_KIND,
    VectorAdapterState,
)
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LORA_CLASSIFIER_ADAPTER_KIND,
)
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState


def build_initial_shared_state(
    *,
    round_runtime_config: Any,
    model_id: str,
    model_revision: str,
    training_scope: str,
    embedding_dim: int,
    labels: tuple[str, ...] | list[str],
    updated_at: datetime,
) -> SharedAdapterState:
    """simulation bootstrap용 초기 shared state를 family별로 만든다."""
    adapter_family_name = _adapter_family_name(round_runtime_config)
    if adapter_family_name == CLASSIFIER_HEAD_ADAPTER_KIND:
        return ClassifierHeadState.zero_initialized(
            model_id=model_id,
            model_revision=model_revision,
            labels=labels,
            embedding_dim=embedding_dim,
            training_scope=training_scope,
            updated_at=updated_at,
        )
    if adapter_family_name == LORA_CLASSIFIER_ADAPTER_KIND:
        lora_config = getattr(round_runtime_config, "lora_classifier", None)
        if lora_config is None:
            raise ValueError(
                "lora_classifier round runtime requires lora_classifier "
                "bootstrap config."
            )
        return build_initial_lora_classifier_state(
            config=lora_config,
            model_id=model_id,
            model_revision=model_revision,
            training_scope=training_scope,
            labels=labels,
            updated_at=updated_at,
        )
    if adapter_family_name == DIAGONAL_SCALE_ADAPTER_KIND:
        return VectorAdapterState.identity(
            model_id=model_id,
            model_revision=model_revision,
            training_scope=training_scope,
            embedding_dim=embedding_dim,
            updated_at=updated_at,
        )
    raise ValueError(f"Unsupported simulation adapter family: {adapter_family_name}")


def _adapter_family_name(round_runtime_config: Any) -> str:
    return str(round_runtime_config.adapter_family_name).strip().lower()
