"""Classifier-head 초기화 helper."""

from __future__ import annotations

from datetime import datetime

from shared.src.contracts.adapter_contract_families.classifier_head import (
    ClassifierHeadState,
)
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState


def build_zero_classifier_head_state(
    *,
    round_runtime_config: object,
    model_id: str,
    model_revision: str,
    training_scope: str,
    embedding_dim: int,
    labels: tuple[str, ...] | list[str],
    updated_at: datetime,
) -> SharedAdapterState:
    """linear-head update family의 zero-initialized shared state를 만든다."""

    _ = round_runtime_config
    return ClassifierHeadState.zero_initialized(
        model_id=model_id,
        model_revision=model_revision,
        labels=labels,
        embedding_dim=embedding_dim,
        training_scope=training_scope,
        updated_at=updated_at,
    )
