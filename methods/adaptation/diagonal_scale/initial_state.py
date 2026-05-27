"""Diagonal-scale update family initial shared-state builder."""

from __future__ import annotations

from datetime import datetime

from shared.src.contracts.adapter_contract_families.diagonal_scale import (
    VectorAdapterState,
)
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState


def build_initial_diagonal_scale_state(
    *,
    round_runtime_config: object,
    model_id: str,
    model_revision: str,
    training_scope: str,
    embedding_dim: int,
    labels: tuple[str, ...] | list[str],
    updated_at: datetime,
) -> SharedAdapterState:
    """diagonal-scale update family의 identity shared state를 만든다."""

    _ = (round_runtime_config, labels)
    return VectorAdapterState.identity(
        model_id=model_id,
        model_revision=model_revision,
        training_scope=training_scope,
        embedding_dim=embedding_dim,
        updated_at=updated_at,
    )
