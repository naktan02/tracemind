"""Server aggregation runtime-boundary validation helpers."""

from __future__ import annotations

from collections.abc import Sequence

from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)


def require_base_adapter_kind(
    *,
    base_state: SharedAdapterState,
    adapter_kind: str,
    context: str,
) -> None:
    """Validate that the server runtime selected the expected adapter family."""

    if base_state.adapter_kind != adapter_kind:
        raise ValueError(
            f"Base state adapter_kind does not match the {context} aggregator: "
            f"{base_state.adapter_kind}"
        )


def select_non_empty_updates(
    update_payloads: Sequence[SharedAdapterUpdate],
) -> list[SharedAdapterUpdate]:
    """Filter zero-example submissions before calling method-level aggregation."""

    valid_updates = [
        payload for payload in update_payloads if payload.example_count > 0
    ]
    if not valid_updates:
        raise ValueError("At least one non-empty update payload is required.")
    return valid_updates


def require_update_matches_base(
    *,
    payload: SharedAdapterUpdate,
    base_state: SharedAdapterState,
    adapter_kind: str,
    context: str,
) -> None:
    """Validate shared update lineage before family-specific payload checks."""

    if payload.adapter_kind != adapter_kind:
        raise ValueError(
            f"Update adapter_kind does not match the {context} aggregator: "
            f"{payload.adapter_kind}"
        )
    if payload.model_id != base_state.model_id:
        raise ValueError("All update payloads must match the base model_id.")
    if payload.base_model_revision != base_state.model_revision:
        raise ValueError("All update payloads must match the base model revision.")
    if payload.training_scope != base_state.training_scope:
        raise ValueError("All update payloads must match the training scope.")
