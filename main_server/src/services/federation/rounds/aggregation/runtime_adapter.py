"""Server aggregation runtime-boundary validation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)

StateT = TypeVar("StateT", bound=SharedAdapterState)
UpdateT = TypeVar("UpdateT", bound=SharedAdapterUpdate)


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


def require_typed_base_state(
    *,
    base_state: SharedAdapterState,
    state_type: type[StateT],
    adapter_kind: str,
    context: str,
    service_name: str,
) -> StateT:
    """Validate and narrow the base state expected by one runtime adapter."""

    if not isinstance(base_state, state_type):
        raise TypeError(
            f"{service_name} expects {state_type.__name__} as the base state, "
            f"got {type(base_state)!r}."
        )
    require_base_adapter_kind(
        base_state=base_state,
        adapter_kind=adapter_kind,
        context=context,
    )
    return base_state


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


def select_validated_updates(
    update_payloads: Sequence[SharedAdapterUpdate],
    *,
    update_type: type[UpdateT],
    base_state: SharedAdapterState,
    adapter_kind: str,
    context: str,
    service_name: str,
) -> list[UpdateT]:
    """Filter non-empty updates and validate common runtime lineage."""

    valid_updates = select_non_empty_updates(update_payloads)
    typed_updates: list[UpdateT] = []
    for payload in valid_updates:
        if not isinstance(payload, update_type):
            raise TypeError(
                f"{service_name} expects {update_type.__name__} updates, "
                f"got {type(payload)!r}."
            )
        require_update_matches_base(
            payload=payload,
            base_state=base_state,
            adapter_kind=adapter_kind,
            context=context,
        )
        typed_updates.append(payload)
    return typed_updates
