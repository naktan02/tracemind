"""Initial shared-state factories for federated simulation."""

from __future__ import annotations

from datetime import datetime
from importlib import import_module
from typing import Any

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
    """simulation bootstrap용 초기 shared state를 config-declared builder로 만든다."""

    builder = _load_initial_state_builder(
        _initial_state_builder_path(round_runtime_config)
    )
    state = builder(
        round_runtime_config=round_runtime_config,
        model_id=model_id,
        model_revision=model_revision,
        training_scope=training_scope,
        embedding_dim=embedding_dim,
        labels=labels,
        updated_at=updated_at,
    )
    if state is None:
        raise ValueError(
            "round_runtime.initial_state_builder returned no initial shared state: "
            f"{_initial_state_builder_path(round_runtime_config)!r}."
        )
    return state


def _initial_state_builder_path(round_runtime_config: Any) -> str:
    raw_value = getattr(round_runtime_config, "initial_state_builder", None)
    builder_path = "" if raw_value is None else str(raw_value).strip()
    if not builder_path:
        raise ValueError("round_runtime.initial_state_builder is required.")
    return builder_path


def _load_initial_state_builder(builder_path: str) -> Any:
    module_name, separator, function_name = builder_path.rpartition(".")
    if not separator or not module_name or not function_name:
        raise ValueError(
            "round_runtime.initial_state_builder must be a fully qualified "
            f"function path: {builder_path!r}."
        )
    module = import_module(module_name)
    builder = getattr(module, function_name, None)
    if not callable(builder):
        raise ValueError(
            "round_runtime.initial_state_builder must point to a callable: "
            f"{builder_path!r}."
        )
    return builder
