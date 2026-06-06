"""Shared adapter family initial state builder dispatcher."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

from methods.adaptation.implementation_modules import (
    adaptation_implementation_module_name,
    normalize_payload_adapter_kind,
)
from shared.src.contracts.adapter_contract_families.base import (
    SharedAdapterStatePayload,
)


@dataclass(frozen=True, slots=True)
class SharedAdapterInitialStateRequest:
    """Initial shared adapter state 생성 요청."""

    payload_adapter_kind: str
    update_family_name: str
    model_id: str
    model_revision: str
    training_scope: str
    labels: Sequence[str] = ()
    embedding_dim: int | None = None
    updated_at: datetime | None = None
    round_runtime_config: object | None = None


SharedAdapterInitialStateBuilder = Callable[
    [SharedAdapterInitialStateRequest],
    SharedAdapterStatePayload,
]

_INITIAL_STATE_BUILDERS: dict[str, SharedAdapterInitialStateBuilder] = {}


def register_shared_adapter_initial_state_builder(
    *payload_adapter_kinds: str,
    builder: SharedAdapterInitialStateBuilder | None = None,
) -> (
    Callable[[SharedAdapterInitialStateBuilder], SharedAdapterInitialStateBuilder]
    | SharedAdapterInitialStateBuilder
):
    """adapter family 구현 옆에서 initial state builder를 등록한다."""

    def _decorator(
        builder: SharedAdapterInitialStateBuilder,
    ) -> SharedAdapterInitialStateBuilder:
        for payload_adapter_kind in payload_adapter_kinds:
            normalized_kind = normalize_payload_adapter_kind(payload_adapter_kind)
            registered = _INITIAL_STATE_BUILDERS.get(normalized_kind)
            if registered is not None and registered is not builder:
                raise ValueError(
                    "Duplicate shared adapter initial state builder registration: "
                    f"{normalized_kind}."
                )
            _INITIAL_STATE_BUILDERS[normalized_kind] = builder
        return builder

    if builder is not None:
        return _decorator(builder)
    return _decorator


def build_initial_shared_adapter_state(
    request: SharedAdapterInitialStateRequest,
) -> SharedAdapterStatePayload:
    """adapter family가 소유한 initial shared state builder를 실행한다."""

    normalized_kind = normalize_payload_adapter_kind(request.payload_adapter_kind)
    builder = _INITIAL_STATE_BUILDERS.get(normalized_kind)
    if builder is None:
        _import_initial_state_module(normalized_kind)
        builder = _INITIAL_STATE_BUILDERS.get(normalized_kind)
    if builder is None:
        raise ValueError(
            "No initial shared adapter state builder is registered for "
            f"{normalized_kind}."
        )
    payload = builder(request)
    if payload.adapter_kind != normalized_kind:
        raise ValueError(
            "Initial state adapter_kind does not match requested payload adapter: "
            f"{payload.adapter_kind} != {normalized_kind}."
        )
    return payload


def _import_initial_state_module(normalized_payload_adapter_kind: str) -> None:
    module_name = adaptation_implementation_module_name(
        payload_adapter_kind=normalized_payload_adapter_kind,
        submodule="initial_state",
    )
    importlib.import_module(module_name)
