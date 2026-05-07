"""Shared adapter payload registry."""

from __future__ import annotations

from collections.abc import Mapping

from .base import (
    AdapterKind,
    SharedAdapterStatePayload,
    SharedAdapterUpdatePayload,
)

_STATE_PAYLOAD_TYPES: dict[str, type[SharedAdapterStatePayload]] = {}
_UPDATE_PAYLOAD_TYPES: dict[str, type[SharedAdapterUpdatePayload]] = {}


def register_shared_adapter_state_payload_type(
    adapter_kind: str,
    payload_type: type[SharedAdapterStatePayload],
) -> None:
    """adapter family별 state payload 타입을 registry에 등록한다."""

    _STATE_PAYLOAD_TYPES[adapter_kind.strip().lower()] = payload_type


def register_shared_adapter_update_payload_type(
    adapter_kind: str,
    payload_type: type[SharedAdapterUpdatePayload],
) -> None:
    """adapter family별 update payload 타입을 registry에 등록한다."""

    _UPDATE_PAYLOAD_TYPES[adapter_kind.strip().lower()] = payload_type


def register_shared_adapter_payload_family(
    adapter_kind: str,
    *,
    state_payload_type: type[SharedAdapterStatePayload],
    update_payload_type: type[SharedAdapterUpdatePayload],
) -> None:
    """adapter family의 state/update payload 타입을 함께 등록한다."""

    register_shared_adapter_state_payload_type(adapter_kind, state_payload_type)
    register_shared_adapter_update_payload_type(adapter_kind, update_payload_type)


def parse_shared_adapter_state_payload(
    source: Mapping[str, object] | SharedAdapterStatePayload,
) -> SharedAdapterStatePayload:
    """mapping 또는 payload instance를 registered state payload로 정규화한다."""
    if isinstance(source, SharedAdapterStatePayload):
        return source
    _ensure_builtin_payload_families_loaded()
    data = dict(source)
    adapter_kind = (
        str(data.get("adapter_kind", AdapterKind.DIAGONAL_SCALE.value)).strip().lower()
    )
    payload_type = _STATE_PAYLOAD_TYPES.get(adapter_kind)
    if payload_type is None:
        raise ValueError(f"Unsupported shared adapter state kind: {adapter_kind}")
    return payload_type.model_validate(data)


def parse_shared_adapter_update_payload(
    source: Mapping[str, object] | SharedAdapterUpdatePayload,
) -> SharedAdapterUpdatePayload:
    """mapping 또는 payload instance를 registered update payload로 정규화한다."""
    if isinstance(source, SharedAdapterUpdatePayload):
        return source
    _ensure_builtin_payload_families_loaded()
    data = dict(source)
    adapter_kind = (
        str(data.get("adapter_kind", AdapterKind.DIAGONAL_SCALE.value)).strip().lower()
    )
    payload_type = _UPDATE_PAYLOAD_TYPES.get(adapter_kind)
    if payload_type is None:
        raise ValueError(f"Unsupported shared adapter update kind: {adapter_kind}")
    return payload_type.model_validate(data)


def _ensure_builtin_payload_families_loaded() -> None:
    from .builtin_loader import load_builtin_shared_adapter_payload_families

    load_builtin_shared_adapter_payload_families()
