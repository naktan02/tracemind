"""Shared adapter payload registry."""

from __future__ import annotations

from collections.abc import Mapping

from .base import SharedAdapterStatePayload, SharedAdapterUpdatePayload

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


def get_shared_adapter_state_payload_type(
    adapter_kind: str,
) -> type[SharedAdapterStatePayload]:
    """adapter family discriminator에 맞는 state payload type을 반환한다."""

    _ensure_builtin_payload_families_loaded()
    normalized_adapter_kind = _normalize_adapter_kind(adapter_kind)
    payload_type = _STATE_PAYLOAD_TYPES.get(normalized_adapter_kind)
    if payload_type is None:
        raise ValueError(
            f"Unsupported shared adapter state kind: {normalized_adapter_kind}"
        )
    return payload_type


def get_shared_adapter_update_payload_type(
    adapter_kind: str,
) -> type[SharedAdapterUpdatePayload]:
    """adapter family discriminator에 맞는 update payload type을 반환한다."""

    _ensure_builtin_payload_families_loaded()
    normalized_adapter_kind = _normalize_adapter_kind(adapter_kind)
    payload_type = _UPDATE_PAYLOAD_TYPES.get(normalized_adapter_kind)
    if payload_type is None:
        raise ValueError(
            f"Unsupported shared adapter update kind: {normalized_adapter_kind}"
        )
    return payload_type


def list_registered_shared_adapter_payload_adapter_kinds() -> tuple[str, ...]:
    """builtin/custom payload parser registry에 등록된 adapter kind 목록."""

    _ensure_builtin_payload_families_loaded()
    return tuple(sorted(set(_STATE_PAYLOAD_TYPES) & set(_UPDATE_PAYLOAD_TYPES)))


def get_shared_adapter_canonical_update_payload_format(
    adapter_kind: str,
) -> str:
    """adapter family update payload type이 선언한 canonical envelope format."""

    normalized_adapter_kind = _normalize_adapter_kind(adapter_kind)
    payload_type = get_shared_adapter_update_payload_type(normalized_adapter_kind)
    return str(
        getattr(
            payload_type,
            "canonical_update_payload_format",
            normalized_adapter_kind,
        )
    )


def get_shared_adapter_update_payload_formats(
    adapter_kind: str,
) -> tuple[str, ...]:
    """adapter family update payload type이 허용하는 envelope format 목록."""

    canonical_format = get_shared_adapter_canonical_update_payload_format(adapter_kind)
    payload_type = get_shared_adapter_update_payload_type(adapter_kind)
    raw_formats = getattr(
        payload_type,
        "accepted_update_payload_formats",
        (canonical_format,),
    )
    return tuple(str(value).strip() for value in raw_formats if str(value).strip())


def parse_shared_adapter_state_payload(
    source: Mapping[str, object] | SharedAdapterStatePayload,
) -> SharedAdapterStatePayload:
    """mapping 또는 payload instance를 registered state payload로 정규화한다."""
    if isinstance(source, SharedAdapterStatePayload):
        return source
    _ensure_builtin_payload_families_loaded()
    data = dict(source)
    adapter_kind = _adapter_kind_from_payload(data, payload_role="state")
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
    adapter_kind = _adapter_kind_from_payload(data, payload_role="update")
    payload_type = _UPDATE_PAYLOAD_TYPES.get(adapter_kind)
    if payload_type is None:
        raise ValueError(f"Unsupported shared adapter update kind: {adapter_kind}")
    return payload_type.model_validate(data)


def _adapter_kind_from_payload(
    data: Mapping[str, object],
    *,
    payload_role: str,
) -> str:
    raw_adapter_kind = data.get("adapter_kind")
    if raw_adapter_kind is not None:
        return _normalize_adapter_kind(raw_adapter_kind)
    raise ValueError("Shared adapter payload requires adapter_kind.")


def _normalize_adapter_kind(adapter_kind: object) -> str:
    return str(adapter_kind).strip().lower()


def _ensure_builtin_payload_families_loaded() -> None:
    from .builtin_loader import load_builtin_shared_adapter_payload_families

    load_builtin_shared_adapter_payload_families()
