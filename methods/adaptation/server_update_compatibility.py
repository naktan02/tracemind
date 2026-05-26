"""Server-side update/state compatibility preflight."""

from __future__ import annotations

import importlib
from collections.abc import Callable

from methods.adaptation.adapter_family_modules import (
    adapter_family_module_name,
    normalize_adapter_kind,
)
from shared.src.contracts.adapter_contract_families.base import (
    SharedAdapterUpdatePayload,
)
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState

ServerUpdateCompatibilityValidator = Callable[
    [SharedAdapterUpdatePayload, SharedAdapterState], None
]
_SERVER_UPDATE_COMPATIBILITY_VALIDATORS: dict[
    str,
    ServerUpdateCompatibilityValidator,
] = {}


def register_server_update_compatibility_validator(
    adapter_kind: str,
) -> Callable[
    [ServerUpdateCompatibilityValidator],
    ServerUpdateCompatibilityValidator,
]:
    """adapter family 구현 옆에서 update/state compatibility 검증을 등록한다."""

    normalized_adapter_kind = _normalize_adapter_kind(adapter_kind)

    def _decorator(
        validator: ServerUpdateCompatibilityValidator,
    ) -> ServerUpdateCompatibilityValidator:
        if normalized_adapter_kind in _SERVER_UPDATE_COMPATIBILITY_VALIDATORS:
            raise ValueError(
                "Duplicate server update compatibility validator registration: "
                f"{normalized_adapter_kind}"
            )
        _SERVER_UPDATE_COMPATIBILITY_VALIDATORS[normalized_adapter_kind] = validator
        return validator

    return _decorator


def require_server_compatible_update_payload(
    *,
    update_payload: SharedAdapterUpdatePayload,
    active_state: SharedAdapterState,
) -> None:
    """서버 active state와 호환되지 않는 update payload를 거부한다."""

    normalized_adapter_kind = _normalize_adapter_kind(update_payload.adapter_kind)
    validator = _SERVER_UPDATE_COMPATIBILITY_VALIDATORS.get(normalized_adapter_kind)
    if validator is None:
        _import_compatibility_module_for_adapter_kind(normalized_adapter_kind)
        validator = _SERVER_UPDATE_COMPATIBILITY_VALIDATORS.get(normalized_adapter_kind)
    if validator is not None:
        validator(update_payload, active_state)


def _import_compatibility_module_for_adapter_kind(
    normalized_adapter_kind: str,
) -> None:
    module_name = adapter_family_module_name(
        adapter_kind=normalized_adapter_kind,
        submodule="server_preflight",
    )
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name or module_name.startswith(f"{error.name}."):
            return
        raise


def _normalize_adapter_kind(adapter_kind: str) -> str:
    return normalize_adapter_kind(adapter_kind)
