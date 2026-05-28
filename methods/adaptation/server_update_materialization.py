"""Server-side update materialization preflight."""

from __future__ import annotations

import importlib
from collections.abc import Callable

from methods.adaptation.payload_adapter_modules import (
    normalize_payload_adapter_kind,
    payload_adapter_module_name,
)
from shared.src.contracts.adapter_contract_families.base import (
    SharedAdapterUpdatePayload,
)

ServerUpdateMaterializationValidator = Callable[[SharedAdapterUpdatePayload], None]
_SERVER_UPDATE_MATERIALIZATION_VALIDATORS: dict[
    str,
    ServerUpdateMaterializationValidator,
] = {}


def register_server_update_materialization_validator(
    adapter_kind: str,
) -> Callable[
    [ServerUpdateMaterializationValidator],
    ServerUpdateMaterializationValidator,
]:
    """payload adapter 구현 owner 옆에서 서버 materialization preflight를 등록한다."""

    normalized_adapter_kind = _normalize_payload_adapter_kind(adapter_kind)

    def _decorator(
        validator: ServerUpdateMaterializationValidator,
    ) -> ServerUpdateMaterializationValidator:
        if normalized_adapter_kind in _SERVER_UPDATE_MATERIALIZATION_VALIDATORS:
            raise ValueError(
                "Duplicate server update materialization validator registration: "
                f"{normalized_adapter_kind}"
            )
        _SERVER_UPDATE_MATERIALIZATION_VALIDATORS[normalized_adapter_kind] = validator
        return validator

    return _decorator


def require_server_materializable_update_payload(
    update_payload: SharedAdapterUpdatePayload,
) -> None:
    """서버가 finalize 전에 materialize할 수 없는 update payload를 거부한다."""

    normalized_adapter_kind = _normalize_payload_adapter_kind(
        update_payload.adapter_kind
    )
    validator = _SERVER_UPDATE_MATERIALIZATION_VALIDATORS.get(normalized_adapter_kind)
    if validator is None:
        _import_materialization_module_for_adapter_kind(normalized_adapter_kind)
        validator = _SERVER_UPDATE_MATERIALIZATION_VALIDATORS.get(
            normalized_adapter_kind
        )
    if validator is not None:
        validator(update_payload)


def _import_materialization_module_for_adapter_kind(
    normalized_adapter_kind: str,
) -> None:
    module_name = payload_adapter_module_name(
        payload_adapter_kind=normalized_adapter_kind,
        submodule="server_preflight",
    )
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name or module_name.startswith(f"{error.name}."):
            return
        raise


def _normalize_payload_adapter_kind(adapter_kind: str) -> str:
    return normalize_payload_adapter_kind(adapter_kind)
