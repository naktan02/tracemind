"""Adapter runtime/objective compatibility dispatcher."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Protocol

from methods.adaptation.adapter_family_modules import (
    adapter_family_module_name,
    normalize_adapter_kind,
)
from shared.src.contracts.training_contracts import TrainingObjectiveConfig


class AdapterRuntimeObjectiveCompatibilityValidator(Protocol):
    """adapter family별 runtime/objective drift 검증 함수 표면."""

    def __call__(
        self,
        *,
        runtime_config: object,
        objective_config: TrainingObjectiveConfig | None,
    ) -> None:
        """runtime config와 training objective payload config를 비교한다."""


_RUNTIME_OBJECTIVE_COMPATIBILITY_VALIDATORS: dict[
    str,
    AdapterRuntimeObjectiveCompatibilityValidator,
] = {}


def register_runtime_objective_compatibility_validator(
    adapter_kind: str,
) -> Callable[
    [AdapterRuntimeObjectiveCompatibilityValidator],
    AdapterRuntimeObjectiveCompatibilityValidator,
]:
    """adapter family 구현 옆에서 runtime/objective compatibility 검증을 등록한다."""

    normalized_adapter_kind = _normalize_adapter_kind(adapter_kind)

    def _decorator(
        validator: AdapterRuntimeObjectiveCompatibilityValidator,
    ) -> AdapterRuntimeObjectiveCompatibilityValidator:
        if normalized_adapter_kind in _RUNTIME_OBJECTIVE_COMPATIBILITY_VALIDATORS:
            raise ValueError(
                "Duplicate runtime/objective compatibility validator "
                f"registration: {normalized_adapter_kind}"
            )
        _RUNTIME_OBJECTIVE_COMPATIBILITY_VALIDATORS[normalized_adapter_kind] = validator
        return validator

    return _decorator


def require_adapter_runtime_matches_objective(
    *,
    adapter_kind: str,
    runtime_config: object | None,
    objective_config: TrainingObjectiveConfig | None,
) -> None:
    """adapter family runtime config와 local objective payload config drift를 막는다."""

    normalized_adapter_kind = _normalize_adapter_kind(adapter_kind)
    validator = _RUNTIME_OBJECTIVE_COMPATIBILITY_VALIDATORS.get(normalized_adapter_kind)
    if validator is None:
        _import_runtime_compatibility_module_for_adapter_kind(normalized_adapter_kind)
        validator = _RUNTIME_OBJECTIVE_COMPATIBILITY_VALIDATORS.get(
            normalized_adapter_kind
        )
    if validator is None:
        return
    if runtime_config is None:
        raise ValueError(
            f"{normalized_adapter_kind} round runtime requires "
            "configured runtime payload."
        )
    validator(runtime_config=runtime_config, objective_config=objective_config)


def _import_runtime_compatibility_module_for_adapter_kind(
    normalized_adapter_kind: str,
) -> None:
    module_name = adapter_family_module_name(
        adapter_kind=normalized_adapter_kind,
        submodule="runtime_compatibility",
    )
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name or module_name.startswith(f"{error.name}."):
            return
        raise


def _normalize_adapter_kind(adapter_kind: str) -> str:
    return normalize_adapter_kind(adapter_kind)
