"""FL SSL method 원본 parameter snapshot 해석."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Mapping

ORIGINAL_PARAMETER_MAPPING_FUNCTION = "original_parameter_mapping"
DEFAULT_ORIGINAL_SCENARIO_NAME = "DEFAULT_ORIGINAL_SCENARIO"


@dataclass(frozen=True, slots=True)
class FederatedSslMethodParameterSnapshot:
    """method 원본값과 실행 override를 함께 기록하는 snapshot."""

    method_name: str
    scenario: str
    use_original_parameters: bool
    original_parameters: dict[str, object]
    parameter_overrides: dict[str, object]
    effective_parameters: dict[str, object]
    parameter_override_status: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "scenario": self.scenario,
            "use_original_parameters": self.use_original_parameters,
            "original_parameters": dict(self.original_parameters),
            "parameter_overrides": dict(self.parameter_overrides),
            "effective_parameters": dict(self.effective_parameters),
            "parameter_override_status": self.parameter_override_status,
        }


def build_federated_ssl_method_parameter_snapshot(
    *,
    method_name: str,
    method_config: Mapping[str, object],
) -> FederatedSslMethodParameterSnapshot:
    """Hydra method descriptor config와 method-local original spec을 합친다."""

    normalized_method_name = _normalize_method_name(method_name)
    use_original_parameters = _bool_value(
        method_config.get("use_original_parameters", True)
    )
    scenario = _resolve_scenario(
        method_name=normalized_method_name,
        method_config=method_config,
    )
    original_parameters = (
        _load_original_parameters(
            method_name=normalized_method_name,
            scenario=scenario,
        )
        if use_original_parameters
        else {}
    )
    parameter_overrides = _mapping_value(method_config.get("parameter_overrides"))
    _validate_override_keys(
        method_name=normalized_method_name,
        original_parameters=original_parameters,
        parameter_overrides=parameter_overrides,
        use_original_parameters=use_original_parameters,
    )
    effective_parameters = {
        **original_parameters,
        **parameter_overrides,
    }
    status = "original"
    if parameter_overrides:
        status = "ablation"
    elif not use_original_parameters:
        status = "custom"
    return FederatedSslMethodParameterSnapshot(
        method_name=normalized_method_name,
        scenario=scenario,
        use_original_parameters=use_original_parameters,
        original_parameters=original_parameters,
        parameter_overrides=parameter_overrides,
        effective_parameters=effective_parameters,
        parameter_override_status=status,
    )


def _load_original_parameters(
    *,
    method_name: str,
    scenario: str,
) -> dict[str, object]:
    module = _import_original_spec_module(method_name)
    mapping_factory = getattr(module, ORIGINAL_PARAMETER_MAPPING_FUNCTION, None)
    if not callable(mapping_factory):
        raise ValueError(
            "method original_spec must expose callable "
            f"{ORIGINAL_PARAMETER_MAPPING_FUNCTION}: method={method_name}"
        )
    parameters = mapping_factory(scenario_name=scenario)
    if not isinstance(parameters, Mapping):
        raise TypeError("original_parameter_mapping() must return a mapping.")
    return dict(parameters)


def _resolve_scenario(
    *,
    method_name: str,
    method_config: Mapping[str, object],
) -> str:
    configured = _optional_str(method_config.get("scenario"))
    if configured is not None:
        return configured
    module = _import_original_spec_module(method_name)
    default_scenario = _optional_str(
        getattr(module, DEFAULT_ORIGINAL_SCENARIO_NAME, "")
    )
    if default_scenario is None:
        raise ValueError(f"method={method_name} requires ssl_method.scenario.")
    return default_scenario


def _import_original_spec_module(method_name: str) -> object:
    module_name = f"methods.federated_ssl.{method_name}.original_spec"
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise ValueError(
                f"method={method_name} does not expose original_spec.py."
            ) from exc
        raise


def _validate_override_keys(
    *,
    method_name: str,
    original_parameters: Mapping[str, object],
    parameter_overrides: Mapping[str, object],
    use_original_parameters: bool,
) -> None:
    if not use_original_parameters:
        return
    unknown_keys = sorted(set(parameter_overrides) - set(original_parameters))
    if unknown_keys:
        raise ValueError(
            f"Unknown parameter_overrides for method={method_name}: {unknown_keys}."
        )


def _mapping_value(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("parameter_overrides must be a mapping.")
    return dict(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return bool(value)


def _normalize_method_name(method_name: str) -> str:
    normalized = method_name.strip()
    if not normalized:
        raise ValueError("method_name must not be empty.")
    if not normalized.replace("_", "").isalnum():
        raise ValueError(f"Unsupported method_name: {method_name!r}")
    return normalized
