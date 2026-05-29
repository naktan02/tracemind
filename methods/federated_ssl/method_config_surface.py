"""Method-owned FL SSL config/report surface helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Mapping

from methods.federated_ssl.base import FederatedSslMethodDescriptor
from methods.federated_ssl.method_parameters import (
    build_federated_ssl_method_parameter_snapshot,
)
from methods.federated_ssl.registry import (
    resolve_federated_ssl_method_descriptor,
    resolve_federated_ssl_method_descriptor_module,
)

DEFAULT_METHOD_SCHEMA_VERSION = "federated_ssl_method.v1"
DEFAULT_LOCAL_BUDGET_POLICY = "iteration_capped"


def build_federated_ssl_method_config_surface(
    *,
    method_name: str,
    method_config: Mapping[str, object],
) -> dict[str, object]:
    """얇은 Hydra method 선택값을 method-owned metadata로 확장한다."""

    descriptor = resolve_federated_ssl_method_descriptor(method_name)
    parameter_snapshot = build_federated_ssl_method_parameter_snapshot(
        method_name=descriptor.name,
        method_config=method_config,
    )
    method_module = _import_method_descriptor_module(descriptor.name)
    return {
        "schema_version": str(
            method_config.get("schema_version", DEFAULT_METHOD_SCHEMA_VERSION)
        ),
        "name": descriptor.name,
        "display_name": descriptor.display_name or descriptor.name,
        "method_role": descriptor.method_role,
        "implementation_status": descriptor.implementation_status,
        "original_source": _module_mapping(
            method_module,
            descriptor.name,
            "ORIGINAL_SOURCE_METADATA",
        ),
        "local_budget_policy": str(
            method_config.get("local_budget_policy", DEFAULT_LOCAL_BUDGET_POLICY)
        ),
        "trace_mapping": _module_mapping(
            method_module,
            descriptor.name,
            "TRACE_MAPPING_METADATA",
        ),
        "client_step": _client_step_mapping(descriptor),
        "server_step": _server_step_mapping(descriptor),
        "round_state_exchange": _round_state_exchange_mapping(descriptor),
        "report_tags": _module_sequence(
            method_module,
            descriptor.name,
            "REPORT_TAGS",
        ),
        "notes": _module_sequence(method_module, descriptor.name, "NOTES"),
        **parameter_snapshot.to_mapping(),
    }


def default_method_local_ssl_policy_name(
    method_descriptor: FederatedSslMethodDescriptor,
) -> str | None:
    """method descriptor module이 선언한 기본 local SSL policy 이름을 읽는다."""

    method_module = _import_method_descriptor_module(method_descriptor.name)
    value = _method_surface_value(
        method_module,
        method_descriptor.name,
        "DEFAULT_LOCAL_SSL_POLICY_NAME",
        None,
    )
    if value is not None:
        text = str(value).strip()
        return text or None
    names = method_descriptor.required_capabilities.local_ssl_policy_names
    if len(names) == 1:
        return names[0]
    return None


def default_method_server_update_policy_name(
    method_descriptor: FederatedSslMethodDescriptor,
) -> str | None:
    """method descriptor module이 선언한 기본 server update policy 이름을 읽는다."""

    method_module = _import_method_descriptor_module(method_descriptor.name)
    value = _method_surface_value(
        method_module,
        method_descriptor.name,
        "DEFAULT_SERVER_UPDATE_POLICY_NAME",
        None,
    )
    if value is not None:
        text = str(value).strip()
        return text or None
    names = method_descriptor.required_capabilities.server_update_policy_names
    if len(names) == 1:
        return names[0]
    return None


def default_method_server_step_policy_name(
    method_descriptor: FederatedSslMethodDescriptor,
    method_config: Mapping[str, object] | None = None,
) -> str | None:
    """method descriptor module이 선언한 기본 server step policy 이름을 읽는다."""

    scenario_value = _scenario_default_value(
        method_descriptor=method_descriptor,
        method_config=method_config,
        attribute_name="DEFAULT_SERVER_STEP_POLICY_BY_SCENARIO",
    )
    if scenario_value is not None:
        return scenario_value
    method_module = _import_method_descriptor_module(method_descriptor.name)
    value = _method_surface_value(
        method_module,
        method_descriptor.name,
        "DEFAULT_SERVER_STEP_POLICY_NAME",
        None,
    )
    if value is not None:
        text = str(value).strip()
        return text or None
    names = method_descriptor.required_capabilities.server_step_policy_names
    if len(names) == 1:
        return names[0]
    return None


def default_method_peer_context_policy_name(
    method_descriptor: FederatedSslMethodDescriptor,
    method_config: Mapping[str, object] | None = None,
) -> str | None:
    """method descriptor module이 선언한 기본 peer context policy 이름을 읽는다."""

    scenario_value = _scenario_default_value(
        method_descriptor=method_descriptor,
        method_config=method_config,
        attribute_name="DEFAULT_PEER_CONTEXT_POLICY_BY_SCENARIO",
    )
    if scenario_value is not None:
        return scenario_value
    method_module = _import_method_descriptor_module(method_descriptor.name)
    value = _method_surface_value(
        method_module,
        method_descriptor.name,
        "DEFAULT_PEER_CONTEXT_POLICY_NAME",
        None,
    )
    if value is not None:
        text = str(value).strip()
        return text or None
    names = method_descriptor.required_capabilities.peer_context_policy_names
    if len(names) == 1:
        return names[0]
    return None


def default_method_labeled_exposure_policy_name(
    method_descriptor: FederatedSslMethodDescriptor,
    method_config: Mapping[str, object] | None = None,
) -> str | None:
    """method descriptor가 선언한 기본 labeled exposure policy 이름을 읽는다."""

    scenario_value = _scenario_default_value(
        method_descriptor=method_descriptor,
        method_config=method_config,
        attribute_name="DEFAULT_LABELED_EXPOSURE_POLICY_BY_SCENARIO",
    )
    if scenario_value is not None:
        return scenario_value
    names = method_descriptor.required_capabilities.labeled_exposure_policy_names
    if len(names) == 1:
        return names[0]
    return None


def default_method_local_supervision_regime_name(
    method_descriptor: FederatedSslMethodDescriptor,
    method_config: Mapping[str, object] | None = None,
) -> str | None:
    """method descriptor가 선언한 기본 local supervision regime 이름을 읽는다."""

    scenario_value = _scenario_default_value(
        method_descriptor=method_descriptor,
        method_config=method_config,
        attribute_name="DEFAULT_LOCAL_SUPERVISION_REGIME_BY_SCENARIO",
    )
    if scenario_value is not None:
        return scenario_value
    names = method_descriptor.required_capabilities.local_supervision_regime_names
    if len(names) == 1:
        return names[0]
    return None


def default_method_update_partition_policy_name(
    method_descriptor: FederatedSslMethodDescriptor,
) -> str | None:
    """method descriptor가 선언한 기본 update partition policy 이름을 읽는다."""

    names = method_descriptor.required_capabilities.update_partition_policy_names
    if len(names) == 1:
        return names[0]
    return None


def default_method_aggregation_weight_policy_name(
    method_descriptor: FederatedSslMethodDescriptor,
) -> str | None:
    """method descriptor가 선언한 기본 aggregation weight policy 이름을 읽는다."""

    names = method_descriptor.required_capabilities.aggregation_weight_policy_names
    if len(names) == 1:
        return names[0]
    return None


def is_public_method_owned_canonical(
    method_descriptor: FederatedSslMethodDescriptor,
) -> bool:
    """descriptor가 public canonical method-owned surface인지 반환한다."""

    try:
        method_module = _import_method_descriptor_module(method_descriptor.name)
    except ModuleNotFoundError:
        return True
    value = _method_surface_value(
        method_module,
        method_descriptor.name,
        "PUBLIC_METHOD_OWNED_CANONICAL",
        True,
    )
    return bool(value)


def recommended_method_owned_variants(
    method_descriptor: FederatedSslMethodDescriptor,
) -> tuple[str, ...]:
    """generic descriptor가 public surface 대신 안내할 canonical variant 목록."""

    try:
        method_module = _import_method_descriptor_module(method_descriptor.name)
    except ModuleNotFoundError:
        return ()
    raw_value = _method_surface_value(
        method_module,
        method_descriptor.name,
        "METHOD_OWNED_VARIANT_RECOMMENDATIONS",
        (),
    )
    if raw_value is None:
        return ()
    if isinstance(raw_value, str):
        return (raw_value,)
    return tuple(str(item) for item in raw_value)


def _client_step_mapping(
    descriptor: FederatedSslMethodDescriptor,
) -> dict[str, object]:
    return {
        "task_type": descriptor.local_step.step_name,
        "custom_method_runtime_required": (
            descriptor.runtime_capabilities.requires_custom_client_runtime
        ),
        "client_trainer_name": descriptor.local_step.client_trainer_name,
        "pseudo_labeler_name": descriptor.local_step.pseudo_labeler_name,
        "training_row_source": descriptor.local_step.training_row_source,
    }


def _server_step_mapping(
    descriptor: FederatedSslMethodDescriptor,
) -> dict[str, object]:
    return {
        "custom_round_policy_required": (
            descriptor.runtime_capabilities.requires_custom_server_runtime
        ),
        "server_aggregator_name": descriptor.server_step.server_aggregator_name,
        "round_policy_name": descriptor.server_step.round_policy_name,
        "server_aggregate_hint": descriptor.server_step.server_aggregate_hint,
    }


def _round_state_exchange_mapping(
    descriptor: FederatedSslMethodDescriptor,
) -> dict[str, object]:
    spec = descriptor.round_state_exchange
    if spec is None:
        return {
            "exchange_name": None,
            "custom_exchange_required": False,
            "required_client_metric_keys": [],
        }
    return {
        "exchange_name": spec.exchange_name,
        "custom_exchange_required": spec.requires_custom_exchange,
        "required_client_metric_keys": list(spec.required_client_metric_keys),
        "summary_metric_prefix": spec.summary_metric_prefix,
    }


def _import_method_descriptor_module(method_name: str) -> object:
    method_package = f"methods.federated_ssl.{method_name}"
    module_name = f"{method_package}.descriptor"
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name not in {method_package, module_name}:
            raise
        return resolve_federated_ssl_method_descriptor_module(method_name)


def _method_surface_value(
    module: object,
    method_name: str,
    attribute_name: str,
    default: object,
) -> object:
    surface_by_name = getattr(module, "METHOD_CONFIG_SURFACE_BY_METHOD_NAME", {})
    if isinstance(surface_by_name, Mapping):
        surface = surface_by_name.get(method_name, {})
        if isinstance(surface, Mapping) and attribute_name in surface:
            return surface[attribute_name]
    return getattr(module, attribute_name, default)


def _scenario_default_value(
    *,
    method_descriptor: FederatedSslMethodDescriptor,
    method_config: Mapping[str, object] | None,
    attribute_name: str,
) -> str | None:
    if method_config is None:
        return None
    raw_scenario = method_config.get("scenario")
    if raw_scenario is None:
        return None
    scenario = str(raw_scenario).strip().lower().replace("_", "-")
    if not scenario:
        return None

    method_module = _import_method_descriptor_module(method_descriptor.name)
    raw_mapping = _method_surface_value(
        method_module,
        method_descriptor.name,
        attribute_name,
        {},
    )
    if not isinstance(raw_mapping, Mapping):
        return None

    for key, value in raw_mapping.items():
        normalized_key = str(key).strip().lower().replace("_", "-")
        if normalized_key == scenario:
            text = str(value).strip()
            return text or None
    return None


def _module_mapping(
    module: object,
    method_name: str,
    attribute_name: str,
) -> dict[str, object]:
    value = _method_surface_value(module, method_name, attribute_name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{attribute_name} must be a mapping.")
    return dict(value)


def _module_sequence(
    module: object,
    method_name: str,
    attribute_name: str,
) -> list[str]:
    value = _method_surface_value(module, method_name, attribute_name, ())
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]
