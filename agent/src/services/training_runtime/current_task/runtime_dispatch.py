"""Agent current-task runtime dispatch 해석."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from methods.federated_ssl.capabilities.plan import FederatedSslCapabilityPlan
from methods.federated_ssl.compatibility import (
    validate_federated_ssl_capability_compatibility,
)
from methods.federated_ssl.registry import resolve_federated_ssl_method_descriptor
from methods.federated_ssl.runtime_fallbacks import (
    PEFT_CLASSIFIER_UPDATE_PROFILE_NAME,
    PEFT_TEXT_ENCODER_UPDATE_FAMILY_NAME,
)
from methods.ssl.runtime.objective_config import QuerySslObjectiveRuntimeConfig

AGENT_RUNTIME_QUERY_SSL_PEFT = "query_ssl_peft"


@dataclass(slots=True, frozen=True)
class AgentCurrentTaskRuntimePlan:
    """agent가 현재 task를 어떤 live runtime으로 실행할지 해석한 결과."""

    runtime_name: str
    update_family_name: str
    query_ssl_config: QuerySslObjectiveRuntimeConfig
    fssl_method: str | None = None
    fssl_capability_plan: FederatedSslCapabilityPlan | None = None
    uses_legacy_fssl_context_only: bool = False


def resolve_current_task_runtime(
    task_payload: object,
) -> AgentCurrentTaskRuntimePlan:
    """TrainingTaskPayload를 agent live runtime plan으로 해석하고 검증한다."""

    query_ssl_config = QuerySslObjectiveRuntimeConfig.from_objective_config(
        getattr(task_payload, "objective_config", None)
    )
    if query_ssl_config is None:
        raise ValueError(
            "Query SSL objective가 없는 legacy stored-event training task는 "
            "agent runtime에서 지원하지 않습니다."
        )
    _require_supported_query_ssl_profile(task_payload)
    capability_plan = _validate_fssl_runtime_snapshot(task_payload)
    return AgentCurrentTaskRuntimePlan(
        runtime_name=AGENT_RUNTIME_QUERY_SSL_PEFT,
        update_family_name=_resolve_update_family_name(task_payload),
        query_ssl_config=query_ssl_config,
        fssl_method=_optional_name(getattr(task_payload, "fssl_method", None)),
        fssl_capability_plan=capability_plan,
        uses_legacy_fssl_context_only=_uses_legacy_fssl_context_only(task_payload),
    )


def _require_supported_query_ssl_profile(task_payload: object) -> None:
    objective_config = getattr(task_payload, "objective_config", None)
    profile_name = _optional_name(
        getattr(objective_config, "algorithm_profile_name", None)
    )
    if profile_name is None:
        return
    if profile_name != PEFT_CLASSIFIER_UPDATE_PROFILE_NAME:
        raise ValueError(
            "live agent Query SSL runtime은 현재 "
            f"{PEFT_CLASSIFIER_UPDATE_PROFILE_NAME!r} profile만 지원합니다: "
            f"{profile_name!r}."
        )


def _resolve_update_family_name(task_payload: object) -> str:
    execution = getattr(task_payload, "fssl_execution", None)
    if not isinstance(execution, Mapping):
        return PEFT_TEXT_ENCODER_UPDATE_FAMILY_NAME
    runtime_surface = execution.get("runtime_surface")
    if not isinstance(runtime_surface, Mapping):
        return PEFT_TEXT_ENCODER_UPDATE_FAMILY_NAME
    update_family_name = (
        _optional_name(runtime_surface.get("update_family_name"))
        or PEFT_TEXT_ENCODER_UPDATE_FAMILY_NAME
    )
    if update_family_name != PEFT_TEXT_ENCODER_UPDATE_FAMILY_NAME:
        raise ValueError(
            "live agent Query SSL runtime은 현재 "
            f"{PEFT_TEXT_ENCODER_UPDATE_FAMILY_NAME!r} update family만 "
            f"지원합니다: {update_family_name!r}."
        )
    return update_family_name


def _validate_fssl_runtime_snapshot(
    task_payload: object,
) -> FederatedSslCapabilityPlan | None:
    execution = getattr(task_payload, "fssl_execution", None)
    capability_payload = getattr(task_payload, "fssl_capability_plan", None)
    if execution is None and capability_payload is None:
        return None
    if not isinstance(execution, Mapping):
        raise ValueError("fssl_execution snapshot이 필요합니다.")
    if not isinstance(capability_payload, Mapping):
        raise ValueError("fssl_capability_plan snapshot이 필요합니다.")

    task_method = _optional_name(getattr(task_payload, "fssl_method", None))
    execution_method = _optional_name(execution.get("method_name"))
    method_name = execution_method or task_method
    if method_name is None:
        raise ValueError("FSSL runtime snapshot에는 method_name이 필요합니다.")
    if task_method is not None and task_method != method_name:
        raise ValueError(
            "fssl_method와 fssl_execution.method_name이 다릅니다: "
            f"{task_method!r} != {method_name!r}."
        )
    if _optional_name(execution.get("execution_role")) != "method_owned":
        raise ValueError("live agent FSSL runtime은 method_owned 실행만 지원합니다.")

    descriptor = resolve_federated_ssl_method_descriptor(method_name)
    if not descriptor.runtime_capabilities.live_agent_supported:
        raise ValueError(
            f"fssl_method={method_name!r}는 live agent runtime을 지원하지 않습니다."
        )
    capability_plan = _capability_plan_from_task_payload(capability_payload)
    validate_federated_ssl_capability_compatibility(
        method_descriptor=descriptor,
        capability_plan=capability_plan,
    )
    return capability_plan


def _uses_legacy_fssl_context_only(task_payload: object) -> bool:
    return (
        _optional_name(getattr(task_payload, "fssl_method", None)) is not None
        and getattr(task_payload, "fssl_execution", None) is None
        and getattr(task_payload, "fssl_capability_plan", None) is None
    )


def _capability_plan_from_task_payload(
    payload: Mapping[str, object],
) -> FederatedSslCapabilityPlan:
    return FederatedSslCapabilityPlan.from_mappings(
        client_participation_policy=_mapping_field(
            payload,
            "client_participation_policy",
        ),
        aggregation_weight_policy=_mapping_field(
            payload,
            "aggregation_weight_policy",
        ),
        labeled_exposure_policy=_mapping_field(
            payload,
            "labeled_exposure_policy",
        ),
        local_supervision_regime=_mapping_field(
            payload,
            "local_supervision_regime",
        ),
        server_step_policy=_mapping_field(payload, "server_step_policy"),
        peer_context_policy=_mapping_field(payload, "peer_context_policy"),
        update_partition_policy=_mapping_field(payload, "update_partition_policy"),
        query_multiview_source=_mapping_field(payload, "query_multiview_source"),
        local_ssl_policy=_mapping_field(payload, "local_ssl_policy"),
        server_update_policy=_mapping_field(payload, "server_update_policy"),
    )


def _mapping_field(
    payload: Mapping[str, object],
    field_name: str,
) -> Mapping[str, object] | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"fssl_capability_plan.{field_name} must be a mapping.")
    return value


def _optional_name(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
