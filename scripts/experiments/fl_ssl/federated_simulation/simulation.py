"""Federated simulation orchestration."""

from __future__ import annotations

from typing import Any

from methods.federated_ssl.base import FederatedSslMethodDescriptor
from methods.federated_ssl.capabilities.plan import FederatedSslCapabilityPlan
from methods.federated_ssl.compatibility import (
    FederatedSslProfileCompatibilityContext,
    validate_federated_ssl_capability_compatibility,
    validate_federated_ssl_local_ssl_policy_alignment,
    validate_federated_ssl_payload_adapter_compatibility,
    validate_federated_ssl_profile_compatibility,
    validate_federated_ssl_simulation_runtime_support,
)
from methods.federated_ssl.execution_plan import (
    COMPOSITION_MODE_MANUAL,
    FederatedSslExecutionPlan,
    build_federated_ssl_execution_plan,
)
from scripts.experiments.fl_ssl.federated_simulation.adapters import (
    peer_context_exchange,
    runtime_compatibility,
    server_step_execution,
)
from scripts.experiments.fl_ssl.federated_simulation.adapters.method_runtime import (
    FederatedSslSimulationRuntime,
    build_federated_ssl_simulation_runtime,
    build_manual_federated_ssl_simulation_runtime,
)
from scripts.experiments.fl_ssl.federated_simulation.flow.bootstrap import (
    bootstrap_simulation,
)
from scripts.experiments.fl_ssl.federated_simulation.flow.result_builder import (
    build_simulation_result,
)
from scripts.experiments.fl_ssl.federated_simulation.flow.round_loop import (
    run_one_round,
)
from scripts.experiments.fl_ssl.federated_simulation.flow.state import (
    BootstrappedSimulation,
)
from scripts.experiments.fl_ssl.federated_simulation.io.resume_checkpoint import (
    write_resume_checkpoint,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedQuerySslObjectiveConfig,
    FederatedSslMethodConfig,
    SimulationResult,
    SimulationRoundSummary,
    SimulationRunRequest,
)
from scripts.runtime_adapters.federated_agent.backend_resolver import (
    resolve_federated_training_backend_adapter_kind,
)
from scripts.runtime_adapters.federated_server.runtime import (
    resolve_simulation_aggregation_backend_name,
)


def run_simulation_request(request: SimulationRunRequest) -> SimulationResult:
    """typed request 기반으로 FL SSL simulation을 실행한다."""

    execution_plan = _resolve_execution_plan(request)
    request.capability_plan = _resolve_capability_plan(request)
    ssl_method_runtime = _build_validated_ssl_runtime(
        request.ssl_method_config,
        execution_plan=execution_plan,
    )
    request.execution_plan = execution_plan
    _require_execution_plan_matches_runtime(
        request=request,
        execution_plan=execution_plan,
    )
    _require_runtime_compatibility(
        request,
        ssl_method_runtime=ssl_method_runtime,
        execution_plan=execution_plan,
    )
    bootstrapped = bootstrap_simulation(
        request,
        ssl_method_descriptor=ssl_method_runtime.descriptor,
    )
    active = bootstrapped.active
    peer_context_state = bootstrapped.peer_context_state
    client_partition_sync_state = bootstrapped.client_partition_sync_state
    query_ssl_algorithm_sync_state = bootstrapped.query_ssl_algorithm_sync_state
    round_summaries: list[SimulationRoundSummary] = list(bootstrapped.completed_rounds)
    if request.resume_config.checkpoint_enabled:
        _write_round_resume_checkpoint(
            request=request,
            bootstrapped=bootstrapped,
            round_summaries=round_summaries,
        )

    for round_index in range(len(round_summaries) + 1, request.rounds + 1):
        round_execution = run_one_round(
            request=request,
            bootstrapped=bootstrapped,
            active=active,
            ssl_method_runtime=ssl_method_runtime,
            round_index=round_index,
            peer_context_state=peer_context_state,
            client_partition_sync_state=client_partition_sync_state,
            query_ssl_algorithm_sync_state=query_ssl_algorithm_sync_state,
        )
        if round_execution.summary is None:
            break
        active = round_execution.active
        peer_context_state = round_execution.peer_context_state
        client_partition_sync_state = round_execution.client_partition_sync_state
        query_ssl_algorithm_sync_state = round_execution.query_ssl_algorithm_sync_state
        round_summaries.append(round_execution.summary)
        if request.resume_config.checkpoint_enabled:
            _write_round_resume_checkpoint(
                request=request,
                bootstrapped=bootstrapped,
                round_summaries=round_summaries,
            )

    return build_simulation_result(
        request=request,
        bootstrapped=bootstrapped,
        active=active,
        round_summaries=round_summaries,
    )


def _write_round_resume_checkpoint(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
    round_summaries: list[SimulationRoundSummary],
) -> None:
    write_resume_checkpoint(
        output_dir=request.output_dir,
        initial_model_revision=bootstrapped.initial_model_revision,
        initial_validation=bootstrapped.initial_validation,
        rounds=tuple(round_summaries),
    )


def _require_runtime_compatibility(
    request: SimulationRunRequest,
    *,
    ssl_method_runtime: FederatedSslSimulationRuntime,
    execution_plan: FederatedSslExecutionPlan,
) -> None:
    """method/local update/round runtime 조합을 bootstrap 전에 검증한다."""

    capability_plan = _resolve_capability_plan(request)
    _require_training_task_type_matches_runtime(
        request=request,
        ssl_method_runtime=ssl_method_runtime,
    )
    runtime_compatibility.require_round_runtime_matches_training_objective(request)
    local_adapter_kind = resolve_federated_training_backend_adapter_kind(
        objective_config=request.training_task_config.objective_config,
    )
    validate_federated_ssl_payload_adapter_compatibility(
        local_update_profile=request.local_update_profile,
        local_update_adapter_kind=local_adapter_kind,
        round_payload_adapter_kind=request.round_runtime_config.payload_adapter_kind,
    )
    ssl_method_descriptor = ssl_method_runtime.descriptor
    validate_federated_ssl_capability_compatibility(
        method_descriptor=ssl_method_descriptor,
        capability_plan=capability_plan,
    )
    query_ssl_objective = _resolve_query_ssl_objective_config(request)
    validate_federated_ssl_local_ssl_policy_alignment(
        capability_plan=capability_plan,
        query_ssl_algorithm_name=(
            None if query_ssl_objective is None else query_ssl_objective.algorithm_name
        ),
    )
    server_step_execution.require_supported_server_step(
        capability_plan,
        server_step_executor=request.server_step_executor,
    )
    peer_context_exchange.require_supported_peer_context(capability_plan)
    resolve_simulation_aggregation_backend_name(
        payload_adapter_kind=request.round_runtime_config.payload_adapter_kind,
        aggregation_backend_name=request.round_runtime_config.aggregation_backend_name,
        capability_plan=capability_plan,
    )
    validate_federated_ssl_simulation_runtime_support(
        capability_plan=capability_plan,
        composition_mode=execution_plan.composition_mode,
        method_descriptor=ssl_method_descriptor,
    )
    if ssl_method_descriptor is None:
        return
    validate_federated_ssl_profile_compatibility(
        FederatedSslProfileCompatibilityContext(
            method_descriptor=ssl_method_descriptor,
            local_update_profile=request.local_update_profile,
            local_update_adapter_kind=local_adapter_kind,
            round_payload_adapter_kind=request.round_runtime_config.payload_adapter_kind,
            round_update_family_name=request.round_runtime_config.update_family_name,
            round_aggregation_backend_name=(
                request.round_runtime_config.aggregation_backend_name
            ),
            capability_plan=capability_plan,
        )
    )


def _resolve_capability_plan(
    request: SimulationRunRequest,
) -> FederatedSslCapabilityPlan:
    if request.capability_plan is not None:
        return request.capability_plan
    return FederatedSslCapabilityPlan.from_mappings(
        client_participation_policy=None,
        aggregation_weight_policy=None,
        labeled_exposure_policy=request.data_source_config.labeled_exposure_policy,
        local_supervision_regime=None,
        server_step_policy=None,
        peer_context_policy=None,
        update_partition_policy=None,
        local_ssl_policy=None,
        server_update_policy=None,
        query_multiview_source=None,
    )


def _resolve_execution_plan(request: SimulationRunRequest) -> FederatedSslExecutionPlan:
    """request에서 실행 계획을 확정하고 필요한 descriptor 검증을 수행한다."""

    if request.execution_plan is not None:
        return request.execution_plan

    if request.ssl_method_config is None:
        return _build_default_manual_execution_plan(request)

    ssl_method_descriptor = build_federated_ssl_simulation_runtime(
        request.ssl_method_config.name
    ).descriptor
    if ssl_method_descriptor is None:
        raise ValueError("method-owned FL SSL runtime requires a method descriptor.")
    return build_federated_ssl_execution_plan(
        fl_method=None,
        security_policy=None,
        method_descriptor=ssl_method_descriptor,
    )


def _build_default_manual_execution_plan(
    request: SimulationRunRequest,
) -> FederatedSslExecutionPlan:
    query_ssl_objective = _resolve_query_ssl_objective_config(request)
    return build_federated_ssl_execution_plan(
        fl_method={
            "composition_mode": COMPOSITION_MODE_MANUAL,
            "manual_axes": {
                "client_ssl_objective": (
                    "pseudo_label"
                    if query_ssl_objective is None
                    else query_ssl_objective.algorithm_name
                ),
                "server_aggregation": (
                    request.round_runtime_config.aggregation_backend_name
                ),
                "update_family": request.round_runtime_config.update_family_name,
            },
        },
        security_policy=None,
        method_descriptor=None,
    )


def _require_execution_plan_matches_runtime(
    *,
    request: SimulationRunRequest,
    execution_plan: FederatedSslExecutionPlan,
) -> None:
    """manual lower axes가 실제 runtime 조합과 drift되지 않았는지 검증한다."""

    if execution_plan.composition_mode != COMPOSITION_MODE_MANUAL:
        return
    manual_axes = execution_plan.manual_axes
    query_ssl_objective = _resolve_query_ssl_objective_config(request)
    expected_client_ssl_objective = (
        None if query_ssl_objective is None else query_ssl_objective.algorithm_name
    )
    if (
        expected_client_ssl_objective is not None
        and manual_axes.client_ssl_objective != expected_client_ssl_objective
    ):
        raise ValueError(
            "manual fl_method.client_ssl_objective must match query_ssl objective: "
            f"{manual_axes.client_ssl_objective!r} != "
            f"{expected_client_ssl_objective!r}."
        )
    if manual_axes.server_aggregation != (
        request.round_runtime_config.aggregation_backend_name
    ):
        raise ValueError(
            "manual fl_method.server_aggregation must match "
            "round_runtime.aggregation_backend_name: "
            f"{manual_axes.server_aggregation!r} != "
            f"{request.round_runtime_config.aggregation_backend_name!r}."
        )
    if manual_axes.update_family != request.round_runtime_config.update_family_name:
        raise ValueError(
            "manual fl_method.update_family must match "
            "round_runtime.update_family_name: "
            f"{manual_axes.update_family!r} != "
            f"{request.round_runtime_config.update_family_name!r}."
        )


def _require_training_task_type_matches_runtime(
    *,
    request: SimulationRunRequest,
    ssl_method_runtime: FederatedSslSimulationRuntime,
) -> None:
    actual = request.training_task_config.task_type
    expected = ssl_method_runtime.training_task_type
    if str(actual) != expected:
        raise ValueError(
            "training_task_config.task_type must match the selected FL SSL runtime: "
            f"{actual!r} != {expected!r}."
        )


def _build_validated_ssl_runtime(
    ssl_method_config: FederatedSslMethodConfig | None,
    *,
    execution_plan: FederatedSslExecutionPlan,
) -> FederatedSslSimulationRuntime:
    if ssl_method_config is None:
        if execution_plan.composition_mode != COMPOSITION_MODE_MANUAL:
            raise ValueError(
                "method-owned FL SSL execution requires ssl_method_config."
            )
        execution_plan.require_manual_plan_without_descriptor()
        return build_manual_federated_ssl_simulation_runtime()

    if execution_plan.composition_mode == COMPOSITION_MODE_MANUAL:
        raise ValueError(
            "manual FL SSL execution must not provide ssl_method_config; "
            "manual_baseline is an execution role, not a descriptor."
        )
    ssl_method_runtime = build_federated_ssl_simulation_runtime(ssl_method_config.name)
    ssl_method_descriptor = ssl_method_runtime.descriptor
    if ssl_method_descriptor is None:
        raise ValueError("method-owned FL SSL runtime requires a method descriptor.")
    execution_plan.require_matches_descriptor(ssl_method_descriptor)
    _require_ssl_method_config_matches_descriptor(
        ssl_method_config,
        ssl_method_descriptor,
    )
    return ssl_method_runtime


def _require_ssl_method_config_matches_descriptor(
    ssl_method_config: FederatedSslMethodConfig,
    ssl_method_descriptor: FederatedSslMethodDescriptor,
) -> None:
    """Hydra method config가 methods spec과 다른 의미를 갖지 않게 막는다."""

    if ssl_method_config.implementation_status != (
        ssl_method_descriptor.implementation_status
    ):
        raise ValueError(
            "ssl_method implementation_status must match the registered "
            f"descriptor for {ssl_method_config.name}."
        )
    if ssl_method_config.method_role != ssl_method_descriptor.method_role:
        raise ValueError(
            "ssl_method method_role must match the registered descriptor for "
            f"{ssl_method_config.name}."
        )
    _require_mapping_value(
        ssl_method_config.client_step,
        key="task_type",
        expected=ssl_method_descriptor.local_step.step_name,
        context=f"ssl_method.client_step for {ssl_method_config.name}",
    )
    _require_mapping_value(
        ssl_method_config.client_step,
        key="custom_method_runtime_required",
        expected=(
            ssl_method_descriptor.runtime_capabilities.requires_custom_client_runtime
        ),
        context=f"ssl_method.client_step for {ssl_method_config.name}",
    )
    _require_mapping_value(
        ssl_method_config.server_step,
        key="custom_round_policy_required",
        expected=(
            ssl_method_descriptor.runtime_capabilities.requires_custom_server_runtime
        ),
        context=f"ssl_method.server_step for {ssl_method_config.name}",
    )
    expected_exchange_name = (
        None
        if ssl_method_descriptor.round_state_exchange is None
        else ssl_method_descriptor.round_state_exchange.exchange_name
    )
    _require_mapping_value(
        ssl_method_config.round_state_exchange,
        key="exchange_name",
        expected=expected_exchange_name,
        context=f"ssl_method.round_state_exchange for {ssl_method_config.name}",
    )
    expected_custom_exchange_required = (
        False
        if ssl_method_descriptor.round_state_exchange is None
        else ssl_method_descriptor.round_state_exchange.requires_custom_exchange
    )
    _require_mapping_value(
        ssl_method_config.round_state_exchange,
        key="custom_exchange_required",
        expected=expected_custom_exchange_required,
        context=f"ssl_method.round_state_exchange for {ssl_method_config.name}",
    )
    expected_metric_keys = (
        []
        if ssl_method_descriptor.round_state_exchange is None
        else list(
            ssl_method_descriptor.round_state_exchange.required_client_metric_keys
        )
    )
    _require_mapping_value(
        ssl_method_config.round_state_exchange,
        key="required_client_metric_keys",
        expected=expected_metric_keys,
        context=f"ssl_method.round_state_exchange for {ssl_method_config.name}",
    )


def _require_mapping_value(
    mapping: dict[str, object],
    *,
    key: str,
    expected: Any,
    context: str,
) -> None:
    actual = mapping.get(key)
    if actual != expected:
        raise ValueError(
            f"{context}.{key} must match the registered method descriptor: "
            f"{actual!r} != {expected!r}."
        )


def _resolve_query_ssl_objective_config(
    request: SimulationRunRequest,
) -> FederatedQuerySslObjectiveConfig | None:
    if request.query_ssl_objective_config is not None:
        return request.query_ssl_objective_config
    return FederatedQuerySslObjectiveConfig.from_objective_config(
        request.training_task_config.objective_config
    )
