"""Federated simulation orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from methods.federated.shard_policy.base import FederatedShardPolicyConfig
from methods.federated_ssl.base import FederatedSslMethodDescriptor
from methods.federated_ssl.compatibility import (
    FederatedSslProfileCompatibilityContext,
    validate_federated_ssl_profile_compatibility,
)
from methods.prototype.building.base import PrototypeBuildStrategy
from scripts.experiments.fl_ssl.federated_simulation.adapters.method_runtime import (
    FederatedSslSimulationRuntime,
    build_federated_ssl_simulation_runtime,
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
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedClientPoolSplitConfig,
    FederatedDiagnosticsConfig,
    FederatedPrototypeRebuildConfig,
    FederatedReportConfig,
    FederatedRoundRuntimeConfig,
    FederatedSslMethodConfig,
    FederatedTrainingTaskConfig,
    FederatedValidationConfig,
    SimulationResult,
    SimulationRoundSummary,
    SimulationRunRequest,
)
from scripts.io.labeled_query_rows import LabeledQueryRow
from scripts.runtime_adapters.federated_agent_runtime import (
    resolve_federated_training_backend_adapter_kind,
)
from shared.src.domain.value_objects.embedding_adapter_spec import EmbeddingAdapterSpec


def run_simulation(
    *,
    train_rows: list[LabeledQueryRow],
    validation_rows: list[LabeledQueryRow],
    output_dir: Path,
    client_count: int,
    rounds: int,
    bootstrap_ratio: float,
    seed: int,
    embedding_spec: EmbeddingAdapterSpec,
    model_id: str,
    training_scope: str,
    round_runtime_config: FederatedRoundRuntimeConfig,
    prototype_build_strategy: PrototypeBuildStrategy,
    shard_policy: FederatedShardPolicyConfig,
    training_task_config: FederatedTrainingTaskConfig,
    validation_config: FederatedValidationConfig,
    prototype_rebuild_config: FederatedPrototypeRebuildConfig,
    diagnostics_config: FederatedDiagnosticsConfig,
    ssl_method_config: FederatedSslMethodConfig,
    client_pool_split_config: FederatedClientPoolSplitConfig | None = None,
    report_config: FederatedReportConfig | None = None,
) -> SimulationResult:
    """bootstrap -> client pseudo-label -> aggregate -> republish 루프를 실행한다."""

    request = SimulationRunRequest(
        train_rows=train_rows,
        validation_rows=validation_rows,
        output_dir=output_dir,
        client_count=client_count,
        rounds=rounds,
        bootstrap_ratio=bootstrap_ratio,
        seed=seed,
        embedding_spec=embedding_spec,
        model_id=model_id,
        training_scope=training_scope,
        round_runtime_config=round_runtime_config,
        prototype_build_strategy=prototype_build_strategy,
        shard_policy=shard_policy,
        training_task_config=training_task_config,
        validation_config=validation_config,
        prototype_rebuild_config=prototype_rebuild_config,
        diagnostics_config=diagnostics_config,
        ssl_method_config=ssl_method_config,
        client_pool_split_config=client_pool_split_config,
        report_config=report_config,
    )
    return run_simulation_request(request)


def run_simulation_request(request: SimulationRunRequest) -> SimulationResult:
    """typed request 기반으로 FL SSL simulation을 실행한다."""

    ssl_method_runtime = _build_validated_ssl_runtime(request.ssl_method_config)
    _require_fl_profile_compatibility(request, ssl_method_runtime.descriptor)
    bootstrapped = bootstrap_simulation(request)
    active = bootstrapped.active
    round_summaries: list[SimulationRoundSummary] = []

    for round_index in range(1, request.rounds + 1):
        round_execution = run_one_round(
            request=request,
            bootstrapped=bootstrapped,
            active=active,
            ssl_method_runtime=ssl_method_runtime,
            round_index=round_index,
        )
        if round_execution.summary is None:
            break
        active = round_execution.active
        round_summaries.append(round_execution.summary)

    return build_simulation_result(
        request=request,
        bootstrapped=bootstrapped,
        active=active,
        round_summaries=round_summaries,
    )


def _require_fl_profile_compatibility(
    request: SimulationRunRequest,
    ssl_method_descriptor: FederatedSslMethodDescriptor,
) -> None:
    """method/local update/round runtime 조합을 bootstrap 전에 검증한다."""

    local_adapter_kind = resolve_federated_training_backend_adapter_kind(
        objective_config=request.training_task_config.objective_config,
    )
    validate_federated_ssl_profile_compatibility(
        FederatedSslProfileCompatibilityContext(
            method_descriptor=ssl_method_descriptor,
            local_update_profile=request.local_update_profile,
            local_update_adapter_kind=local_adapter_kind,
            round_adapter_family_name=request.round_runtime_config.adapter_family_name,
            round_aggregation_backend_name=(
                request.round_runtime_config.aggregation_backend_name
            ),
            round_runtime_profile_name=request.round_runtime_config.profile_name,
        )
    )


def _build_validated_ssl_runtime(
    ssl_method_config: FederatedSslMethodConfig,
) -> FederatedSslSimulationRuntime:
    ssl_method_runtime = build_federated_ssl_simulation_runtime(ssl_method_config.name)
    ssl_method_descriptor = ssl_method_runtime.descriptor
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
