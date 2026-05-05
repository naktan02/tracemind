"""Federated simulation orchestration."""

from __future__ import annotations

from pathlib import Path

from methods.federated.shard_policy.base import FederatedShardPolicyConfig
from methods.prototype.building.base import PrototypeBuildStrategy
from scripts.experiments.fl_ssl.federated_simulation.bootstrap import (
    bootstrap_simulation,
)
from scripts.experiments.fl_ssl.federated_simulation.method_runtime import (
    FederatedSslSimulationRuntime,
    build_federated_ssl_simulation_runtime,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
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
from scripts.experiments.fl_ssl.federated_simulation.result_builder import (
    build_simulation_result,
)
from scripts.experiments.fl_ssl.federated_simulation.round_loop import run_one_round
from scripts.io.labeled_query_rows import LabeledQueryRow
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
        report_config=report_config,
    )
    return run_simulation_request(request)


def run_simulation_request(request: SimulationRunRequest) -> SimulationResult:
    """typed request 기반으로 FL SSL simulation을 실행한다."""

    ssl_method_runtime = _build_validated_ssl_runtime(request.ssl_method_config)
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


def _build_validated_ssl_runtime(
    ssl_method_config: FederatedSslMethodConfig,
) -> FederatedSslSimulationRuntime:
    ssl_method_runtime = build_federated_ssl_simulation_runtime(ssl_method_config.name)
    ssl_method_descriptor = ssl_method_runtime.descriptor
    if ssl_method_config.implementation_status != (
        ssl_method_descriptor.implementation_status
    ):
        raise ValueError(
            "ssl_method implementation_status must match the registered "
            f"descriptor for {ssl_method_config.name}."
        )
    return ssl_method_runtime
