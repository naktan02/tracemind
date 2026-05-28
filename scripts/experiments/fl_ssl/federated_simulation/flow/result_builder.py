"""FL simulation 결과와 report 조립."""

from __future__ import annotations

import gc
import time
from typing import Any

from scripts.experiments.fl_ssl.federated_simulation.adapters.evaluation import (
    evaluate_simulation_validation,
)
from scripts.experiments.fl_ssl.federated_simulation.flow.state import (
    ActiveSimulationState,
    BootstrappedSimulation,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    ClientEvaluationSummary,
    SimulationResult,
    SimulationRoundSummary,
    SimulationRunRequest,
)

from ..adapters.runtime_callable_loader import load_configured_callable
from ..io.final_projection import build_final_projection_artifacts
from ..io.simulation_report_builder import SimulationReportBuilder
from ..io.simulation_report_writer import SimulationReportWriter


def build_simulation_result(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
    active: ActiveSimulationState,
    round_summaries: list[SimulationRoundSummary],
) -> SimulationResult:
    """round loop 종료 후 final validation, client validation, report를 조립한다."""

    result_timing: dict[str, float] = {}
    result_started_at = time.perf_counter()
    final_validation = (
        round_summaries[-1].validation
        if round_summaries
        else bootstrapped.initial_validation
    )
    started_at = time.perf_counter()
    _release_helper_model_cache_before_final_evaluation(
        request=request,
        bootstrapped=bootstrapped,
    )
    result_timing["result_helper_cache_release_seconds"] = (
        time.perf_counter() - started_at
    )
    started_at = time.perf_counter()
    client_evaluations = _build_client_evaluations(
        request=request,
        bootstrapped=bootstrapped,
        active=active,
    )
    result_timing["result_client_evaluation_seconds"] = time.perf_counter() - started_at
    result = SimulationResult(
        initial_model_revision=bootstrapped.initial_model_revision,
        initial_validation=bootstrapped.initial_validation,
        final_validation=final_validation,
        rounds=tuple(round_summaries),
        client_evaluations=client_evaluations,
        result_timing_breakdown=result_timing,
    )
    started_at = time.perf_counter()
    final_projection_artifacts = build_final_projection_artifacts(
        request=request,
        active=active,
        runtime_resource_cache=bootstrapped.runtime_resource_cache,
    )
    result_timing["result_final_projection_seconds"] = time.perf_counter() - started_at
    if request.report_config is not None:
        started_at = time.perf_counter()
        report_payload = SimulationReportBuilder().build_payload(
            result=result,
            report_config=request.report_config,
            client_count=request.client_count,
            round_budget=request.rounds,
            bootstrap_ratio=request.bootstrap_ratio,
            seed=request.seed,
            run_budget_name=request.run_budget_name,
            run_output_dir=request.run_output_dir,
            shard_policy=request.shard_policy,
            dataset_split=bootstrapped.dataset_split,
            ssl_method_config=request.ssl_method_config,
            client_pool_split_config=request.client_pool_split_config,
            training_task_config=request.training_task_config,
            validation_config=request.validation_config,
            round_runtime_config=request.round_runtime_config,
            execution_plan=request.execution_plan,
            capability_plan=request.capability_plan,
            server_step_executor=request.server_step_executor,
            data_source_config=request.data_source_config,
            embedding_spec=request.embedding_spec,
            local_trainer_runtime_config=request.local_trainer_runtime_config,
            artifact_persistence_config=request.artifact_persistence_config,
            diagnostic_view_config=request.diagnostic_view_config,
            final_projection_artifacts=final_projection_artifacts,
            peer_probe_manifest=bootstrapped.peer_probe_manifest,
        )
        result_timing["result_report_build_seconds"] = time.perf_counter() - started_at
        diagnostics = report_payload.get("diagnostics")
        if isinstance(diagnostics, dict):
            diagnostics["result_timing_breakdown"] = dict(result_timing)
        writer = SimulationReportWriter()
        started_at = time.perf_counter()
        report_path = writer.write(
            output_dir=request.output_dir,
            report_config=request.report_config,
            payload=report_payload,
        )
        result_timing["result_report_write_seconds"] = time.perf_counter() - started_at
        result_timing["result_total_seconds"] = time.perf_counter() - result_started_at
        if isinstance(diagnostics, dict):
            diagnostics["result_timing_breakdown"] = dict(result_timing)
            writer.write(
                output_dir=request.output_dir,
                report_config=request.report_config,
                payload=report_payload,
            )
        result.report_path = str(report_path)
    else:
        result_timing["result_total_seconds"] = time.perf_counter() - result_started_at
    return result


def _build_client_evaluations(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
    active: ActiveSimulationState,
) -> tuple[ClientEvaluationSummary, ...]:
    return tuple(
        ClientEvaluationSummary(
            client_id=shard.client_id,
            validation=evaluate_simulation_validation(
                request=request,
                active=active,
                rows=shard.rows,
                objective_config=request.training_task_config.objective_config,
                runtime_resource_cache=bootstrapped.runtime_resource_cache,
            ),
        )
        for shard in bootstrapped.validation_client_shards
    )


def _release_helper_model_cache_before_final_evaluation(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
) -> None:
    """최종 평가 전에 update family가 선언한 transient cache를 정리한다."""

    cleaner_path = request.round_runtime_config.transient_resource_cleaner
    if cleaner_path:
        cleaner = _load_transient_resource_cleaner(cleaner_path)
        cleaner(bootstrapped.runtime_resource_cache)
    gc.collect()
    try:
        import torch
    except ImportError:  # pragma: no cover - optional dependency guard
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_transient_resource_cleaner(cleaner_path: str) -> Any:
    return load_configured_callable(
        cleaner_path,
        field_name="round_runtime.transient_resource_cleaner",
    )
