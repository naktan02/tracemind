"""FL simulation 결과와 report 조립."""

from __future__ import annotations

from scripts.experiments.fl_ssl.federated_simulation.adapters.evaluation import (
    build_validation_scoring_service,
    evaluate_rows,
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

    final_validation = (
        round_summaries[-1].validation
        if round_summaries
        else bootstrapped.initial_validation
    )
    result = SimulationResult(
        initial_model_revision=bootstrapped.initial_model_revision,
        initial_prototype_version=bootstrapped.initial_prototype_version,
        initial_validation=bootstrapped.initial_validation,
        final_validation=final_validation,
        rounds=tuple(round_summaries),
        client_evaluations=_build_client_evaluations(
            request=request,
            bootstrapped=bootstrapped,
            active=active,
        ),
    )
    if request.report_config is not None:
        report_payload = SimulationReportBuilder().build_payload(
            result=result,
            report_config=request.report_config,
            client_count=request.client_count,
            round_budget=request.rounds,
            bootstrap_ratio=request.bootstrap_ratio,
            seed=request.seed,
            shard_policy=request.shard_policy,
            dataset_split=bootstrapped.dataset_split,
            ssl_method_config=request.ssl_method_config,
            client_pool_split_config=request.client_pool_split_config,
            training_task_config=request.training_task_config,
            validation_config=request.validation_config,
            round_runtime_config=request.round_runtime_config,
            execution_plan=request.execution_plan,
        )
        result.report_path = str(
            SimulationReportWriter().write(
                output_dir=request.output_dir,
                report_config=request.report_config,
                payload=report_payload,
            )
        )
    return result


def _build_client_evaluations(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
    active: ActiveSimulationState,
) -> tuple[ClientEvaluationSummary, ...]:
    final_validation_scoring_service = build_validation_scoring_service(
        request.validation_config,
        shared_state=active.adapter_state,
    )
    return tuple(
        ClientEvaluationSummary(
            client_id=shard.client_id,
            validation=evaluate_rows(
                rows=shard.rows,
                adapter=bootstrapped.adapter,
                adapter_state=active.adapter_state,
                prototype_pack=active.prototype_pack,
                model_id=request.model_id,
                scoring_service=final_validation_scoring_service,
                confidence_threshold=request.validation_config.confidence_threshold,
                margin_threshold=request.validation_config.margin_threshold,
                objective_config=request.training_task_config.objective_config,
            ),
        )
        for shard in bootstrapped.validation_client_shards
    )
