"""FL simulation client local training adapter."""

from __future__ import annotations

import time
from typing import Any

from methods.evaluation.pseudo_label_quality import (
    build_pseudo_label_quality_summary,
)
from scripts.experiments.fl_ssl.federated_simulation.adapters.method_runtime import (
    FederatedClientLocalTrainingContext,
    FederatedSslSimulationRuntime,
)
from scripts.experiments.fl_ssl.federated_simulation.flow.state import (
    ActiveSimulationState,
    BootstrappedSimulation,
    ClientRoundExecution,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    ClientRoundSummary,
    FederatedClientShard,
    SimulationRunRequest,
)
from scripts.runtime_adapters.federated_agent.scoring_runtime import (
    build_federated_scoring_service,
)
from scripts.runtime_adapters.federated_agent.training_runtime import (
    run_federated_local_training,
)

from ..io.selection_diagnostics_writer import SelectionDiagnosticsWriter
from .client_update_submission import (
    accept_client_update,
    extract_aggregation_example_count,
    extract_delta_l2_norm,
    payload_byte_count,
)
from .local_objective_execution import run_method_or_manual_local_objective_if_supported


def build_round_training_scoring_service(
    *,
    request: SimulationRunRequest,
    active: ActiveSimulationState,
    training_task: Any,
) -> Any:
    """현재 global state 기준 client local training scoring service를 만든다."""

    return build_federated_scoring_service(
        objective_config=training_task.objective_config,
        similarity_name=request.validation_config.similarity_name,
        shared_state=active.adapter_state,
    )


def run_client_round(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
    active: ActiveSimulationState,
    ssl_method_runtime: FederatedSslSimulationRuntime,
    round_id: str,
    shard: FederatedClientShard,
    training_task: Any,
    training_scoring_service: Any,
) -> ClientRoundExecution:
    """client shard 하나의 local training을 실행하고 update를 제출한다."""

    query_ssl_execution = run_method_or_manual_local_objective_if_supported(
        request=request,
        bootstrapped=bootstrapped,
        active=active,
        round_id=round_id,
        shard=shard,
        training_task=training_task,
    )
    if query_ssl_execution is not None:
        return query_ssl_execution

    training_plan = ssl_method_runtime.build_local_training_plan(
        context=FederatedClientLocalTrainingContext(
            shard=shard,
            adapter=bootstrapped.adapter,
            adapter_state=active.adapter_state,
            prototype_pack=active.prototype_pack,
            model_id=request.model_id,
            scoring_service=training_scoring_service,
            objective_config=training_task.objective_config,
            client_state_root=request.output_dir / "agents" / shard.client_id,
            training_task=training_task,
        )
    )
    training_started_at = time.perf_counter()
    local_result = run_federated_local_training(
        local_training_service=training_plan.service,
        training_examples=training_plan.examples,
        training_task=training_task,
        model_manifest=active.manifest,
    )
    client_train_time_seconds = time.perf_counter() - training_started_at
    SelectionDiagnosticsWriter().save(
        output_dir=request.output_dir,
        round_id=round_id,
        client_id=shard.client_id,
        rows=training_plan.rows,
        training_examples=training_plan.examples,
        selection_result=local_result.selection_result,
        diagnostics_config=request.diagnostics_config,
    )
    update_submitted = accept_client_update(
        server_runtime=bootstrapped.server_runtime,
        round_id=round_id,
        update_envelope=local_result.update_envelope,
        update_payload=local_result.update_payload,
    )
    selection_quality = build_pseudo_label_quality_summary(
        candidates=tuple(local_result.selection_result.candidates),
        rows_with_simulation_labels=training_plan.rows,
    )
    return ClientRoundExecution(
        summary=ClientRoundSummary(
            client_id=shard.client_id,
            candidate_count=local_result.selection_result.total_count,
            accepted_count=local_result.selection_result.accepted_count,
            update_generated=update_submitted,
            delta_l2_norm=extract_delta_l2_norm(local_result.update_envelope),
            aggregation_example_count=extract_aggregation_example_count(
                local_result.update_envelope
            ),
            client_train_time_seconds=client_train_time_seconds,
            client_payload_bytes=(
                payload_byte_count(local_result.update_payload)
                if update_submitted
                else None
            ),
            pseudo_label_confidence_mean=(
                selection_quality.pseudo_label_confidence_mean
            ),
            pseudo_label_margin_mean=selection_quality.pseudo_label_margin_mean,
            pseudo_label_correct_count=selection_quality.pseudo_label_correct_count,
            pseudo_label_evaluated_count=selection_quality.pseudo_label_evaluated_count,
            accepted_label_distribution=selection_quality.accepted_label_distribution,
            rejected_label_distribution=selection_quality.rejected_label_distribution,
        ),
        update_submitted=update_submitted,
    )
