"""FL simulation round 실행 단계."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.experiments.fl_ssl.federated_simulation.adapters.evaluation import (
    evaluate_simulation_validation,
)
from scripts.experiments.fl_ssl.federated_simulation.adapters.method_runtime import (
    FederatedSslSimulationRuntime,
)
from scripts.experiments.fl_ssl.federated_simulation.flow.state import (
    ActiveSimulationState,
    BootstrappedSimulation,
    ClientRoundExecution,
    RoundExecution,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    ClientRoundSummary,
    FederatedClientShard,
    SimulationRoundSummary,
    SimulationRunRequest,
)
from scripts.runtime_adapters.federated_agent.scoring_runtime import (
    build_federated_scoring_service,
)
from scripts.runtime_adapters.federated_agent.training_runtime import (
    run_federated_local_training,
)
from scripts.runtime_adapters.federated_server.runtime import SimulationServerRuntime
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.prototype_contracts import load_prototype_pack_payload
from shared.src.contracts.training_contracts import (
    ClientMetricKeys,
    TrainingUpdateEnvelope,
)

from ..io.run_artifact_writer import RunArtifactWriter
from ..io.selection_diagnostics_writer import SelectionDiagnosticsWriter


def run_one_round(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
    active: ActiveSimulationState,
    ssl_method_runtime: FederatedSslSimulationRuntime,
    round_index: int,
) -> RoundExecution:
    """한 communication round를 열고 client update를 모아 publication까지 진행한다."""

    round_id = f"round_{round_index:04d}"
    round_record = bootstrapped.server_runtime.open_round(
        ssl_method_runtime.build_round_open_request(
            active_manifest=active.manifest,
            round_id=round_id,
            training_task_config=request.training_task_config,
        )
    )
    training_task = round_record.training_task
    training_scoring_service = build_federated_scoring_service(
        objective_config=training_task.objective_config,
        similarity_name=request.validation_config.similarity_name,
        shared_state=active.adapter_state,
    )

    client_executions = tuple(
        _run_client_round(
            request=request,
            bootstrapped=bootstrapped,
            active=active,
            ssl_method_runtime=ssl_method_runtime,
            round_id=round_id,
            shard=shard,
            training_task=training_task,
            training_scoring_service=training_scoring_service,
        )
        for shard in bootstrapped.dataset_split.client_shards
    )
    update_count = sum(
        1 for execution in client_executions if execution.update_submitted
    )
    if update_count == 0:
        return RoundExecution(active=active, summary=None)

    next_model_revision = f"sim_rev_{round_index:04d}"
    next_prototype_version = f"proto_sim_{round_index:04d}"
    next_active = _finalize_round_publication(
        request=request,
        bootstrapped=bootstrapped,
        round_id=round_id,
        next_model_revision=next_model_revision,
        next_prototype_version=next_prototype_version,
    )
    validation = evaluate_simulation_validation(
        request=request,
        adapter=bootstrapped.adapter,
        active=next_active,
        rows=request.validation_rows,
        objective_config=training_task.objective_config,
    )
    return RoundExecution(
        active=next_active,
        summary=SimulationRoundSummary(
            round_id=round_id,
            model_revision=next_model_revision,
            prototype_version=next_prototype_version,
            update_count=update_count,
            validation=validation,
            clients=tuple(execution.summary for execution in client_executions),
        ),
    )


def _run_client_round(
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
    training_rows = _resolve_client_training_rows(shard)
    training_examples = ssl_method_runtime.build_training_examples(
        rows=training_rows,
        adapter=bootstrapped.adapter,
        adapter_state=active.adapter_state,
        prototype_pack=active.prototype_pack,
        model_id=request.model_id,
        scoring_service=training_scoring_service,
        objective_config=training_task.objective_config,
    )
    local_training_service = ssl_method_runtime.build_local_training_service(
        client_state_root=request.output_dir / "agents" / shard.client_id
    )
    local_result = run_federated_local_training(
        local_training_service=local_training_service,
        training_examples=training_examples,
        training_task=training_task,
        model_manifest=active.manifest,
    )
    SelectionDiagnosticsWriter().save(
        output_dir=request.output_dir,
        round_id=round_id,
        client_id=shard.client_id,
        rows=training_rows,
        training_examples=training_examples,
        selection_result=local_result.selection_result,
        diagnostics_config=request.diagnostics_config,
    )
    update_submitted = _accept_client_update(
        server_runtime=bootstrapped.server_runtime,
        round_id=round_id,
        update_envelope=local_result.update_envelope,
        update_payload=local_result.update_payload,
    )
    return ClientRoundExecution(
        summary=ClientRoundSummary(
            client_id=shard.client_id,
            candidate_count=local_result.selection_result.total_count,
            accepted_count=local_result.selection_result.accepted_count,
            update_generated=update_submitted,
            delta_l2_norm=_extract_delta_l2_norm(local_result.update_envelope),
        ),
        update_submitted=update_submitted,
    )


def _resolve_client_training_rows(
    shard: FederatedClientShard,
) -> list[LabeledQueryRow]:
    if shard.client_pool_split_enforced:
        return list(shard.unlabeled_rows)
    return list(shard.rows)


def _accept_client_update(
    *,
    server_runtime: SimulationServerRuntime,
    round_id: str,
    update_envelope: TrainingUpdateEnvelope | None,
    update_payload: Any | None,
) -> bool:
    if update_envelope is None:
        return False
    if update_payload is None:
        raise ValueError("Update envelope exists without update payload.")
    server_runtime.accept_update(
        round_id,
        update_envelope,
        update_payload,
    )
    return True


def _extract_delta_l2_norm(
    update_envelope: TrainingUpdateEnvelope | None,
) -> float | None:
    if update_envelope is None:
        return None
    value = update_envelope.client_metrics.get(ClientMetricKeys.DELTA_L2_NORM)
    if value is None:
        return None
    return float(value)


def _finalize_round_publication(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
    round_id: str,
    next_model_revision: str,
    next_prototype_version: str,
) -> ActiveSimulationState:
    finalized_round = bootstrapped.server_runtime.finalize_round(
        round_id=round_id,
        next_model_revision=next_model_revision,
        next_prototype_version=next_prototype_version,
    )
    if finalized_round.publication is None:
        raise ValueError("Finalized simulation round must contain publication.")
    run_artifact_writer = RunArtifactWriter()
    active_manifest = finalized_round.publication.next_manifest
    active_state = bootstrapped.server_runtime.load_active_state(active_manifest)
    run_artifact_writer.save_model_manifest(
        output_dir=request.output_dir,
        manifest=active_manifest,
    )
    if finalized_round.publication.prototype_pack_ref is None:
        raise ValueError("Simulation finalize must publish a prototype pack reference.")
    active_prototype = load_prototype_pack_payload(
        Path(finalized_round.publication.prototype_pack_ref)
    )
    run_artifact_writer.save_prototype_pack(
        output_dir=request.output_dir,
        payload=active_prototype,
    )
    return ActiveSimulationState(
        manifest=active_manifest,
        adapter_state=active_state,
        prototype_pack=active_prototype,
    )
