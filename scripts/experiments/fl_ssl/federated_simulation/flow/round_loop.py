"""FL simulation round lifecycle 실행 단계."""

from __future__ import annotations

import time

from methods.federated.participation import select_participating_clients
from methods.federated_ssl.capability_plan import FederatedSslCapabilityPlan
from scripts.experiments.fl_ssl.federated_simulation.adapters import (
    peer_context_exchange,
)
from scripts.experiments.fl_ssl.federated_simulation.adapters.client_training import (
    build_round_training_scoring_service,
    run_client_round,
)
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
    PeerContextSimulationState,
    RoundExecution,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    SimulationRoundSummary,
    SimulationRunRequest,
)

from ..io.run_artifact_writer import RunArtifactWriter


def run_one_round(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
    active: ActiveSimulationState,
    ssl_method_runtime: FederatedSslSimulationRuntime,
    round_index: int,
    peer_context_state: PeerContextSimulationState,
) -> RoundExecution:
    """한 communication round를 열고 client update를 모아 publication까지 진행한다."""

    round_started_at = time.perf_counter()
    round_id = f"round_{round_index:04d}"
    round_record = bootstrapped.server_runtime.open_round(
        ssl_method_runtime.build_round_open_request(
            round_id=round_id,
            training_task_config=request.training_task_config,
        )
    )
    training_task = round_record.training_task
    training_scoring_service = build_round_training_scoring_service(
        request=request,
        active=active,
        training_task=training_task,
    )

    capability_plan = (
        request.capability_plan
        or FederatedSslCapabilityPlan.from_mappings(
            client_participation_policy=None,
            aggregation_weight_policy=None,
            labeled_exposure_policy=None,
            local_supervision_regime=None,
            server_step_policy=None,
            peer_context_policy=None,
            update_partition_policy=None,
            query_multiview_source=None,
        )
    )
    selected_shards, participation_selection = select_participating_clients(
        clients=bootstrapped.dataset_split.client_shards,
        policy=capability_plan.client_participation_policy,
        seed=request.seed,
        round_index=round_index,
    )
    skipped_client_ids = tuple(
        bootstrapped.dataset_split.client_shards[index].client_id
        for index in participation_selection.skipped_indices
    )
    peer_context_by_client = peer_context_exchange.build_peer_context_by_client(
        capability_plan=capability_plan,
        ssl_method_config=request.ssl_method_config,
        selected_client_ids=tuple(shard.client_id for shard in selected_shards),
        round_index=round_index,
        client_vectors=peer_context_state.selection_vectors(),
    )

    client_executions = tuple(
        run_client_round(
            request=request,
            bootstrapped=bootstrapped,
            active=active,
            ssl_method_runtime=ssl_method_runtime,
            round_id=round_id,
            shard=shard,
            training_task=training_task,
            training_scoring_service=training_scoring_service,
            peer_context=peer_context_by_client.get(shard.client_id),
            peer_snapshots=peer_context_state.client_snapshots,
        )
        for shard in selected_shards
    )
    update_count = sum(
        1 for execution in client_executions if execution.update_submitted
    )
    if update_count == 0:
        return RoundExecution(
            active=active,
            peer_context_state=peer_context_state,
            summary=None,
        )
    next_peer_context_state = _build_next_peer_context_state(
        previous=peer_context_state,
        client_executions=client_executions,
    )

    next_model_revision = f"sim_rev_{round_index:04d}"
    next_active = _finalize_round_publication(
        request=request,
        bootstrapped=bootstrapped,
        round_id=round_id,
        next_model_revision=next_model_revision,
    )
    validation = evaluate_simulation_validation(
        request=request,
        active=next_active,
        rows=request.validation_rows,
        objective_config=training_task.objective_config,
    )
    return RoundExecution(
        active=next_active,
        peer_context_state=next_peer_context_state,
        summary=SimulationRoundSummary(
            round_id=round_id,
            model_revision=next_model_revision,
            update_count=update_count,
            validation=validation,
            clients=tuple(execution.summary for execution in client_executions),
            round_time_seconds=time.perf_counter() - round_started_at,
            total_payload_bytes=sum(
                execution.summary.client_payload_bytes or 0
                for execution in client_executions
            ),
            total_client_count=len(bootstrapped.dataset_split.client_shards),
            selected_client_count=participation_selection.selected_count,
            skipped_client_count=participation_selection.skipped_count,
            skipped_client_ids=skipped_client_ids,
        ),
    )


def _build_next_peer_context_state(
    *,
    previous: PeerContextSimulationState,
    client_executions: tuple[ClientRoundExecution, ...],
) -> PeerContextSimulationState:
    snapshots = dict(previous.client_snapshots)
    for execution in client_executions:
        if execution.peer_client_snapshot is None:
            continue
        snapshots[execution.peer_client_snapshot.client_id] = (
            execution.peer_client_snapshot
        )
    return PeerContextSimulationState(client_snapshots=snapshots)


def _finalize_round_publication(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
    round_id: str,
    next_model_revision: str,
) -> ActiveSimulationState:
    finalized_round = bootstrapped.server_runtime.finalize_round(
        round_id=round_id,
        next_model_revision=next_model_revision,
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
    return ActiveSimulationState(
        manifest=active_manifest,
        adapter_state=active_state,
    )
