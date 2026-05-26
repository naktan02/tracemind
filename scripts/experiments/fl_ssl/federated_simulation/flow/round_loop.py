"""FL simulation round lifecycle 실행 단계."""

from __future__ import annotations

import time

from methods.federated.participation import select_participating_clients
from methods.federated_ssl.capability_plan import FederatedSslCapabilityPlan
from scripts.experiments.fl_ssl.federated_simulation.adapters import (
    peer_context_exchange,
    server_step_execution,
)
from scripts.experiments.fl_ssl.federated_simulation.adapters.client_training import (
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
    ClientPartitionSyncSimulationState,
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
    client_partition_sync_state: ClientPartitionSyncSimulationState,
) -> RoundExecution:
    """한 communication round를 열고 client update를 모아 publication까지 진행한다."""

    round_started_at = time.perf_counter()
    round_timing: dict[str, float] = {}
    round_id = f"round_{round_index:04d}"
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

    started_at = time.perf_counter()
    server_step = server_step_execution.run_server_step_if_supported(
        request=request,
        bootstrapped=bootstrapped,
        active=active,
        capability_plan=capability_plan,
        round_index=round_index,
    )
    active = server_step.active
    round_timing["round_server_step_seconds"] = time.perf_counter() - started_at

    started_at = time.perf_counter()
    if bootstrapped.round_base_snapshot_cache is not None:
        bootstrapped.round_base_snapshot_cache.clear()
    round_timing["round_base_snapshot_cache_clear_seconds"] = (
        time.perf_counter() - started_at
    )

    started_at = time.perf_counter()
    round_record = bootstrapped.server_runtime.open_round(
        ssl_method_runtime.build_round_open_request(
            round_id=round_id,
            training_task_config=request.training_task_config,
        )
    )
    round_timing["round_open_seconds"] = time.perf_counter() - started_at
    training_task = round_record.training_task

    started_at = time.perf_counter()
    selected_shards, participation_selection = select_participating_clients(
        clients=bootstrapped.dataset_split.client_shards,
        policy=capability_plan.client_participation_policy,
        seed=request.seed,
        round_index=round_index,
    )
    round_timing["round_client_selection_seconds"] = time.perf_counter() - started_at
    skipped_client_ids = tuple(
        bootstrapped.dataset_split.client_shards[index].client_id
        for index in participation_selection.skipped_indices
    )

    started_at = time.perf_counter()
    peer_context_by_client = peer_context_exchange.build_peer_context_by_client(
        capability_plan=capability_plan,
        ssl_method_config=request.ssl_method_config,
        selected_client_ids=tuple(shard.client_id for shard in selected_shards),
        round_index=round_index,
        client_vectors=peer_context_state.selection_vectors(),
    )
    round_timing["round_peer_context_prepare_seconds"] = (
        time.perf_counter() - started_at
    )

    started_at = time.perf_counter()
    client_executions = tuple(
        run_client_round(
            request=request,
            bootstrapped=bootstrapped,
            active=active,
            ssl_method_runtime=ssl_method_runtime,
            round_id=round_id,
            shard=shard,
            training_task=training_task,
            capability_plan=capability_plan,
            peer_context=peer_context_by_client.get(shard.client_id),
            peer_snapshots=peer_context_state.client_snapshots,
            previous_client_partition_parameters=(
                client_partition_sync_state.snapshot_for_client(shard.client_id)
            ),
        )
        for shard in selected_shards
    )
    round_timing["round_client_execution_seconds"] = time.perf_counter() - started_at
    update_count = sum(
        1 for execution in client_executions if execution.update_submitted
    )
    if update_count == 0:
        return RoundExecution(
            active=active,
            peer_context_state=peer_context_state,
            client_partition_sync_state=client_partition_sync_state,
            summary=None,
        )
    started_at = time.perf_counter()
    next_peer_context_state = _build_next_peer_context_state(
        previous=peer_context_state,
        client_executions=client_executions,
    )
    next_client_partition_sync_state = _build_next_client_partition_sync_state(
        previous=client_partition_sync_state,
        client_executions=client_executions,
    )
    round_timing["round_peer_state_build_seconds"] = time.perf_counter() - started_at

    next_model_revision = f"sim_rev_{round_index:04d}"
    started_at = time.perf_counter()
    next_active = _finalize_round_publication(
        request=request,
        bootstrapped=bootstrapped,
        round_id=round_id,
        next_model_revision=next_model_revision,
    )
    round_timing["round_finalize_publication_seconds"] = (
        time.perf_counter() - started_at
    )
    started_at = time.perf_counter()
    validation = evaluate_simulation_validation(
        request=request,
        active=next_active,
        rows=request.validation_rows,
        objective_config=training_task.objective_config,
        runtime_resource_cache=bootstrapped.runtime_resource_cache,
    )
    round_timing["round_validation_seconds"] = time.perf_counter() - started_at
    round_elapsed = time.perf_counter() - round_started_at
    round_timing["round_total_seconds"] = round_elapsed
    measured_without_total = sum(
        value
        for key, value in round_timing.items()
        if key not in {"round_total_seconds", "round_unattributed_seconds"}
    )
    round_timing["round_unattributed_seconds"] = max(
        0.0,
        round_elapsed - measured_without_total,
    )
    return RoundExecution(
        active=next_active,
        peer_context_state=next_peer_context_state,
        client_partition_sync_state=next_client_partition_sync_state,
        summary=SimulationRoundSummary(
            round_id=round_id,
            model_revision=next_model_revision,
            update_count=update_count,
            validation=validation,
            clients=tuple(execution.summary for execution in client_executions),
            round_time_seconds=round_elapsed,
            round_timing_breakdown=round_timing,
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


def _build_next_client_partition_sync_state(
    *,
    previous: ClientPartitionSyncSimulationState,
    client_executions: tuple[ClientRoundExecution, ...],
) -> ClientPartitionSyncSimulationState:
    snapshots = dict(previous.client_partition_snapshots)
    for execution in client_executions:
        if not execution.client_partition_snapshot:
            continue
        snapshots[execution.summary.client_id] = execution.client_partition_snapshot
    return ClientPartitionSyncSimulationState(client_partition_snapshots=snapshots)


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
