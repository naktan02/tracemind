"""FL simulation round lifecycle 실행 단계."""

from __future__ import annotations

import gc
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from methods.federated.participation import (
    ParticipationSelection,
    select_participating_clients,
)
from methods.federated_ssl.capabilities.plan import FederatedSslCapabilityPlan
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
    QuerySslAlgorithmSyncSimulationState,
    RoundExecution,
)
from scripts.experiments.fl_ssl.federated_simulation.model_revisions import (
    build_simulation_model_revision,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedClientShard,
    SimulationEvaluation,
    SimulationRoundSummary,
    SimulationRunRequest,
)
from scripts.support.configured_callable import load_configured_callable
from shared.src.contracts.training_contracts import TrainingTask

from ..io.run_artifact_writer import RunArtifactWriter


@dataclass(frozen=True, slots=True)
class _ClientSelection:
    selected_shards: tuple[FederatedClientShard, ...]
    participation_selection: ParticipationSelection
    skipped_client_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _RoundSyncStates:
    peer_context_state: PeerContextSimulationState
    client_partition_sync_state: ClientPartitionSyncSimulationState
    query_ssl_algorithm_sync_state: QuerySslAlgorithmSyncSimulationState


def run_one_round(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
    active: ActiveSimulationState,
    ssl_method_runtime: FederatedSslSimulationRuntime,
    round_index: int,
    peer_context_state: PeerContextSimulationState,
    client_partition_sync_state: ClientPartitionSyncSimulationState,
    query_ssl_algorithm_sync_state: QuerySslAlgorithmSyncSimulationState,
) -> RoundExecution:
    """한 communication round를 열고 client update를 모아 publication까지 진행한다."""

    round_started_at = time.perf_counter()
    round_timing: dict[str, float] = {}
    round_id = f"round_{round_index:04d}"
    capability_plan = _require_round_capability_plan(request)
    active = _run_server_step_phase(
        request=request,
        bootstrapped=bootstrapped,
        active=active,
        capability_plan=capability_plan,
        round_index=round_index,
        round_timing=round_timing,
    )
    _clear_round_base_snapshot_cache(
        bootstrapped=bootstrapped,
        round_timing=round_timing,
    )
    training_task = _open_round_phase(
        request=request,
        bootstrapped=bootstrapped,
        ssl_method_runtime=ssl_method_runtime,
        round_id=round_id,
        round_timing=round_timing,
    )
    client_selection = _select_clients_phase(
        request=request,
        bootstrapped=bootstrapped,
        capability_plan=capability_plan,
        round_index=round_index,
        round_timing=round_timing,
    )
    peer_context_by_client = _prepare_peer_context_phase(
        capability_plan=capability_plan,
        request=request,
        selected_shards=client_selection.selected_shards,
        round_index=round_index,
        peer_context_state=peer_context_state,
        round_timing=round_timing,
    )
    client_executions = _train_clients_phase(
        request=request,
        bootstrapped=bootstrapped,
        active=active,
        ssl_method_runtime=ssl_method_runtime,
        round_id=round_id,
        training_task=training_task,
        capability_plan=capability_plan,
        client_selection=client_selection,
        peer_context_by_client=peer_context_by_client,
        peer_context_state=peer_context_state,
        client_partition_sync_state=client_partition_sync_state,
        query_ssl_algorithm_sync_state=query_ssl_algorithm_sync_state,
        round_timing=round_timing,
    )
    update_count = sum(
        1 for execution in client_executions if execution.update_submitted
    )
    if update_count == 0:
        _record_round_transient_resource_cleanup(
            request=request,
            bootstrapped=bootstrapped,
            round_timing=round_timing,
        )
        return RoundExecution(
            active=active,
            peer_context_state=peer_context_state,
            client_partition_sync_state=client_partition_sync_state,
            query_ssl_algorithm_sync_state=query_ssl_algorithm_sync_state,
            summary=None,
        )

    sync_states = _build_sync_state_phase(
        peer_context_state=peer_context_state,
        client_partition_sync_state=client_partition_sync_state,
        query_ssl_algorithm_sync_state=query_ssl_algorithm_sync_state,
        client_executions=client_executions,
        round_timing=round_timing,
    )
    next_model_revision = build_simulation_model_revision(round_index)
    next_active, aggregation_metrics = _finalize_publication_phase(
        request=request,
        bootstrapped=bootstrapped,
        round_id=round_id,
        next_model_revision=next_model_revision,
        round_timing=round_timing,
    )
    validation = _evaluate_validation_phase(
        request=request,
        active=next_active,
        bootstrapped=bootstrapped,
        training_task=training_task,
        round_timing=round_timing,
    )
    _record_round_transient_resource_cleanup(
        request=request,
        bootstrapped=bootstrapped,
        round_timing=round_timing,
    )
    summary = _assemble_round_summary(
        bootstrapped=bootstrapped,
        round_id=round_id,
        next_model_revision=next_model_revision,
        validation=validation,
        client_executions=client_executions,
        update_count=update_count,
        client_selection=client_selection,
        aggregation_metrics=aggregation_metrics,
        round_started_at=round_started_at,
        round_timing=round_timing,
    )
    return RoundExecution(
        active=next_active,
        peer_context_state=sync_states.peer_context_state,
        client_partition_sync_state=sync_states.client_partition_sync_state,
        query_ssl_algorithm_sync_state=sync_states.query_ssl_algorithm_sync_state,
        summary=summary,
    )


def _require_round_capability_plan(
    request: SimulationRunRequest,
) -> FederatedSslCapabilityPlan:
    if request.capability_plan is None:
        raise ValueError("SimulationRunRequest.capability_plan must be resolved.")
    return request.capability_plan


def _run_server_step_phase(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
    active: ActiveSimulationState,
    capability_plan: FederatedSslCapabilityPlan,
    round_index: int,
    round_timing: dict[str, float],
) -> ActiveSimulationState:
    started_at = time.perf_counter()
    server_step = server_step_execution.run_server_step_if_supported(
        request=request,
        bootstrapped=bootstrapped,
        active=active,
        capability_plan=capability_plan,
        round_index=round_index,
    )
    round_timing["round_server_step_seconds"] = time.perf_counter() - started_at
    return server_step.active


def _clear_round_base_snapshot_cache(
    *,
    bootstrapped: BootstrappedSimulation,
    round_timing: dict[str, float],
) -> None:
    started_at = time.perf_counter()
    if bootstrapped.round_base_snapshot_cache is not None:
        bootstrapped.round_base_snapshot_cache.clear()
    round_timing["round_base_snapshot_cache_clear_seconds"] = (
        time.perf_counter() - started_at
    )


def _open_round_phase(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
    ssl_method_runtime: FederatedSslSimulationRuntime,
    round_id: str,
    round_timing: dict[str, float],
) -> TrainingTask:
    started_at = time.perf_counter()
    round_record = bootstrapped.server_runtime.open_round(
        ssl_method_runtime.build_round_open_request(
            round_id=round_id,
            training_task_config=request.training_task_config,
        )
    )
    round_timing["round_open_seconds"] = time.perf_counter() - started_at
    return round_record.training_task


def _select_clients_phase(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
    capability_plan: FederatedSslCapabilityPlan,
    round_index: int,
    round_timing: dict[str, float],
) -> _ClientSelection:
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
    return _ClientSelection(
        selected_shards=selected_shards,
        participation_selection=participation_selection,
        skipped_client_ids=skipped_client_ids,
    )


def _prepare_peer_context_phase(
    *,
    capability_plan: FederatedSslCapabilityPlan,
    request: SimulationRunRequest,
    selected_shards: tuple[FederatedClientShard, ...],
    round_index: int,
    peer_context_state: PeerContextSimulationState,
    round_timing: dict[str, float],
) -> Mapping[str, object]:
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
    return peer_context_by_client


def _train_clients_phase(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
    active: ActiveSimulationState,
    ssl_method_runtime: FederatedSslSimulationRuntime,
    round_id: str,
    training_task: TrainingTask,
    capability_plan: FederatedSslCapabilityPlan,
    client_selection: _ClientSelection,
    peer_context_by_client: Mapping[str, object],
    peer_context_state: PeerContextSimulationState,
    client_partition_sync_state: ClientPartitionSyncSimulationState,
    query_ssl_algorithm_sync_state: QuerySslAlgorithmSyncSimulationState,
    round_timing: dict[str, float],
) -> tuple[ClientRoundExecution, ...]:
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
            previous_query_ssl_algorithm_state=(
                query_ssl_algorithm_sync_state.state_for_client(shard.client_id)
            ),
        )
        for shard in client_selection.selected_shards
    )
    round_timing["round_client_execution_seconds"] = time.perf_counter() - started_at
    return client_executions


def _build_sync_state_phase(
    *,
    peer_context_state: PeerContextSimulationState,
    client_partition_sync_state: ClientPartitionSyncSimulationState,
    query_ssl_algorithm_sync_state: QuerySslAlgorithmSyncSimulationState,
    client_executions: tuple[ClientRoundExecution, ...],
    round_timing: dict[str, float],
) -> _RoundSyncStates:
    started_at = time.perf_counter()
    sync_states = _RoundSyncStates(
        peer_context_state=_build_next_peer_context_state(
            previous=peer_context_state,
            client_executions=client_executions,
        ),
        client_partition_sync_state=_build_next_client_partition_sync_state(
            previous=client_partition_sync_state,
            client_executions=client_executions,
        ),
        query_ssl_algorithm_sync_state=_build_next_query_ssl_algorithm_sync_state(
            previous=query_ssl_algorithm_sync_state,
            client_executions=client_executions,
        ),
    )
    round_timing["round_peer_state_build_seconds"] = time.perf_counter() - started_at
    return sync_states


def _finalize_publication_phase(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
    round_id: str,
    next_model_revision: str,
    round_timing: dict[str, float],
) -> tuple[ActiveSimulationState, dict[str, float]]:
    started_at = time.perf_counter()
    next_active, aggregation_metrics = _finalize_round_publication(
        request=request,
        bootstrapped=bootstrapped,
        round_id=round_id,
        next_model_revision=next_model_revision,
    )
    round_timing["round_finalize_publication_seconds"] = (
        time.perf_counter() - started_at
    )
    return next_active, aggregation_metrics


def _evaluate_validation_phase(
    *,
    request: SimulationRunRequest,
    active: ActiveSimulationState,
    bootstrapped: BootstrappedSimulation,
    training_task: TrainingTask,
    round_timing: dict[str, float],
) -> SimulationEvaluation:
    started_at = time.perf_counter()
    validation = evaluate_simulation_validation(
        request=request,
        active=active,
        rows=request.validation_rows,
        objective_config=training_task.objective_config,
        runtime_resource_cache=bootstrapped.runtime_resource_cache,
    )
    round_timing["round_validation_seconds"] = time.perf_counter() - started_at
    return validation


def _assemble_round_summary(
    *,
    bootstrapped: BootstrappedSimulation,
    round_id: str,
    next_model_revision: str,
    validation: SimulationEvaluation,
    client_executions: tuple[ClientRoundExecution, ...],
    update_count: int,
    client_selection: _ClientSelection,
    aggregation_metrics: dict[str, float],
    round_started_at: float,
    round_timing: dict[str, float],
) -> SimulationRoundSummary:
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
    return SimulationRoundSummary(
        round_id=round_id,
        model_revision=next_model_revision,
        update_count=update_count,
        validation=validation,
        clients=tuple(execution.summary for execution in client_executions),
        round_time_seconds=round_elapsed,
        round_timing_breakdown=round_timing,
        aggregation_metrics=aggregation_metrics,
        total_payload_bytes=sum(
            execution.summary.client_payload_bytes or 0
            for execution in client_executions
        ),
        total_client_count=len(bootstrapped.dataset_split.client_shards),
        selected_client_count=client_selection.participation_selection.selected_count,
        skipped_client_count=client_selection.participation_selection.skipped_count,
        skipped_client_ids=client_selection.skipped_client_ids,
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


def _build_next_query_ssl_algorithm_sync_state(
    *,
    previous: QuerySslAlgorithmSyncSimulationState,
    client_executions: tuple[ClientRoundExecution, ...],
) -> QuerySslAlgorithmSyncSimulationState:
    states = dict(previous.client_algorithm_states)
    for execution in client_executions:
        if not execution.query_ssl_algorithm_state:
            continue
        states[execution.summary.client_id] = execution.query_ssl_algorithm_state
    return QuerySslAlgorithmSyncSimulationState(client_algorithm_states=states)


def _finalize_round_publication(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
    round_id: str,
    next_model_revision: str,
) -> tuple[ActiveSimulationState, dict[str, float]]:
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
    return (
        ActiveSimulationState(
            manifest=active_manifest,
            adapter_state=active_state,
        ),
        dict(finalized_round.publication.aggregated_metrics),
    )


def _record_round_transient_resource_cleanup(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
    round_timing: dict[str, float],
) -> None:
    started_at = time.perf_counter()
    removed_count = _release_transient_resources_at_round_boundary(
        request=request,
        bootstrapped=bootstrapped,
    )
    round_timing["round_transient_resource_clean_seconds"] = (
        time.perf_counter() - started_at
    )
    round_timing["round_transient_resource_removed_count"] = float(removed_count)


def _release_transient_resources_at_round_boundary(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
) -> int:
    """round 사이에 update family가 선언한 무거운 transient cache를 정리한다."""

    cleaner_path = request.round_runtime_config.transient_resource_cleaner
    removed_count = 0
    if cleaner_path:
        cleaner = _load_transient_resource_cleaner(cleaner_path)
        removed_count = int(cleaner(bootstrapped.runtime_resource_cache) or 0)
    gc.collect()
    try:
        import torch
    except ImportError:  # pragma: no cover - optional dependency guard
        return removed_count
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return removed_count


def _load_transient_resource_cleaner(cleaner_path: str) -> Any:
    return load_configured_callable(
        cleaner_path,
        field_name="round_runtime.transient_resource_cleaner",
    )
