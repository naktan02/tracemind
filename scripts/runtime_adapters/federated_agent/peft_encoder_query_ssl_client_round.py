"""FL simulation Query SSL client-round runtime bridge."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from methods.adaptation.peft_text_encoder.runtime_family import (
    is_peft_encoder_update_family,
)
from methods.adaptation.peft_text_encoder.update.delta_artifacts import (
    server_owned_peft_encoder_update_artifact_byte_count,
    upload_agent_local_peft_encoder_update,
)
from methods.common.timing import TimingRecorder
from methods.federated_ssl.capability_plan import FederatedSslCapabilityPlan
from methods.federated_ssl.peer_context import (
    FederatedSslPeerClientSnapshot,
    FederatedSslPeerContext,
)
from scripts.experiments.fl_ssl.federated_simulation.flow.state import (
    ActiveSimulationState,
    BootstrappedSimulation,
    ClientRoundExecution,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedClientShard,
    SimulationRunRequest,
)
from scripts.runtime_adapters.federated_agent.client_update_flow import (
    build_round_diagnostic_unlabeled_rows,
    submit_local_training_result,
)
from scripts.runtime_adapters.federated_agent.peft_encoder_local_training import (
    run_query_ssl_peft_encoder_local_training,
)


def run_peft_encoder_query_ssl_client_round_if_supported(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
    active: ActiveSimulationState,
    round_id: str,
    shard: FederatedClientShard,
    training_task: Any,
    capability_plan: FederatedSslCapabilityPlan | None = None,
    peer_context: FederatedSslPeerContext | None = None,
    peer_snapshots: Mapping[str, FederatedSslPeerClientSnapshot] | None = None,
    previous_client_partition_parameters: Mapping[str, Any] | None = None,
    previous_query_ssl_algorithm_state: Mapping[str, Any] | None = None,
) -> ClientRoundExecution | None:
    """Query SSL raw-row client training이 가능한 조합이면 해당 경로로 실행한다."""

    del (
        capability_plan,
        peer_context,
        peer_snapshots,
        previous_client_partition_parameters,
    )
    if not _supports_query_ssl_peft_encoder_client_training(request):
        return None
    return _run_query_ssl_peft_encoder_client_round(
        request=request,
        bootstrapped=bootstrapped,
        active=active,
        round_id=round_id,
        shard=shard,
        training_task=training_task,
        previous_query_ssl_algorithm_state=previous_query_ssl_algorithm_state,
    )


def _supports_query_ssl_peft_encoder_client_training(
    request: SimulationRunRequest,
) -> bool:
    return (
        request.ssl_method_config is None
        and request.query_ssl_objective_config is not None
        and is_peft_encoder_update_family(
            request.round_runtime_config.update_family_name
        )
        and request.round_runtime_config.runtime_payload_for_update_family() is not None
    )


def _run_query_ssl_peft_encoder_client_round(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
    active: ActiveSimulationState,
    round_id: str,
    shard: FederatedClientShard,
    training_task: Any,
    previous_query_ssl_algorithm_state: Mapping[str, Any] | None = None,
) -> ClientRoundExecution:
    if request.query_ssl_objective_config is None:
        raise ValueError("query_ssl_objective_config is required.")
    if request.round_runtime_config.runtime_payload_for_update_family() is None:
        raise ValueError("PEFT classifier runtime config is required.")

    timing = TimingRecorder()
    training_started_at = time.perf_counter()
    with timing.measure("diagnostic_view_select_seconds"):
        diagnostic_unlabeled_rows = build_round_diagnostic_unlabeled_rows(
            request=request,
            round_id=round_id,
            shard=shard,
        )
    with timing.measure("local_training_total_seconds"):
        local_result = run_query_ssl_peft_encoder_local_training(
            client_id=shard.client_id,
            seed=request.seed,
            output_dir=request.output_dir,
            labeled_rows=shard.labeled_rows,
            unlabeled_rows=shard.unlabeled_rows,
            diagnostic_unlabeled_rows=diagnostic_unlabeled_rows,
            active_adapter_state=active.adapter_state,
            training_task=training_task,
            model_manifest=active.manifest,
            query_ssl_config=request.query_ssl_objective_config,
            trainer_runtime_config=request.local_trainer_runtime_config,
            runtime_resource_cache=bootstrapped.runtime_resource_cache,
            round_base_snapshot_cache=bootstrapped.round_base_snapshot_cache,
            timing_recorder=timing,
            persist_agent_local_update=(
                request.artifact_persistence_config.persist_agent_local_updates
            ),
            initial_query_ssl_algorithm_state=previous_query_ssl_algorithm_state,
        )
    client_train_time_seconds = time.perf_counter() - training_started_at
    return submit_local_training_result(
        bootstrapped=bootstrapped,
        round_id=round_id,
        output_dir=request.output_dir,
        client_id=shard.client_id,
        diagnostic_candidate_count=len(diagnostic_unlabeled_rows),
        client_train_time_seconds=client_train_time_seconds,
        timing_recorder=timing,
        local_result=local_result,
        upload_client_update=upload_agent_local_peft_encoder_update,
        client_artifact_byte_counter=(
            server_owned_peft_encoder_update_artifact_byte_count
        ),
        query_ssl_algorithm_state=local_result.query_ssl_algorithm_state,
    )
