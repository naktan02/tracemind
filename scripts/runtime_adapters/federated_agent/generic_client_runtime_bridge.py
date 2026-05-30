"""FL generic client-round runtime bridge for simulation."""

from __future__ import annotations

import importlib
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from methods.common.timing import TimingRecorder
from methods.federated_ssl.capabilities.plan import FederatedSslCapabilityPlan
from methods.federated_ssl.hooks.peer_context import (
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
from scripts.runtime_adapters.federated_agent.artifact_store import (
    SimulationClientArtifactStore,
    save_agent_local_update_payload,
)
from scripts.runtime_adapters.federated_agent.client_update_flow import (
    build_round_diagnostic_unlabeled_rows,
    submit_local_training_result,
)
from scripts.runtime_adapters.federated_agent.training_runtime import (
    build_query_ssl_local_training_service,
)


def _load_round_runtime_module(update_family_name: str) -> Any:
    normalized = update_family_name.strip().lower().replace("-", "_")
    module_path = f"methods.adaptation.{normalized}.simulation_runtime.round_runtime"
    return importlib.import_module(module_path)


def _normalize_prefix(update_family_name: str) -> str:
    normalized = update_family_name.replace("-", "_").lower()
    return "peft_encoder" if "peft" in normalized else normalized


def run_method_owned_client_round_if_supported(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
    active: ActiveSimulationState,
    round_id: str,
    shard: FederatedClientShard,
    training_task: Any,
    capability_plan: FederatedSslCapabilityPlan,
    peer_context: FederatedSslPeerContext | None = None,
    peer_snapshots: Mapping[str, FederatedSslPeerClientSnapshot] | None = None,
    previous_client_partition_parameters: Mapping[str, Any] | None = None,
    previous_query_ssl_algorithm_state: Mapping[str, Any] | None = None,
) -> ClientRoundExecution | None:
    """method-owned client-round training이 가능한 update_family면 실행한다."""

    update_family_name = request.round_runtime_config.update_family_name
    normalized = update_family_name.strip().lower().replace("-", "_")
    try:
        runtime_module = importlib.import_module(
            f"methods.adaptation.{normalized}.update_family_runtime"
        )
    except ImportError:
        return None

    prefix = _normalize_prefix(update_family_name)
    checker_name = f"is_{prefix}_update_family"
    checker = getattr(runtime_module, checker_name, None)
    if checker is not None and not checker(update_family_name):
        return None

    if request.ssl_method_config is None:
        return None
    if request.round_runtime_config.runtime_payload_for_update_family() is None:
        return None

    m = _load_round_runtime_module(update_family_name)

    query_ssl_config = request.query_ssl_objective_config
    timing = TimingRecorder()
    training_started_at = time.perf_counter()

    with timing.measure("diagnostic_view_select_seconds"):
        diagnostic_unlabeled_rows = build_round_diagnostic_unlabeled_rows(
            request=request,
            round_id=round_id,
            shard=shard,
        )

    effective_created_at = datetime.now(tz=timezone.utc)

    # base parameters materialization (convention lookup)
    materialization_module = importlib.import_module(
        "scripts.runtime_adapters.federated_agent.base_state_materialization"
    )
    load_base_parameters = getattr(
        materialization_module, f"load_{prefix}_base_parameters_with_timing"
    )
    load_base_partition_parameters = getattr(
        materialization_module, f"load_{prefix}_base_partition_parameters_with_timing"
    )

    base_parameters = load_base_parameters(
        active_adapter_state=active.adapter_state,
        output_dir=request.output_dir,
        aggregated_at=effective_created_at,
        round_base_snapshot_cache=bootstrapped.round_base_snapshot_cache,
        timing_recorder=timing,
    )
    base_partition_parameters = load_base_partition_parameters(
        active_adapter_state=active.adapter_state,
        output_dir=request.output_dir,
        aggregated_at=effective_created_at,
        round_base_snapshot_cache=bootstrapped.round_base_snapshot_cache,
        timing_recorder=timing,
    )

    delta_module = importlib.import_module(
        f"methods.adaptation.{normalized}.update.delta_artifacts"
    )
    class_title = "".join(part.title() for part in prefix.split("_"))
    class_name = f"{class_title}DeltaMaterializer"
    delta_materializer_class = getattr(delta_module, class_name)
    delta_materializer = delta_materializer_class(
        artifact_store=SimulationClientArtifactStore(output_dir=request.output_dir)
    )

    from methods.adaptation.query_text_views.data import DEFAULT_STRONG_VIEW_POLICY

    strong_view_policy = (
        DEFAULT_STRONG_VIEW_POLICY
        if query_ssl_config is None
        else query_ssl_config.strong_view_policy
    )
    unlabeled_batch_size = (
        None if query_ssl_config is None else query_ssl_config.unlabeled_batch_size
    )

    run_training_core = getattr(m, f"run_method_owned_{prefix}_local_training_core")

    with timing.measure("local_training_total_seconds"):
        local_result = run_training_core(
            client_id=shard.client_id,
            seed=request.seed,
            labeled_rows=shard.labeled_rows,
            unlabeled_rows=shard.unlabeled_rows,
            diagnostic_unlabeled_rows=diagnostic_unlabeled_rows,
            active_adapter_state=active.adapter_state,
            training_task=training_task,
            model_manifest=active.manifest,
            ssl_method_config=request.ssl_method_config,
            local_ssl_policy_name=capability_plan.local_ssl_policy_name,
            query_ssl_config=request.query_ssl_objective_config,
            strong_view_policy=strong_view_policy,
            unlabeled_batch_size=unlabeled_batch_size,
            trainer_runtime_config=request.local_trainer_runtime_config,
            peer_context=peer_context,
            peer_snapshots=peer_snapshots,
            peer_probe_rows=(
                bootstrapped.peer_probe_rows
                if bootstrapped.peer_probe_rows
                else tuple(request.validation_rows)
            ),
            runtime_resource_cache=bootstrapped.runtime_resource_cache,
            created_at=effective_created_at,
            base_parameters=base_parameters,
            base_partition_parameters=base_partition_parameters,
            previous_client_partition_parameters=previous_client_partition_parameters,
            initial_query_ssl_algorithm_state=previous_query_ssl_algorithm_state,
            timing_recorder=timing,
            delta_materializer=delta_materializer,
        )

    with timing.measure("helper_model_cache_release_seconds"):
        m.release_transient_model_cache(bootstrapped.runtime_resource_cache)

    client_train_time_seconds = time.perf_counter() - training_started_at

    if request.artifact_persistence_config.persist_agent_local_updates:
        save_agent_local_update_payload(
            output_dir=request.output_dir,
            client_id=shard.client_id,
            update_id=local_result.update_envelope.update_id,
            update_payload=local_result.update_payload,
            timing_recorder=timing,
        )

    byte_counter_name = f"server_owned_{prefix}_update_artifact_byte_count"
    upload_func_name = f"upload_agent_local_{prefix}_update"
    client_artifact_byte_counter = getattr(delta_module, byte_counter_name)
    upload_client_update = getattr(delta_module, upload_func_name)

    from methods.federated_ssl.diagnostics.client import (
        extract_client_method_diagnostics,
    )

    return submit_local_training_result(
        bootstrapped=bootstrapped,
        round_id=round_id,
        output_dir=request.output_dir,
        client_id=shard.client_id,
        diagnostic_candidate_count=len(diagnostic_unlabeled_rows),
        client_train_time_seconds=client_train_time_seconds,
        timing_recorder=timing,
        local_result=local_result,
        upload_client_update=upload_client_update,
        client_artifact_byte_counter=client_artifact_byte_counter,
        method_diagnostics=extract_client_method_diagnostics(
            method_name=request.ssl_method_config.name,
            metrics=local_result.client_metrics,
        ),
        peer_client_snapshot=local_result.peer_client_snapshot,
        client_partition_snapshot=local_result.client_partition_parameters,
        query_ssl_algorithm_state=local_result.query_ssl_algorithm_state,
    )


def run_query_ssl_client_round_if_supported(
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
    """Query SSL client-round training이 가능한 update_family면 실행한다."""

    del (
        capability_plan,
        peer_context,
        peer_snapshots,
        previous_client_partition_parameters,
    )
    update_family_name = request.round_runtime_config.update_family_name
    normalized = update_family_name.strip().lower().replace("-", "_")
    try:
        runtime_module = importlib.import_module(
            f"methods.adaptation.{normalized}.update_family_runtime"
        )
    except ImportError:
        return None

    prefix = _normalize_prefix(update_family_name)
    checker_name = f"is_{prefix}_update_family"
    checker = getattr(runtime_module, checker_name, None)
    if checker is not None and not checker(update_family_name):
        return None

    if request.ssl_method_config is not None:
        return None
    if request.query_ssl_objective_config is None:
        return None
    if request.round_runtime_config.runtime_payload_for_update_family() is None:
        return None

    timing = TimingRecorder()
    training_started_at = time.perf_counter()

    with timing.measure("diagnostic_view_select_seconds"):
        diagnostic_unlabeled_rows = build_round_diagnostic_unlabeled_rows(
            request=request,
            round_id=round_id,
            shard=shard,
        )

    effective_created_at = datetime.now(tz=timezone.utc)

    # base parameters materialization (convention lookup)
    materialization_module = importlib.import_module(
        "scripts.runtime_adapters.federated_agent.base_state_materialization"
    )
    load_base_parameters = getattr(
        materialization_module, f"load_{prefix}_base_parameters_with_timing"
    )

    base_parameters = load_base_parameters(
        active_adapter_state=active.adapter_state,
        output_dir=request.output_dir,
        aggregated_at=effective_created_at,
        round_base_snapshot_cache=bootstrapped.round_base_snapshot_cache,
        timing_recorder=timing,
    )

    delta_module = importlib.import_module(
        f"methods.adaptation.{normalized}.update.delta_artifacts"
    )
    class_title = "".join(part.title() for part in prefix.split("_"))
    class_name = f"{class_title}DeltaMaterializer"
    delta_materializer_class = getattr(delta_module, class_name)
    delta_materializer = delta_materializer_class(
        artifact_store=SimulationClientArtifactStore(output_dir=request.output_dir)
    )

    build_training_backend = getattr(
        runtime_module, f"build_training_backend_for_{prefix}_state"
    )
    local_training_service = build_query_ssl_local_training_service(
        client_state_root=request.output_dir / "agents" / shard.client_id,
        backend=build_training_backend(
            active_adapter_state=active.adapter_state,
            objective_config=training_task.objective_config,
        ),
    )

    service_module_path = (
        "agent.src.services.training.execution.query_ssl_local_training_service"
    )
    service_module = importlib.import_module(service_module_path)
    class_title = "".join(part.title() for part in prefix.split("_"))
    request_class_name = f"QuerySsl{class_title}LocalTrainingRequest"
    request_class = getattr(service_module, request_class_name)

    if not hasattr(active.adapter_state, "label_schema"):
        raise ValueError(
            "Query SSL local training requires active state with label_schema."
        )

    with timing.measure("local_training_total_seconds"):
        local_result = local_training_service.run_peft_encoder(
            request_class(
                client_id=shard.client_id,
                seed=request.seed,
                labeled_rows=shard.labeled_rows,
                unlabeled_rows=shard.unlabeled_rows,
                diagnostic_unlabeled_rows=diagnostic_unlabeled_rows,
                labels=tuple(str(label) for label in active.adapter_state.label_schema),
                base_parameters=base_parameters,
                training_task=training_task,
                model_manifest=active.manifest,
                query_ssl_config=request.query_ssl_objective_config,
                trainer_runtime_config=request.local_trainer_runtime_config,
                created_at=effective_created_at,
                runtime_resource_cache=bootstrapped.runtime_resource_cache,
                timing_recorder=timing,
                persist_update_artifact=True,
                initial_query_ssl_algorithm_state=previous_query_ssl_algorithm_state,
                delta_materializer=delta_materializer,
            )
        )

    client_train_time_seconds = time.perf_counter() - training_started_at

    if request.artifact_persistence_config.persist_agent_local_updates:
        save_agent_local_update_payload(
            output_dir=request.output_dir,
            client_id=shard.client_id,
            update_id=local_result.update_envelope.update_id,
            update_payload=local_result.update_payload,
            timing_recorder=timing,
        )

    byte_counter_name = f"server_owned_{prefix}_update_artifact_byte_count"
    upload_func_name = f"upload_agent_local_{prefix}_update"
    client_artifact_byte_counter = getattr(delta_module, byte_counter_name)
    upload_client_update = getattr(delta_module, upload_func_name)

    return submit_local_training_result(
        bootstrapped=bootstrapped,
        round_id=round_id,
        output_dir=request.output_dir,
        client_id=shard.client_id,
        diagnostic_candidate_count=len(diagnostic_unlabeled_rows),
        client_train_time_seconds=client_train_time_seconds,
        timing_recorder=timing,
        local_result=local_result,
        upload_client_update=upload_client_update,
        client_artifact_byte_counter=client_artifact_byte_counter,
        query_ssl_algorithm_state=local_result.query_ssl_algorithm_state,
    )
