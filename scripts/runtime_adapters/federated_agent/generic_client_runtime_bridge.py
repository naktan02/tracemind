"""FL generic client-round runtime bridge for simulation."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

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
from scripts.support.configured_callable import load_configured_callable


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

    return _run_client_round_with_adapter(
        request=request,
        bootstrapped=bootstrapped,
        active=active,
        round_id=round_id,
        shard=shard,
        training_task=training_task,
        adapter=_MethodOwnedClientObjectiveAdapter(
            capability_plan=capability_plan,
            peer_context=peer_context,
            peer_snapshots=peer_snapshots,
            previous_client_partition_parameters=previous_client_partition_parameters,
            previous_query_ssl_algorithm_state=previous_query_ssl_algorithm_state,
        ),
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
    return _run_client_round_with_adapter(
        request=request,
        bootstrapped=bootstrapped,
        active=active,
        round_id=round_id,
        shard=shard,
        training_task=training_task,
        adapter=_QuerySslClientObjectiveAdapter(
            previous_query_ssl_algorithm_state=previous_query_ssl_algorithm_state,
        ),
    )


@dataclass(frozen=True, slots=True)
class _ClientRoundContext:
    """client round 공통 lifecycle에서 공유되는 실행 문맥."""

    request: SimulationRunRequest
    bootstrapped: BootstrappedSimulation
    active: ActiveSimulationState
    round_id: str
    shard: FederatedClientShard
    training_task: Any
    timing: TimingRecorder
    diagnostic_unlabeled_rows: list[Any]
    effective_created_at: datetime
    base_parameters: Any
    delta_materializer: Any


@dataclass(frozen=True, slots=True)
class _ClientObjectiveExecution:
    """공통 submit flow가 필요로 하는 local objective 실행 결과."""

    local_result: Any
    method_diagnostics: Mapping[str, float] | None = None
    peer_client_snapshot: FederatedSslPeerClientSnapshot | None = None
    client_partition_snapshot: Mapping[str, Any] | None = None
    query_ssl_algorithm_state: Mapping[str, Any] | None = None


class _ClientObjectiveAdapter(Protocol):
    """client round lifecycle에 끼워 넣는 objective별 실행 adapter."""

    def supports(self, request: SimulationRunRequest) -> bool:
        """request가 이 objective 경로로 실행 가능한지 확인한다."""

    def run(self, context: _ClientRoundContext) -> _ClientObjectiveExecution:
        """objective별 local training을 실행하고 submit용 결과를 반환한다."""


@dataclass(frozen=True, slots=True)
class _MethodOwnedClientObjectiveAdapter:
    """FedMatch 같은 method-owned local objective 실행 adapter."""

    capability_plan: FederatedSslCapabilityPlan
    peer_context: FederatedSslPeerContext | None
    peer_snapshots: Mapping[str, FederatedSslPeerClientSnapshot] | None
    previous_client_partition_parameters: Mapping[str, Any] | None
    previous_query_ssl_algorithm_state: Mapping[str, Any] | None

    def supports(self, request: SimulationRunRequest) -> bool:
        return (
            request.ssl_method_config is not None
            and request.round_runtime_config.runtime_payload_for_update_family()
            is not None
        )

    def run(self, context: _ClientRoundContext) -> _ClientObjectiveExecution:
        request = context.request
        query_ssl_config = request.query_ssl_objective_config

        load_base_partition_parameters = _load_client_round_callable(
            request=request,
            callable_name="base_partition_state_materializer",
        )
        base_partition_parameters = load_base_partition_parameters(
            active_adapter_state=context.active.adapter_state,
            output_dir=request.output_dir,
            aggregated_at=context.effective_created_at,
            round_base_snapshot_cache=context.bootstrapped.round_base_snapshot_cache,
            timing_recorder=context.timing,
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

        run_training_core = _load_client_round_callable(
            request=request,
            callable_name="method_owned_local_training_core",
        )

        with context.timing.measure("local_training_total_seconds"):
            local_result = run_training_core(
                client_id=context.shard.client_id,
                seed=request.seed,
                labeled_rows=context.shard.labeled_rows,
                unlabeled_rows=context.shard.unlabeled_rows,
                diagnostic_unlabeled_rows=context.diagnostic_unlabeled_rows,
                active_adapter_state=context.active.adapter_state,
                training_task=context.training_task,
                model_manifest=context.active.manifest,
                ssl_method_config=request.ssl_method_config,
                local_ssl_policy_name=self.capability_plan.local_ssl_policy_name,
                query_ssl_config=request.query_ssl_objective_config,
                strong_view_policy=strong_view_policy,
                unlabeled_batch_size=unlabeled_batch_size,
                trainer_runtime_config=request.local_trainer_runtime_config,
                peer_context=self.peer_context,
                peer_snapshots=self.peer_snapshots,
                peer_probe_rows=(
                    context.bootstrapped.peer_probe_rows
                    if context.bootstrapped.peer_probe_rows
                    else tuple(request.validation_rows)
                ),
                runtime_resource_cache=context.bootstrapped.runtime_resource_cache,
                created_at=context.effective_created_at,
                base_parameters=context.base_parameters,
                base_partition_parameters=base_partition_parameters,
                previous_client_partition_parameters=(
                    self.previous_client_partition_parameters
                ),
                initial_query_ssl_algorithm_state=(
                    self.previous_query_ssl_algorithm_state
                ),
                timing_recorder=context.timing,
                delta_materializer=context.delta_materializer,
            )

        from methods.federated_ssl.diagnostics.client import (
            extract_client_method_diagnostics,
        )

        if request.ssl_method_config is None:
            raise ValueError("method-owned client round requires ssl_method_config.")
        return _ClientObjectiveExecution(
            local_result=local_result,
            method_diagnostics=extract_client_method_diagnostics(
                method_name=request.ssl_method_config.name,
                metrics=local_result.client_metrics,
            ),
            peer_client_snapshot=local_result.peer_client_snapshot,
            client_partition_snapshot=local_result.client_partition_parameters,
            query_ssl_algorithm_state=local_result.query_ssl_algorithm_state,
        )


@dataclass(frozen=True, slots=True)
class _QuerySslClientObjectiveAdapter:
    """FixMatch/SoftMatch/DASH 같은 Query SSL objective 실행 adapter."""

    previous_query_ssl_algorithm_state: Mapping[str, Any] | None

    def supports(self, request: SimulationRunRequest) -> bool:
        return (
            request.ssl_method_config is None
            and request.query_ssl_objective_config is not None
            and request.round_runtime_config.runtime_payload_for_update_family()
            is not None
        )

    def run(self, context: _ClientRoundContext) -> _ClientObjectiveExecution:
        request = context.request
        build_training_backend = _load_client_round_callable(
            request=request,
            callable_name="query_ssl_training_backend_factory",
        )
        local_training_service = build_query_ssl_local_training_service(
            client_state_root=request.output_dir / "agents" / context.shard.client_id,
            backend=build_training_backend(
                active_adapter_state=context.active.adapter_state,
                objective_config=context.training_task.objective_config,
            ),
        )

        request_factory = _load_client_round_callable(
            request=request,
            callable_name="query_ssl_request_factory",
        )
        run_query_ssl_training = _load_client_round_callable(
            request=request,
            callable_name="query_ssl_training_runner",
        )

        if not hasattr(context.active.adapter_state, "label_schema"):
            raise ValueError(
                "Query SSL local training requires active state with label_schema."
            )

        with context.timing.measure("local_training_total_seconds"):
            local_result = run_query_ssl_training(
                local_training_service=local_training_service,
                request=request_factory(
                    client_id=context.shard.client_id,
                    seed=request.seed,
                    labeled_rows=context.shard.labeled_rows,
                    unlabeled_rows=context.shard.unlabeled_rows,
                    diagnostic_unlabeled_rows=context.diagnostic_unlabeled_rows,
                    labels=tuple(
                        str(label)
                        for label in context.active.adapter_state.label_schema
                    ),
                    base_parameters=context.base_parameters,
                    training_task=context.training_task,
                    model_manifest=context.active.manifest,
                    query_ssl_config=request.query_ssl_objective_config,
                    trainer_runtime_config=request.local_trainer_runtime_config,
                    created_at=context.effective_created_at,
                    runtime_resource_cache=context.bootstrapped.runtime_resource_cache,
                    timing_recorder=context.timing,
                    persist_update_artifact=True,
                    initial_query_ssl_algorithm_state=(
                        self.previous_query_ssl_algorithm_state
                    ),
                    delta_materializer=context.delta_materializer,
                ),
            )

        return _ClientObjectiveExecution(
            local_result=local_result,
            query_ssl_algorithm_state=local_result.query_ssl_algorithm_state,
        )


def _run_client_round_with_adapter(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
    active: ActiveSimulationState,
    round_id: str,
    shard: FederatedClientShard,
    training_task: Any,
    adapter: _ClientObjectiveAdapter,
) -> ClientRoundExecution | None:
    if not adapter.supports(request):
        return None

    timing = TimingRecorder()
    training_started_at = time.perf_counter()
    context = _prepare_client_round_context(
        request=request,
        bootstrapped=bootstrapped,
        active=active,
        round_id=round_id,
        shard=shard,
        training_task=training_task,
        timing=timing,
    )
    objective_execution = adapter.run(context)
    _release_transient_model_cache_if_configured(context)
    client_train_time_seconds = time.perf_counter() - training_started_at
    _save_agent_update_payload_if_configured(
        context=context,
        local_result=objective_execution.local_result,
    )
    client_artifact_byte_counter = _load_client_round_callable(
        request=request,
        callable_name="update_artifact_byte_counter",
    )
    upload_client_update = _load_client_round_callable(
        request=request,
        callable_name="update_uploader",
    )
    return submit_local_training_result(
        bootstrapped=bootstrapped,
        round_id=round_id,
        output_dir=request.output_dir,
        client_id=shard.client_id,
        diagnostic_candidate_count=len(context.diagnostic_unlabeled_rows),
        client_train_time_seconds=client_train_time_seconds,
        timing_recorder=timing,
        local_result=objective_execution.local_result,
        upload_client_update=upload_client_update,
        client_artifact_byte_counter=client_artifact_byte_counter,
        method_diagnostics=objective_execution.method_diagnostics,
        peer_client_snapshot=objective_execution.peer_client_snapshot,
        client_partition_snapshot=objective_execution.client_partition_snapshot,
        query_ssl_algorithm_state=objective_execution.query_ssl_algorithm_state,
    )


def _prepare_client_round_context(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
    active: ActiveSimulationState,
    round_id: str,
    shard: FederatedClientShard,
    training_task: Any,
    timing: TimingRecorder,
) -> _ClientRoundContext:
    with timing.measure("diagnostic_view_select_seconds"):
        diagnostic_unlabeled_rows = build_round_diagnostic_unlabeled_rows(
            request=request,
            round_id=round_id,
            shard=shard,
        )

    effective_created_at = datetime.now(tz=timezone.utc)
    load_base_parameters = _load_client_round_callable(
        request=request,
        callable_name="base_state_materializer",
    )
    base_parameters = load_base_parameters(
        active_adapter_state=active.adapter_state,
        output_dir=request.output_dir,
        aggregated_at=effective_created_at,
        round_base_snapshot_cache=bootstrapped.round_base_snapshot_cache,
        timing_recorder=timing,
    )

    delta_materializer_factory = _load_client_round_callable(
        request=request,
        callable_name="delta_materializer_factory",
    )
    delta_materializer = delta_materializer_factory(
        artifact_store=SimulationClientArtifactStore(output_dir=request.output_dir)
    )
    return _ClientRoundContext(
        request=request,
        bootstrapped=bootstrapped,
        active=active,
        round_id=round_id,
        shard=shard,
        training_task=training_task,
        timing=timing,
        diagnostic_unlabeled_rows=diagnostic_unlabeled_rows,
        effective_created_at=effective_created_at,
        base_parameters=base_parameters,
        delta_materializer=delta_materializer,
    )


def _release_transient_model_cache_if_configured(
    context: _ClientRoundContext,
) -> None:
    if not bool(
        getattr(
            context.request.round_runtime_config,
            "release_transient_model_cache_after_client",
            False,
        )
    ):
        return
    release_transient_model_cache = _load_client_round_callable(
        request=context.request,
        callable_name="transient_model_cache_releaser",
    )
    with context.timing.measure("helper_model_cache_release_seconds"):
        release_transient_model_cache(context.bootstrapped.runtime_resource_cache)


def _save_agent_update_payload_if_configured(
    *,
    context: _ClientRoundContext,
    local_result: Any,
) -> None:
    if not context.request.artifact_persistence_config.persist_agent_local_updates:
        return
    save_agent_local_update_payload(
        output_dir=context.request.output_dir,
        client_id=context.shard.client_id,
        update_id=local_result.update_envelope.update_id,
        update_payload=local_result.update_payload,
        timing_recorder=context.timing,
    )


def _load_client_round_callable(
    *,
    request: SimulationRunRequest,
    callable_name: str,
) -> Any:
    path = _client_round_callable_path(
        request.round_runtime_config,
        callable_name,
    )
    if path is None:
        raise NotImplementedError(
            "round_runtime.client_round_runtime is missing required callable: "
            f"{callable_name}"
        )
    return load_configured_callable(
        path,
        field_name=f"round_runtime.client_round_runtime.{callable_name}",
    )


def _client_round_callable_path(
    round_runtime_config: object,
    callable_name: str,
) -> str | None:
    resolver = getattr(round_runtime_config, "client_round_callable_path", None)
    if callable(resolver):
        return resolver(callable_name)
    mapping = getattr(round_runtime_config, "client_round_runtime", {})
    if not isinstance(mapping, Mapping):
        return None
    normalized_name = callable_name.strip().lower().replace("-", "_")
    raw_value = mapping.get(normalized_name)
    if raw_value is None:
        return None
    normalized_value = str(raw_value).strip()
    return normalized_value or None
