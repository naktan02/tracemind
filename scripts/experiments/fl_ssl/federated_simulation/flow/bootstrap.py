"""FL simulation bootstrap 단계."""

from __future__ import annotations

from datetime import datetime, timezone

from methods.adaptation.text_classifier.peft_encoder.runtime_family import (
    is_peft_encoder_adapter_family,
)
from methods.federated_ssl.base import FederatedSslMethodDescriptor
from methods.federated_ssl.capability_plan import PEER_CONTEXT_NONE
from scripts.experiments.fl_ssl.federated_simulation.adapters.evaluation import (
    evaluate_simulation_validation,
)
from scripts.experiments.fl_ssl.federated_simulation.adapters.peer_probe import (
    build_fixed_peer_probe,
)
from scripts.experiments.fl_ssl.federated_simulation.adapters.sharding import (
    split_rows_for_federation,
    split_rows_into_client_shards,
)
from scripts.experiments.fl_ssl.federated_simulation.flow.state import (
    ActiveSimulationState,
    BootstrappedSimulation,
    PeerContextSimulationState,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedClientShard,
    FederatedDatasetSplit,
    SimulationRunRequest,
)
from scripts.experiments.fl_ssl.federated_simulation.runtime_resources import (
    InMemoryRuntimeResourceCache,
    RoundBaseSnapshotCache,
)
from scripts.runtime_adapters.federated_server.initial_state_factory import (
    build_initial_shared_state,
)
from scripts.runtime_adapters.federated_server.runtime import SimulationServerRuntime
from shared.src.contracts.common_types import TrainingTaskType
from shared.src.contracts.model_contracts import ModelManifest

from ..io.resume_checkpoint import load_resume_checkpoint
from ..io.run_artifact_writer import RunArtifactWriter


def bootstrap_simulation(
    request: SimulationRunRequest,
    *,
    ssl_method_descriptor: FederatedSslMethodDescriptor | None,
) -> BootstrappedSimulation:
    """초기 shared state와 manifest를 만들고 active state로 고정한다."""

    dataset_split = _resolve_dataset_split(request)
    validation_client_shards = split_rows_into_client_shards(
        request.validation_rows,
        client_count=request.client_count,
        seed=request.seed + 1,
        shard_policy=request.shard_policy,
    )
    _require_classifier_simulation_runtime(request)
    server_runtime = SimulationServerRuntime.build(
        output_dir=request.output_dir,
        round_runtime_config=request.round_runtime_config,
        method_descriptor=ssl_method_descriptor,
        capability_plan=request.capability_plan,
    )

    runtime_resource_cache = InMemoryRuntimeResourceCache()
    if request.resume_config.enabled:
        return _resume_simulation(
            request=request,
            dataset_split=dataset_split,
            validation_client_shards=validation_client_shards,
            server_runtime=server_runtime,
            runtime_resource_cache=runtime_resource_cache,
        )

    initial_model_revision = "sim_rev_0000"
    now = datetime.now(timezone.utc)
    initial_state = build_initial_shared_state(
        round_runtime_config=request.round_runtime_config,
        model_id=request.model_id,
        model_revision=initial_model_revision,
        training_scope=request.training_scope,
        embedding_dim=0,
        labels=_category_labels(request),
        updated_at=now,
    )
    run_artifact_writer = RunArtifactWriter()
    initial_state_ref = server_runtime.save_shared_adapter_state(initial_state)
    active_manifest = _build_bootstrap_manifest(
        request=request,
        initial_model_revision=initial_model_revision,
        initial_state_ref=initial_state_ref,
        compatible_task_type=request.training_task_config.task_type,
        published_at=now,
    )
    run_artifact_writer.save_model_manifest(
        output_dir=request.output_dir,
        manifest=active_manifest,
    )
    server_runtime.activate_manifest(active_manifest)

    active = ActiveSimulationState(
        manifest=active_manifest,
        adapter_state=initial_state,
    )
    initial_validation = evaluate_simulation_validation(
        request=request,
        active=active,
        rows=request.validation_rows,
        objective_config=request.training_task_config.objective_config,
        runtime_resource_cache=runtime_resource_cache,
    )
    peer_probe_rows, peer_probe_manifest = build_fixed_peer_probe(
        rows=request.validation_rows,
        config=request.peer_probe_config,
        run_seed=request.seed,
    )
    return BootstrappedSimulation(
        dataset_split=dataset_split,
        validation_client_shards=validation_client_shards,
        server_runtime=server_runtime,
        initial_model_revision=initial_model_revision,
        initial_validation=initial_validation,
        active=active,
        peer_probe_rows=peer_probe_rows,
        peer_probe_manifest=peer_probe_manifest,
        runtime_resource_cache=runtime_resource_cache,
        round_base_snapshot_cache=RoundBaseSnapshotCache(),
    )


def _resume_simulation(
    *,
    request: SimulationRunRequest,
    dataset_split: FederatedDatasetSplit,
    validation_client_shards: tuple[FederatedClientShard, ...],
    server_runtime: SimulationServerRuntime,
    runtime_resource_cache: InMemoryRuntimeResourceCache,
) -> BootstrappedSimulation:
    """마지막 완료 round의 active state에서 simulation을 재개한다."""

    checkpoint = load_resume_checkpoint(request.output_dir)
    _require_resume_supported(request)
    server_runtime.archive_incomplete_round_for_resume(
        next_round_index=checkpoint.completed_round_count + 1
    )
    active_manifest = server_runtime.activate_manifest_revision(
        (
            checkpoint.rounds[-1].model_revision
            if checkpoint.rounds
            else checkpoint.initial_model_revision
        ),
    )
    active = ActiveSimulationState(
        manifest=active_manifest,
        adapter_state=server_runtime.load_active_state(active_manifest),
    )
    peer_probe_rows, peer_probe_manifest = build_fixed_peer_probe(
        rows=request.validation_rows,
        config=request.peer_probe_config,
        run_seed=request.seed,
    )
    return BootstrappedSimulation(
        dataset_split=dataset_split,
        validation_client_shards=validation_client_shards,
        server_runtime=server_runtime,
        initial_model_revision=checkpoint.initial_model_revision,
        initial_validation=checkpoint.initial_validation,
        active=active,
        completed_rounds=checkpoint.rounds,
        peer_context_state=PeerContextSimulationState(),
        peer_probe_rows=peer_probe_rows,
        peer_probe_manifest=peer_probe_manifest,
        runtime_resource_cache=runtime_resource_cache,
        round_base_snapshot_cache=RoundBaseSnapshotCache(),
    )


def _require_resume_supported(request: SimulationRunRequest) -> None:
    capability_plan = request.capability_plan
    if capability_plan is None:
        return
    if capability_plan.peer_context_policy_name != PEER_CONTEXT_NONE:
        raise ValueError(
            "FL SSL round resume currently supports peer_context_policy=none only. "
            "Peer helper snapshots are in-memory state and must be persisted as "
            "artifact refs before resuming peer-context methods."
        )


def _resolve_dataset_split(request: SimulationRunRequest) -> FederatedDatasetSplit:
    if request.materialized_dataset_split is not None:
        return request.materialized_dataset_split
    return split_rows_for_federation(
        request.train_rows,
        bootstrap_ratio=request.bootstrap_ratio,
        client_count=request.client_count,
        seed=request.seed,
        shard_policy=request.shard_policy,
        client_pool_split_config=request.client_pool_split_config,
    )


def _category_labels(request: SimulationRunRequest) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(row["mapped_label_4"])
                for row in (*request.train_rows, *request.validation_rows)
            }
        )
    )


def _require_classifier_simulation_runtime(request: SimulationRunRequest) -> None:
    """현재 FL SSL simulation은 PEFT-backed classifier 경로만 지원한다."""

    adapter_family_name = request.round_runtime_config.adapter_family_name
    if not is_peft_encoder_adapter_family(adapter_family_name):
        raise ValueError(
            "FL SSL simulation no longer wires embedding/prototype scoring. "
            "Use round_runtime.adapter_family_name=peft_classifier or "
            "lora_classifier."
        )
    if request.validation_config.scorer_backend_name != "lora_classifier_eval":
        raise ValueError(
            "FL SSL simulation validation must use lora_classifier_eval after "
            "embedding/prototype scorer removal."
        )


def _build_bootstrap_manifest(
    *,
    request: SimulationRunRequest,
    initial_model_revision: str,
    initial_state_ref: str,
    compatible_task_type: TrainingTaskType,
    published_at: datetime,
) -> ModelManifest:
    return ModelManifest(
        schema_version="model_manifest.v1",
        model_id=request.model_id,
        model_revision=initial_model_revision,
        published_at=published_at,
        artifact_kind="shared_adapter_state",
        artifact_ref=initial_state_ref,
        training_scope=request.training_scope,
        training_enabled=True,
        compatible_task_types=(compatible_task_type,),
        base_model_id=request.embedding_spec.model_id,
        base_model_revision=request.embedding_spec.revision,
        notes="round_active_pair_only bootstrap manifest",
    )
