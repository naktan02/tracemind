"""FL simulation bootstrap 단계."""

from __future__ import annotations

from datetime import datetime, timezone

from methods.federated_ssl.base import FederatedSslMethodDescriptor
from scripts.experiments.fl_ssl.federated_simulation.adapters.evaluation import (
    evaluate_simulation_validation,
)
from scripts.experiments.fl_ssl.federated_simulation.adapters.sharding import (
    split_rows_for_federation,
    split_rows_into_client_shards,
)
from scripts.experiments.fl_ssl.federated_simulation.flow.state import (
    ActiveSimulationState,
    BootstrappedSimulation,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedDatasetSplit,
    SimulationRunRequest,
)
from scripts.runtime_adapters.embedding_runtime import create_embedding_adapter
from scripts.runtime_adapters.federated_server.initial_state_factory import (
    build_classifier_head_state_from_prototype_pack,
    build_initial_shared_state,
)
from scripts.runtime_adapters.federated_server.runtime import SimulationServerRuntime
from shared.src.contracts.common_types import TrainingTaskType
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.prototype_contracts import PrototypePackPayload
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.services.embedding_adapter import EmbeddingAdapter

from ..io.run_artifact_writer import RunArtifactWriter


def bootstrap_simulation(
    request: SimulationRunRequest,
    *,
    ssl_method_descriptor: FederatedSslMethodDescriptor,
) -> BootstrappedSimulation:
    """초기 shared state, prototype, manifest를 만들고 active pair로 고정한다."""

    dataset_split = _resolve_dataset_split(request)
    validation_client_shards = split_rows_into_client_shards(
        request.validation_rows,
        client_count=request.client_count,
        seed=request.seed + 1,
        shard_policy=request.shard_policy,
    )
    adapter = create_embedding_adapter(request.embedding_spec)
    embedding_dim = _resolve_bootstrap_embedding_dim(
        adapter=adapter,
        dataset_split=dataset_split,
    )
    server_runtime = SimulationServerRuntime.build(
        output_dir=request.output_dir,
        round_runtime_config=request.round_runtime_config,
        prototype_build_strategy=request.prototype_build_strategy,
        method_descriptor=ssl_method_descriptor,
    )

    initial_model_revision = "sim_rev_0000"
    initial_prototype_version = "proto_sim_0000"
    now = datetime.now(timezone.utc)
    initial_state = build_initial_shared_state(
        round_runtime_config=request.round_runtime_config,
        model_id=request.model_id,
        model_revision=initial_model_revision,
        training_scope=request.training_scope,
        embedding_dim=embedding_dim,
        labels=_category_labels(request),
        updated_at=now,
    )
    server_runtime.set_embedding_adapter(adapter)
    server_runtime.store_prototype_rebuild_input(
        rows=dataset_split.bootstrap_rows,
        embedding_spec=request.embedding_spec,
        rebuild_config=request.prototype_rebuild_config,
    )
    active_prototype = server_runtime.rebuild_reference_prototype_pack(
        adapter_state=initial_state,
        prototype_version=initial_prototype_version,
        embedding_model_id=request.model_id,
        embedding_model_revision=initial_model_revision,
        built_at=now,
    )
    initial_state = _finalize_bootstrap_shared_state(
        request=request,
        initial_state=initial_state,
        active_prototype=active_prototype,
        initial_model_revision=initial_model_revision,
        built_at=now,
    )
    run_artifact_writer = RunArtifactWriter()
    initial_state_ref = server_runtime.save_shared_adapter_state(initial_state)
    run_artifact_writer.save_prototype_pack(
        output_dir=request.output_dir,
        payload=active_prototype,
    )
    active_manifest = _build_bootstrap_manifest(
        request=request,
        initial_model_revision=initial_model_revision,
        initial_prototype_version=initial_prototype_version,
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
        prototype_pack=active_prototype,
    )
    initial_validation = evaluate_simulation_validation(
        request=request,
        adapter=adapter,
        active=active,
        rows=request.validation_rows,
        objective_config=request.training_task_config.objective_config,
    )
    return BootstrappedSimulation(
        dataset_split=dataset_split,
        validation_client_shards=validation_client_shards,
        adapter=adapter,
        server_runtime=server_runtime,
        initial_model_revision=initial_model_revision,
        initial_prototype_version=initial_prototype_version,
        initial_validation=initial_validation,
        active=active,
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


def _resolve_bootstrap_embedding_dim(
    *,
    adapter: EmbeddingAdapter,
    dataset_split: FederatedDatasetSplit,
) -> int:
    if not dataset_split.bootstrap_rows:
        raise ValueError("Bootstrap split must contain at least one row.")
    return len(adapter.embed_texts([str(dataset_split.bootstrap_rows[0]["text"])])[0])


def _finalize_bootstrap_shared_state(
    *,
    request: SimulationRunRequest,
    initial_state: SharedAdapterState,
    active_prototype: PrototypePackPayload,
    initial_model_revision: str,
    built_at: datetime,
) -> SharedAdapterState:
    if (
        request.round_runtime_config.adapter_family_name == "classifier_head"
        and getattr(request.prototype_build_strategy, "name", "") != "single"
    ):
        raise ValueError(
            "classifier_head bootstrap currently requires "
            "strategy_axes/prototype/build_strategy=single. "
            "The current bootstrap path converts one centroid per category into "
            "classifier weights and does not support multi-prototype packs yet."
        )
    if request.round_runtime_config.adapter_family_name != "classifier_head":
        return initial_state
    return build_classifier_head_state_from_prototype_pack(
        prototype_pack=active_prototype,
        model_id=request.model_id,
        model_revision=initial_model_revision,
        training_scope=request.training_scope,
        updated_at=built_at,
        logit_scale=(
            request.round_runtime_config.classifier_head_bootstrap_logit_scale
        ),
    )


def _build_bootstrap_manifest(
    *,
    request: SimulationRunRequest,
    initial_model_revision: str,
    initial_prototype_version: str,
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
        prototype_version=initial_prototype_version,
        training_scope=request.training_scope,
        training_enabled=True,
        compatible_task_types=(compatible_task_type,),
        base_model_id=request.embedding_spec.model_id,
        base_model_revision=request.embedding_spec.revision,
        notes="round_active_pair_only bootstrap manifest",
    )
