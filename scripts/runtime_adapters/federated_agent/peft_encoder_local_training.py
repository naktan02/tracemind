"""FL simulation agent local-training runtime bridge."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.src.infrastructure.repositories.training_artifact_repository import (
    TrainingArtifactRepository,
)
from agent.src.services.training.execution.query_ssl_local_training_service import (
    QuerySslLocalTrainingService,
    QuerySslPeftEncoderLocalTrainingRequest,
)
from methods.adaptation.peft_text_encoder.config import (
    PeftEncoderTrainingBackendConfig,
)
from methods.adaptation.peft_text_encoder.federated_ssl import (
    helper_provider,
    method_owned_training,
)
from methods.adaptation.peft_text_encoder.runtime_family import (
    build_training_backend_config_for_peft_encoder_state,
    build_training_backend_for_peft_encoder_state,
)
from methods.adaptation.peft_text_encoder.training import (
    query_ssl_local_training as qssl_training,
)
from methods.adaptation.peft_text_encoder.training_backend import (
    PeftEncoderTrainingBackend,
)
from methods.adaptation.peft_text_encoder.update.delta_artifacts import (
    PeftEncoderDeltaMaterializer,
)
from methods.adaptation.peft_text_encoder.update.materialization import (
    PeftEncoderMaterializedState,
)
from methods.common.runtime_resources import RuntimeResourceCache
from methods.common.timing import TimingRecorder
from methods.federated_ssl.peer_context import (
    FederatedSslPeerClientSnapshot,
    FederatedSslPeerContext,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedLocalTrainerRuntimeConfig,
    FederatedQuerySslObjectiveConfig,
    FederatedSslMethodConfig,
)
from scripts.experiments.fl_ssl.federated_simulation.runtime_resources import (
    RoundBaseSnapshotCache,
)
from scripts.runtime_adapters.federated_agent.artifact_store import (
    SimulationClientArtifactStore,
    save_agent_local_update_payload,
)
from scripts.runtime_adapters.federated_agent.base_state_materialization import (
    load_peft_encoder_base_parameters_with_timing,
    load_peft_encoder_base_partition_parameters_with_timing,
)
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PeftClassifierState,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import TrainingTask

QuerySslPeftEncoderClientTrainingResult = (
    qssl_training.QuerySslPeftEncoderClientTrainingResult
)


def run_method_owned_peft_encoder_local_training(
    *,
    client_id: str,
    seed: int,
    output_dir: Path,
    labeled_rows: Sequence[LabeledQueryRow],
    unlabeled_rows: Sequence[LabeledQueryRow],
    diagnostic_unlabeled_rows: Sequence[LabeledQueryRow] | None = None,
    active_adapter_state: object,
    training_task: TrainingTask,
    model_manifest: ModelManifest,
    ssl_method_config: FederatedSslMethodConfig,
    local_ssl_policy_name: str,
    query_ssl_config: FederatedQuerySslObjectiveConfig | None,
    strong_view_policy: str,
    unlabeled_batch_size: int | None,
    trainer_runtime_config: FederatedLocalTrainerRuntimeConfig,
    peer_context: FederatedSslPeerContext | None = None,
    peer_snapshots: Mapping[str, FederatedSslPeerClientSnapshot] | None = None,
    peer_probe_rows: Sequence[LabeledQueryRow] | None = None,
    runtime_resource_cache: RuntimeResourceCache | None = None,
    round_base_snapshot_cache: RoundBaseSnapshotCache | None = None,
    peft_config: PeftEncoderTrainingBackendConfig | None = None,
    created_at: datetime | None = None,
    base_parameters: PeftEncoderMaterializedState | None = None,
    base_partition_parameters: (
        Mapping[str, PeftEncoderMaterializedState] | None
    ) = None,
    previous_client_partition_parameters: (
        Mapping[str, PeftEncoderMaterializedState] | None
    ) = None,
    initial_query_ssl_algorithm_state: Mapping[str, Any] | None = None,
    timing_recorder: TimingRecorder | None = None,
    persist_agent_local_update: bool = True,
) -> QuerySslPeftEncoderClientTrainingResult:
    """simulation runtime state를 선택된 method-owned PEFT encoder core에 연결한다."""

    if not isinstance(active_adapter_state, PeftClassifierState):
        raise ValueError(
            "Method-owned PEFT classifier local training requires active classifier "
            "state."
        )
    effective_created_at = created_at or datetime.now(tz=timezone.utc)
    if base_parameters is None:
        base_parameters = load_peft_encoder_base_parameters_with_timing(
            active_adapter_state=active_adapter_state,
            output_dir=output_dir,
            aggregated_at=effective_created_at,
            round_base_snapshot_cache=round_base_snapshot_cache,
            timing_recorder=timing_recorder,
        )
    if base_partition_parameters is None:
        base_partition_parameters = (
            load_peft_encoder_base_partition_parameters_with_timing(
                active_adapter_state=active_adapter_state,
                output_dir=output_dir,
                aggregated_at=effective_created_at,
                round_base_snapshot_cache=round_base_snapshot_cache,
                timing_recorder=timing_recorder,
            )
        )
    effective_peft_config = (
        peft_config
        or build_training_backend_config_for_peft_encoder_state(
            active_adapter_state=active_adapter_state,
            objective_config=training_task.objective_config,
        )
    )
    labels = tuple(str(label) for label in active_adapter_state.label_schema)
    helper_weak_probability_provider = (
        helper_provider.build_peft_encoder_helper_provider_for_local_ssl_policy(
            method_name=ssl_method_config.name,
            local_ssl_policy_name=local_ssl_policy_name,
            peer_context=peer_context,
            peer_snapshots=peer_snapshots,
            labels=labels,
            peft_config=effective_peft_config,
            trainer_runtime_config=trainer_runtime_config,
            runtime_resource_cache=runtime_resource_cache,
            timing_recorder=timing_recorder,
        )
    )
    result = method_owned_training.run_method_owned_peft_encoder_training_core(
        client_id=client_id,
        seed=seed,
        labeled_rows=labeled_rows,
        unlabeled_rows=unlabeled_rows,
        diagnostic_unlabeled_rows=diagnostic_unlabeled_rows,
        labels=labels,
        base_parameters=base_parameters,
        base_partition_parameters=base_partition_parameters,
        previous_client_partition_parameters=previous_client_partition_parameters,
        training_task=training_task,
        model_manifest=model_manifest,
        ssl_method_config=ssl_method_config,
        local_ssl_policy_name=local_ssl_policy_name,
        query_ssl_config=query_ssl_config,
        peer_context=peer_context,
        strong_view_policy=strong_view_policy,
        unlabeled_batch_size=unlabeled_batch_size,
        peft_config=effective_peft_config,
        trainer_runtime_config=trainer_runtime_config,
        created_at=effective_created_at,
        delta_materializer=PeftEncoderDeltaMaterializer(
            artifact_store=SimulationClientArtifactStore(output_dir=output_dir)
        ),
        helper_weak_probability_provider=helper_weak_probability_provider,
        peer_probe_rows=peer_probe_rows,
        runtime_resource_cache=runtime_resource_cache,
        timing_recorder=timing_recorder,
        initial_query_ssl_algorithm_state=initial_query_ssl_algorithm_state,
    )
    if persist_agent_local_update:
        save_agent_local_update_payload(
            output_dir=output_dir,
            client_id=client_id,
            update_id=result.update_envelope.update_id,
            update_payload=result.update_payload,
            timing_recorder=timing_recorder,
        )
    return result


def run_query_ssl_peft_encoder_local_training(
    *,
    client_id: str,
    seed: int,
    output_dir: Path,
    labeled_rows: Sequence[LabeledQueryRow],
    unlabeled_rows: Sequence[LabeledQueryRow],
    diagnostic_unlabeled_rows: Sequence[LabeledQueryRow] | None = None,
    active_adapter_state: object,
    training_task: TrainingTask,
    model_manifest: ModelManifest,
    query_ssl_config: FederatedQuerySslObjectiveConfig,
    trainer_runtime_config: FederatedLocalTrainerRuntimeConfig,
    peft_config: PeftEncoderTrainingBackendConfig | None = None,
    created_at: datetime | None = None,
    base_parameters: PeftEncoderMaterializedState | None = None,
    runtime_resource_cache: RuntimeResourceCache | None = None,
    round_base_snapshot_cache: RoundBaseSnapshotCache | None = None,
    timing_recorder: TimingRecorder | None = None,
    persist_agent_local_update: bool = True,
    initial_query_ssl_algorithm_state: Mapping[str, Any] | None = None,
) -> QuerySslPeftEncoderClientTrainingResult:
    """simulation runtime state를 Query SSL PEFT encoder core에 연결한다."""

    if not isinstance(active_adapter_state, PeftClassifierState):
        raise ValueError(
            "Query SSL PEFT classifier local training requires active classifier state."
        )
    effective_created_at = created_at or datetime.now(tz=timezone.utc)
    if base_parameters is None:
        base_parameters = load_peft_encoder_base_parameters_with_timing(
            active_adapter_state=active_adapter_state,
            output_dir=output_dir,
            aggregated_at=effective_created_at,
            round_base_snapshot_cache=round_base_snapshot_cache,
            timing_recorder=timing_recorder,
        )
    service = QuerySslLocalTrainingService(
        repository=TrainingArtifactRepository(
            state_root=output_dir / "agents" / client_id
        ),
        backend=PeftEncoderTrainingBackend(
            config=peft_config,
        )
        if peft_config is not None
        else build_training_backend_for_peft_encoder_state(
            active_adapter_state=active_adapter_state,
            objective_config=training_task.objective_config,
        ),
    )
    return service.run_peft_encoder(
        QuerySslPeftEncoderLocalTrainingRequest(
            client_id=client_id,
            seed=seed,
            labeled_rows=labeled_rows,
            unlabeled_rows=unlabeled_rows,
            diagnostic_unlabeled_rows=diagnostic_unlabeled_rows,
            labels=tuple(str(label) for label in active_adapter_state.label_schema),
            base_parameters=base_parameters,
            training_task=training_task,
            model_manifest=model_manifest,
            query_ssl_config=query_ssl_config,
            trainer_runtime_config=trainer_runtime_config,
            created_at=effective_created_at,
            runtime_resource_cache=runtime_resource_cache,
            timing_recorder=timing_recorder,
            persist_update_artifact=persist_agent_local_update,
            initial_query_ssl_algorithm_state=initial_query_ssl_algorithm_state,
            delta_materializer=PeftEncoderDeltaMaterializer(
                artifact_store=SimulationClientArtifactStore(output_dir=output_dir)
            ),
        )
    )
