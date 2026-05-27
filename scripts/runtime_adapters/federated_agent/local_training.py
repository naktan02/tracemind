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
from methods.adaptation.peft_text_classifier.config import (
    LoraClassifierTrainingBackendConfig,
)
from methods.adaptation.peft_text_classifier.federated_ssl import (
    helper_provider,
    method_owned_training,
)
from methods.adaptation.peft_text_classifier.runtime_family import (
    PeftEncoderState,
    build_training_backend_config_for_peft_encoder_state,
    build_training_backend_for_peft_encoder_state,
)
from methods.adaptation.peft_text_classifier.training import (
    query_ssl_local_training as qssl_training,
)
from methods.adaptation.peft_text_classifier.training_backend import (
    PeftEncoderTrainingBackend,
)
from methods.adaptation.peft_text_classifier.update.delta_artifacts import (
    PeftEncoderDeltaMaterializer,
)
from methods.adaptation.peft_text_classifier.update.materialization import (
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
)
from scripts.runtime_adapters.federated_agent.base_state_materialization import (
    load_peft_encoder_base_parameters,
    load_peft_encoder_base_partition_parameters,
)
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LoraClassifierState,
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
    lora_config: LoraClassifierTrainingBackendConfig | None = None,
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

    if not isinstance(active_adapter_state, LoraClassifierState | PeftClassifierState):
        raise ValueError(
            "Method-owned PEFT classifier local training requires active classifier "
            "state."
        )
    effective_created_at = created_at or datetime.now(tz=timezone.utc)
    base_parameters = _load_base_parameters_if_needed(
        active_adapter_state=active_adapter_state,
        output_dir=output_dir,
        aggregated_at=effective_created_at,
        round_base_snapshot_cache=round_base_snapshot_cache,
        base_parameters=base_parameters,
        timing_recorder=timing_recorder,
    )
    if base_partition_parameters is None:
        base_partition_parameters = _load_base_partition_parameters_if_needed(
            active_adapter_state=active_adapter_state,
            output_dir=output_dir,
            aggregated_at=effective_created_at,
            round_base_snapshot_cache=round_base_snapshot_cache,
            timing_recorder=timing_recorder,
        )
    effective_lora_config = (
        lora_config
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
            lora_config=effective_lora_config,
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
        lora_config=effective_lora_config,
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
        _save_agent_local_update(
            output_dir=output_dir,
            client_id=client_id,
            result=result,
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
    lora_config: LoraClassifierTrainingBackendConfig | None = None,
    created_at: datetime | None = None,
    base_parameters: PeftEncoderMaterializedState | None = None,
    runtime_resource_cache: RuntimeResourceCache | None = None,
    round_base_snapshot_cache: RoundBaseSnapshotCache | None = None,
    timing_recorder: TimingRecorder | None = None,
    persist_agent_local_update: bool = True,
    initial_query_ssl_algorithm_state: Mapping[str, Any] | None = None,
) -> QuerySslPeftEncoderClientTrainingResult:
    """simulation runtime state를 Query SSL PEFT encoder core에 연결한다."""

    if not isinstance(active_adapter_state, LoraClassifierState | PeftClassifierState):
        raise ValueError(
            "Query SSL PEFT classifier local training requires active classifier state."
        )
    effective_created_at = created_at or datetime.now(tz=timezone.utc)
    base_parameters = _load_base_parameters_if_needed(
        active_adapter_state=active_adapter_state,
        output_dir=output_dir,
        aggregated_at=effective_created_at,
        round_base_snapshot_cache=round_base_snapshot_cache,
        base_parameters=base_parameters,
        timing_recorder=timing_recorder,
    )
    service = QuerySslLocalTrainingService(
        repository=TrainingArtifactRepository(
            state_root=output_dir / "agents" / client_id
        ),
        backend=PeftEncoderTrainingBackend(
            config=lora_config,
        )
        if lora_config is not None
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


def _load_base_parameters_if_needed(
    *,
    active_adapter_state: PeftEncoderState,
    output_dir: Path,
    aggregated_at: datetime,
    round_base_snapshot_cache: RoundBaseSnapshotCache | None,
    base_parameters: PeftEncoderMaterializedState | None,
    timing_recorder: TimingRecorder | None,
) -> PeftEncoderMaterializedState:
    if base_parameters is not None:
        return base_parameters
    if timing_recorder is None:
        return load_peft_encoder_base_parameters(
            active_adapter_state=active_adapter_state,
            output_dir=output_dir,
            aggregated_at=aggregated_at,
            round_base_snapshot_cache=round_base_snapshot_cache,
        )
    with timing_recorder.measure("adapter_base_materialization_seconds"):
        return load_peft_encoder_base_parameters(
            active_adapter_state=active_adapter_state,
            output_dir=output_dir,
            aggregated_at=aggregated_at,
            round_base_snapshot_cache=round_base_snapshot_cache,
        )


def _load_base_partition_parameters_if_needed(
    *,
    active_adapter_state: PeftEncoderState,
    output_dir: Path,
    aggregated_at: datetime,
    round_base_snapshot_cache: RoundBaseSnapshotCache | None,
    timing_recorder: TimingRecorder | None,
) -> dict[str, PeftEncoderMaterializedState]:
    if timing_recorder is None:
        return load_peft_encoder_base_partition_parameters(
            active_adapter_state=active_adapter_state,
            output_dir=output_dir,
            aggregated_at=aggregated_at,
            round_base_snapshot_cache=round_base_snapshot_cache,
        )
    with timing_recorder.measure("adapter_base_partition_materialization_seconds"):
        return load_peft_encoder_base_partition_parameters(
            active_adapter_state=active_adapter_state,
            output_dir=output_dir,
            aggregated_at=aggregated_at,
            round_base_snapshot_cache=round_base_snapshot_cache,
        )


def _save_agent_local_update(
    *,
    output_dir: Path,
    client_id: str,
    result: QuerySslPeftEncoderClientTrainingResult,
    timing_recorder: TimingRecorder | None,
) -> None:
    repository = TrainingArtifactRepository(
        state_root=output_dir / "agents" / client_id
    )
    if timing_recorder is None:
        repository.save_shared_adapter_update(
            result.update_envelope.update_id,
            result.update_payload,
        )
        return
    with timing_recorder.measure("agent_repository_save_seconds"):
        repository.save_shared_adapter_update(
            result.update_envelope.update_id,
            result.update_payload,
        )
