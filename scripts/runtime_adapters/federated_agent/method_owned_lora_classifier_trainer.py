"""FL simulation method-owned LoRA-classifier local trainer adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from agent.src.infrastructure.repositories.training_artifact_repository import (
    TrainingArtifactRepository,
)
from main_server.src.services.federation.rounds.aggregation.artifact_refs import (
    AggregationArtifactStore,
)
from methods.adaptation.lora_classifier.aggregation.materialization import (
    LoraClassifierMaterializedState,
    materialize_base_lora_classifier_state,
)
from methods.adaptation.lora_classifier.config import (
    LoraClassifierTrainingBackendConfig,
    build_lora_classifier_training_backend_config,
)
from methods.adaptation.lora_classifier.federated_ssl.method_owned_training import (
    run_method_owned_lora_classifier_training_core,
)
from methods.adaptation.lora_classifier.federated_ssl.peer_predictions import (
    build_lora_classifier_helper_probability_provider,
)
from methods.adaptation.lora_classifier.training.query_ssl_local_training import (
    QuerySslLoraClientTrainingResult,
)
from methods.common.runtime_resources import RuntimeResourceCache
from methods.common.timing import TimingRecorder
from methods.federated.aggregation.base import FederatedAggregationContext
from methods.federated_ssl.capability_axes import LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT
from methods.federated_ssl.peer_context import (
    FederatedSslPeerClientSnapshot,
    FederatedSslPeerContext,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedLocalTrainerRuntimeConfig,
    FederatedQuerySslObjectiveConfig,
    FederatedSslMethodConfig,
)
from scripts.runtime_adapters.federated_agent.lora_classifier_artifacts import (
    SimulationQuerySslLoraDeltaMaterializer,
)
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LoraClassifierState,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import TrainingTask


def run_method_owned_lora_classifier_local_training(
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
    lora_config: LoraClassifierTrainingBackendConfig | None = None,
    created_at: datetime | None = None,
    timing_recorder: TimingRecorder | None = None,
    persist_agent_local_update: bool = True,
) -> QuerySslLoraClientTrainingResult:
    """simulation runtime state를 선택된 method-owned LoRA core에 연결한다."""

    if not isinstance(active_adapter_state, LoraClassifierState):
        raise ValueError(
            "Method-owned LoRA local training requires active LoraClassifierState."
        )
    effective_created_at = created_at or datetime.now(tz=timezone.utc)
    if timing_recorder is None:
        base_parameters = _load_base_parameters(
            active_adapter_state=active_adapter_state,
            output_dir=output_dir,
            aggregated_at=effective_created_at,
        )
    else:
        with timing_recorder.measure("adapter_base_materialization_seconds"):
            base_parameters = _load_base_parameters(
                active_adapter_state=active_adapter_state,
                output_dir=output_dir,
                aggregated_at=effective_created_at,
            )
    effective_lora_config = (
        lora_config
        or build_lora_classifier_training_backend_config(training_task.objective_config)
    )
    labels = tuple(str(label) for label in active_adapter_state.label_schema)
    helper_weak_probability_provider = None
    if (
        local_ssl_policy_name.strip().lower().replace("-", "_")
        == LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT
    ):
        if timing_recorder is None:
            helper_weak_probability_provider = (
                build_lora_classifier_helper_probability_provider(
                    peer_context=peer_context,
                    peer_snapshots=peer_snapshots,
                    labels=labels,
                    lora_config=effective_lora_config,
                    trainer_runtime_config=trainer_runtime_config,
                    runtime_resource_cache=runtime_resource_cache,
                )
            )
        else:
            with timing_recorder.measure("adapter_helper_provider_prepare_seconds"):
                helper_weak_probability_provider = (
                    build_lora_classifier_helper_probability_provider(
                        peer_context=peer_context,
                        peer_snapshots=peer_snapshots,
                        labels=labels,
                        lora_config=effective_lora_config,
                        trainer_runtime_config=trainer_runtime_config,
                        runtime_resource_cache=runtime_resource_cache,
                    )
                )
    result = run_method_owned_lora_classifier_training_core(
        client_id=client_id,
        seed=seed,
        labeled_rows=labeled_rows,
        unlabeled_rows=unlabeled_rows,
        diagnostic_unlabeled_rows=diagnostic_unlabeled_rows,
        labels=labels,
        base_parameters=base_parameters,
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
        delta_materializer=SimulationQuerySslLoraDeltaMaterializer(
            output_dir=output_dir
        ),
        helper_weak_probability_provider=helper_weak_probability_provider,
        peer_probe_rows=peer_probe_rows,
        runtime_resource_cache=runtime_resource_cache,
        timing_recorder=timing_recorder,
    )
    repository = TrainingArtifactRepository(
        state_root=output_dir / "agents" / client_id
    )
    if persist_agent_local_update:
        if timing_recorder is None:
            repository.save_shared_adapter_update(
                result.update_envelope.update_id,
                result.update_payload,
            )
        else:
            with timing_recorder.measure("agent_repository_save_seconds"):
                repository.save_shared_adapter_update(
                    result.update_envelope.update_id,
                    result.update_payload,
                )
    return result


def _load_base_parameters(
    *,
    active_adapter_state: LoraClassifierState,
    output_dir: Path,
    aggregated_at: datetime,
) -> LoraClassifierMaterializedState:
    return materialize_base_lora_classifier_state(
        base_state=active_adapter_state,
        context=FederatedAggregationContext(
            next_model_revision=active_adapter_state.model_revision,
            aggregated_at=aggregated_at,
            artifact_loader=AggregationArtifactStore(
                state_root=output_dir / "main_server" / "aggregation_artifacts"
            ),
        ),
    )
